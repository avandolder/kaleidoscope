#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//----- LEXER

// The lexer returns token [0-255] if it is an unkown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
};

static std::string identifier_str; // Filled in if tok_identifier
static double num_val;             // Filled in if tok_number

// gettok - Return the next token from stdin
static int gettok() {
  static int last_char = ' ';

  // Skip any whitespace.
  while (std::isspace(last_char)) {
    last_char = std::getchar();
  }

  if (std::isalpha(last_char)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    identifier_str = last_char;
    while (std::isalnum((last_char = std::getchar()))) {
      identifier_str += last_char;
    }

    if (identifier_str == "def") {
      return tok_def;
    } else if (identifier_str == "extern") {
      return tok_extern;
    }
    return tok_identifier;
  }
  
  if (std::isdigit(last_char) || last_char == '.') { // Number:: [0-9.]+
    std::string num_str;
    do {
      num_str += last_char;
      last_char = std::getchar();
    } while (std::isdigit(last_char) || last_char == '.');

    num_val = std::strtod(num_str.c_str(), nullptr);
    return tok_number;
  }

  if (last_char == '#') {
    // Comment until end of line.
    do {
      last_char = std::getchar();
    } while (last_char != EOF && last_char != '\n' && last_char != '\r');

    if (last_char != EOF) {
      return gettok();
    }
  }

  // Check for end of file. Don't eat the EOF. 
  if (last_char == EOF) {
    return tok_eof;
  }

  // Otherwise, just return the character as its ascii value.
  int this_char = last_char;
  last_char = getchar();
  return this_char;
}

//----- PARSER

// ExprAST - base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() {}
  virtual Value *codegen() = 0;
};

class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(double val) : val(val) {}
  virtual Value *codegen();
};

class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(const std::string &name) : name(name) {}
  virtual Value *codegen();
};

class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
    : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  virtual Value *codegen();
};

class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
    : callee(callee), args(std::move(args)) {}
  virtual Value *codegen();
};

class PrototypeAST {
  std::string name;
  std::vector<std::string> args;

public:
  PrototypeAST(const std::string &name, std::vector<std::string> args)
    : name(name), args(std::move(args)) {}
  virtual Function *codegen();

  const std::string &get_name() const { return name; }
};

class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprAST> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprAST> body)
    : proto(std::move(proto)), body(std::move(body)) {}
  virtual Function *codegen();
};

static int cur_tok;
static int get_next_token() {
  return cur_tok = gettok();
}

std::unique_ptr<ExprAST> log_error(const char *str) {
  std::fprintf(stderr, "LogError: %s\n", str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> log_error_p(const char *str) {
  log_error(str);
  return nullptr;
}

/// binop_precedence - This holds the precedence for each binary operator
/// that is defined.
static std::map<char, int> binop_precedence;

/// get_tok_precedence - Get the precedence of the pending binary operator
/// token.
static int get_tok_precedence() {
  if (!isascii(cur_tok)) {
    return -1;
  }

  // Make sure it's a declared binop.
  int tok_prec = binop_precedence[cur_tok];
  if (tok_prec <= 0) {
    return -1;
  }
  return tok_prec;
}

static std::unique_ptr<ExprAST> parse_expression();

// numberexpr ::= number
static std::unique_ptr<ExprAST> parse_number_expr() {
  auto result = llvm::make_unique<NumberExprAST>(num_val);
  get_next_token();
  return std::move(result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> parse_paren_expr() {
  get_next_token();
  auto v = parse_expression();
  if (!v) {
    return nullptr;
  }

  if (cur_tok != ')') {
    return log_error("expected ')'");
  }
  get_next_token();
  return v;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> parse_identifier_expr() {
  std::string id_name = identifier_str;

  get_next_token();

  if (cur_tok != '(') {
    return llvm::make_unique<VariableExprAST>(id_name);
  }

  get_next_token();
  std::vector<std::unique_ptr<ExprAST>> args;
  if (cur_tok != ')') {
    for (;;) {
      if (auto arg = parse_expression()) {
        args.push_back(std::move(arg));
      } else {
        return nullptr;
      }

      if (cur_tok == ')') {
        break;
      }

      if (cur_tok != ',') {
        return log_error("expected ')' or ',' in argument list");
      }
      get_next_token();
    }
  }

  get_next_token();

  return llvm::make_unique<CallExprAST>(id_name, std::move(args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static std::unique_ptr<ExprAST> parse_primary() {
  switch (cur_tok) {
    default:
      return log_error("unknown token when expecting an expression");
    case tok_identifier:
      return parse_identifier_expr();
    case tok_number:
      return parse_number_expr();
    case '(':
      return parse_paren_expr();
  }
}

/// binorphs
///   ::= ('+' primary)*
static std::unique_ptr<ExprAST> parse_binop_rhs(int expr_prec,
                                                std::unique_ptr<ExprAST> lhs) {
  // If this is a binop, find its precedence.
  for (;;) {
    int tok_prec = get_tok_precedence();

    // If this is a binop that binds at least as tightly as the current
    // binop, consume it, otherwise we are done.
    if (tok_prec < expr_prec) {
      return lhs;
    }

    // Okay, we know this is a binop.
    int binop = cur_tok;
    get_next_token();

    auto rhs = parse_primary();
    if (!rhs) {
      return nullptr;
    }

    // If binop binds less tightly with rhs than the operator after rhs,
    // let the pending operator take rhs as its lhs.
    int next_prec = get_tok_precedence();
    if (tok_prec < next_prec) {
      rhs = parse_binop_rhs(tok_prec + 1, std::move(rhs));
      if (!rhs) {
        return nullptr;
      }
    }

    // Merge lhs/rhs.
    lhs = llvm::make_unique<BinaryExprAST>(
        binop, std::move(lhs), std::move(rhs));
  }
}

/// expression
///   ::= primary binoprhs
static std::unique_ptr<ExprAST> parse_expression() {
  auto lhs = parse_primary();
  if (!lhs) {
    return nullptr;
  }
  return parse_binop_rhs(0, std::move(lhs));
}

/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> parse_prototype() {
  if (cur_tok != tok_identifier) {
    return log_error_p("expected function name in prototype");
  }

  std::string fn_name = identifier_str;
  get_next_token();

  if (cur_tok != '(') {
    return log_error_p("expected '(' in prototype");
  }

  // Read the list of argument names.
  std::vector<std::string> arg_names;
  while (get_next_token() == tok_identifier) {
    arg_names.push_back(identifier_str);
  }
  if (cur_tok != ')') {
    return log_error_p("expected ')' in prototype");
  }

  // Success.
  get_next_token();

  return llvm::make_unique<PrototypeAST>(fn_name, std::move(arg_names));
}

// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> parse_definition() {
  get_next_token();
  auto proto = parse_prototype();
  if (!proto) {
    return nullptr;
  }

  if (auto e = parse_expression()) {
    return llvm::make_unique<FunctionAST>(std::move(proto), std::move(e));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> parse_extern() {
  get_next_token();
  return parse_prototype();
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> parse_top_level_expr() {
  if (auto e = parse_expression()) {
    // Make an anonymous proto.
    auto proto =
      llvm::make_unique<PrototypeAST>("", std::vector<std::string>());
    return llvm::make_unique<FunctionAST>(std::move(proto), std::move(e));
  }
  return nullptr;
}

//----- CODE GEN
static LLVMContext context;
static IRBuilder<> builder(context);
static std::unique_ptr<Module> module;
static std::map<std::string, Value*> named_values;

Value *log_error_v(const char *str) {
  log_error(str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
  return ConstantFP::get(context, APFloat(val));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *v = named_values[name];
  if (!v) {
    log_error_v("Unknown variable name");
  }
  return v;
}

Value *BinaryExprAST::codegen() {
  Value *l = lhs->codegen();
  Value *r = rhs->codegen();
  if (!l || !r) {
    return nullptr;
  }

  switch (op) {
  case '+':
    return builder.CreateFAdd(l, r, "addtmp");
  case '-':
    return builder.CreateFSub(l, r, "subtmp");
  case '*':
    return builder.CreateFMul(l, r, "multmp");
  case '<':
    l = builder.CreateFCmpULT(l, r, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return builder.CreateUIToFP(l, Type::getDoubleTy(context), "booltmp");
  default:
    return log_error_v("invalid binary operator");
  }
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *callee_fn = module->getFunction(callee);
  if (!callee_fn) {
    return log_error_v("Unknown function referenced");
  }

  // If argument mismatch error.
  if (callee_fn->arg_size() != args.size()) {
    return log_error_v("Incorrect # of arguments passed");
  }

  std::vector<Value *> argsv;
  for (unsigned i = 0, e = args.size(); i != e; ++i) {
    argsv.push_back(args[i]->codegen());
    if (!argsv.back()) {
      return nullptr;
    }
  }

  return builder.CreateCall(callee_fn, argsv, "calltmp");
}

Function *PrototypeAST::codegen() {
  // Make the function type: double(double,double) etc.
  std::vector<Type*> doubles(args.size(), Type::getDoubleTy(context));
  FunctionType *ft =
    FunctionType::get(Type::getDoubleTy(context), doubles, false);
  Function *f =
    Function::Create(ft, Function::ExternalLinkage, name, module.get());

  // Set names for all arguments.
  unsigned idx = 0;
  for (auto &arg: f->args()) {
    arg.setName(args[idx++]);
  }

  return f;
}

Function *FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' decl.
  Function *fn = module->getFunction(proto->get_name());
  
  if (!fn) {
    fn = proto->codegen();
  }
  if (!fn) {
    return nullptr;
  }
  if (!fn->empty()) {
    return (Function*) log_error_v("Function cannot be redefined.");
  }

  // Create a new basic block to start insertion into.
  BasicBlock *bb = BasicBlock::Create(context, "entry", fn);
  builder.SetInsertPoint(bb);

  // Record the function arguments in the named_values map.
  named_values.clear();
  for (auto &arg: fn->args()) {
    named_values[arg.getName()] = &arg;
  }

  if (Value *ret_val = body->codegen()) {
    // Finish off the function.
    builder.CreateRet(ret_val);

    // Validate the generate code, checking for consistency.
    verifyFunction(*fn);

    return fn;
  }

  // Error reading body, remove function.
  fn->eraseFromParent();
  return nullptr;
}

//----- JIT DRIVER

static void handle_definition() {
  if (auto fn_ast = parse_definition()) {
    if (auto *fn_ir = fn_ast->codegen()) {
      std::fprintf(stderr, "Read function definition:");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    get_next_token();
  }
}

static void handle_extern() {
  if (auto proto_ast = parse_extern()) {
    if (auto *fn_ir = proto_ast->codegen()) {
      std::fprintf(stderr, "Read extern: ");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    get_next_token();
  }
}

static void handle_top_level_expression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto fn_ast = parse_top_level_expr()) {
    if (auto *fn_ir = fn_ast->codegen()) {
      std::fprintf(stderr, "Read top-level expr:");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    get_next_token();
  }
}

/// top ::= definition | external | expression | ';'
static void main_loop() {
  for (;;) {
    std::fprintf(stderr, "ready> ");
    switch (cur_tok) {
    case tok_eof:
      return;
    case ';':
      get_next_token();
      break;
    case tok_def:
      handle_definition();
    case tok_extern:
      handle_extern();
      break;
    default:
      handle_top_level_expression();
      break;
    }
  }
}

int main() {
  // Install standard binary operators.
  // 1 is lowest precedence.
  binop_precedence['<'] = 10;
  binop_precedence['+'] = 20;
  binop_precedence['-'] = 20;
  binop_precedence['*'] = 40;

  std::fprintf(stderr, "ready> ");
  get_next_token();

  // Make the module, which holds all the code.
  module = llvm::make_unique<Module>("my cool jit", context);

  main_loop();

  // Print out all generate code.
  module->print(errs(), nullptr);

  return 0;
}
