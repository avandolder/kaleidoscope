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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include "KaleidoscopeJIT.h"

using namespace llvm;
using namespace llvm::orc;

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

  // control
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9,
  tok_in = -10,
  tok_binary = -11,
  tok_unary = -12,
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
    } else if (identifier_str == "if") {
      return tok_if;
    } else if (identifier_str == "then") {
      return tok_then;
    } else if (identifier_str == "else") {
      return tok_else;
    } else if (identifier_str == "for") {
      return tok_for;
    } else if (identifier_str == "in") {
      return tok_in;
    } else if (identifier_str == "binary") {
      return tok_binary;
    } else if (identifier_str == "unary") {
      return tok_unary;
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
  Value *codegen() override;
};

class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(const std::string &name) : name(name) {}
  Value *codegen() override;
};

class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
    : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  Value *codegen() override;
};

class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
    : callee(callee), args(std::move(args)) {}
  Value *codegen() override;
};

class PrototypeAST {
  std::string name;
  std::vector<std::string> args;
  bool is_operator;
  unsigned precedence; // Precedence if a binary op

public:
  PrototypeAST(const std::string &name, std::vector<std::string> args,
               bool is_operator = false, unsigned prec = 0)
    : name(name), args(std::move(args)), is_operator(is_operator),
      precedence(prec) {}

  Function *codegen();
  const std::string &get_name() const { return name; }

  bool is_unary_op() const { return is_operator && args.size() == 1; }
  bool is_binary_op() const { return is_operator && args.size() == 2; }

  char get_operator_name() const {
    assert(is_unary_op() || is_binary_op());
    return name[name.size() - 1];
  }

  unsigned get_binary_precedence() const { return precedence; }
};

class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprAST> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprAST> body)
    : proto(std::move(proto)), body(std::move(body)) {}
  Function *codegen();
};

class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> cond, thenexpr, elseexpr;

public:
  IfExprAST(std::unique_ptr<ExprAST> cond,
            std::unique_ptr<ExprAST> thenexpr,
            std::unique_ptr<ExprAST> elseexpr)
    : cond(std::move(cond)), thenexpr(std::move(thenexpr)),
      elseexpr(std::move(elseexpr)) {}

  Value *codegen() override;
};

class ForExprAST : public ExprAST {
  std::string var_name;
  std::unique_ptr<ExprAST> start, end, step, body;

public:
  ForExprAST(const std::string &var_name, std::unique_ptr<ExprAST> start,
             std::unique_ptr<ExprAST> end, std::unique_ptr<ExprAST> step,
             std::unique_ptr<ExprAST> body)
    : var_name(var_name), start(std::move(start)), end(std::move(end)),
      step(std::move(step)), body(std::move(body)) {}

  Value *codegen() override;
};

class UnaryExprAST : public ExprAST {
  char opcode;
  std::unique_ptr<ExprAST> operand;

public:
  UnaryExprAST(char opcode, std::unique_ptr<ExprAST> operand)
    : opcode(opcode), operand(std::move(operand)) {}

  Value *codegen() override;
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

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> parse_if_expr() {
  get_next_token(); // eat the if

  // condition
  auto cond = parse_expression();
  if (!cond) {
    return nullptr;
  }

  if (cur_tok != tok_then) {
    return log_error("expected then");
  }
  get_next_token(); // eat the then

  auto thenexpr = parse_expression();
  if (!thenexpr) {
    return nullptr;
  }

  if (cur_tok != tok_else) {
    return log_error("expected else");
  }

  get_next_token();

  auto elseexpr = parse_expression();
  if (!elseexpr) {
    return nullptr;
  }

  return llvm::make_unique<IfExprAST>(
      std::move(cond), std::move(thenexpr), std::move(elseexpr));
}

static std::unique_ptr<ExprAST> parse_for_expr() {
  get_next_token();

  if (cur_tok != tok_identifier) {
    return log_error("expected identifier after for");
  }

  std::string id_name = identifier_str;
  get_next_token();

  if (cur_tok != '=') {
    return log_error("expected '=' after for");
  }
  get_next_token();

  auto start = parse_expression();
  if (!start) {
    return nullptr;
  }
  if (cur_tok != ',') {
    return log_error("expected ',' after for start value");
  }
  get_next_token();

  auto end = parse_expression();
  if (!end) {
    return nullptr;
  }

  // The step value is optional.
  std::unique_ptr<ExprAST> step;
  if (cur_tok == ',') {
    get_next_token();
    step = parse_expression();
    if (!step) {
      return nullptr;
    }
  }

  if (cur_tok != tok_in) {
    return log_error("expected 'in' after for");
  }
  get_next_token();

  auto body = parse_expression();
  if (!body) {
    return nullptr;
  }

  return llvm::make_unique<ForExprAST>(id_name, std::move(start),
      std::move(end), std::move(step), std::move(body));
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
    case tok_if:
      return parse_if_expr();
    case tok_for:
      return parse_for_expr();
  }
}

static std::unique_ptr<ExprAST> parse_unary() {
  // If the current token is not an operaotr, it must be a primary expr.
  if (!isascii(cur_tok) || cur_tok == '(' || cur_tok == ',') {
    return parse_primary();
  }

  // If this is a unary operator, read it.
  int opcode = cur_tok;
  get_next_token();
  if (auto operand = parse_unary()) {
    return llvm::make_unique<UnaryExprAST>(opcode, std::move(operand));
  }
  return nullptr;
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

    auto rhs = parse_unary();
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
  auto lhs = parse_unary();
  if (!lhs) {
    return nullptr;
  }
  return parse_binop_rhs(0, std::move(lhs));
}

/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> parse_prototype() {
  std::string fn_name;

  unsigned kind = 0; // 0 = identifier, 1 = unary, 2 = binary
  unsigned binary_prec = 30;

  switch (cur_tok) {
    default:
      return log_error_p("Expected function name in prototype");
    case tok_identifier:
      fn_name = identifier_str;
      kind = 0;
      get_next_token();
      break;
    case tok_unary:
      get_next_token();
      if (!isascii(cur_tok)) {
        return log_error_p("expected unary operator");
      }
      fn_name = "unary";
      fn_name += (char) cur_tok;
      kind = 1;
      get_next_token();
      break;
    case tok_binary:
      get_next_token();
      if (!isascii(cur_tok)) {
        return log_error_p("expected binary operator");
      }
      fn_name = "binary";
      fn_name += (char) cur_tok;
      kind = 2;
      get_next_token();

      // Read the precedence if present.
      if (cur_tok == tok_number) {
        if (num_val < 1 || num_val > 100) {
          return log_error_p("invalid precedence: must be 1..100");
        }
        binary_prec = (unsigned)num_val;
        get_next_token();
      }
      break;
  }

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

  if (kind && arg_names.size() != kind) {
    return log_error_p("invalid number of operands for operator");
  }

  return llvm::make_unique<PrototypeAST>(fn_name, std::move(arg_names),
      kind != 0, binary_prec);
}

// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> parse_definition() {
  get_next_token(); // eat def.
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
    auto proto = llvm::make_unique<PrototypeAST>("__anon_expr",
                                                 std::vector<std::string>());
    return llvm::make_unique<FunctionAST>(std::move(proto), std::move(e));
  }
  return nullptr;
}

//----- CODE GEN
static LLVMContext context;
static IRBuilder<> builder(context);
static std::unique_ptr<Module> module;
static std::map<std::string, Value*> named_values;
static std::unique_ptr<legacy::FunctionPassManager> fpm;
static std::unique_ptr<KaleidoscopeJIT> jit;
static std::map<std::string, std::unique_ptr<PrototypeAST>> fn_protos;

Value *log_error_v(const char *str) {
  log_error(str);
  return nullptr;
}

Function *get_function(std::string name) {
  if (auto *f = module->getFunction(name)) {
    return f;
  }

  auto fi = fn_protos.find(name);
  if (fi != fn_protos.end()) {
    return fi->second->codegen();
  }

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

Value *UnaryExprAST::codegen() {
  Value *operandv = operand->codegen();
  if (!operandv) {
    return nullptr;
  }

  Function *fn = get_function(std::string("unary") + opcode);
  if (!fn) {
    return log_error_v("unknown unary operator");
  }

  return builder.CreateCall(fn, operandv, "unop");
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
    break;
  }

  // If it wasn't a builtin binary operator, it must be a user-defined one.
  // Emit a call to it.
  Function *fn = get_function(std::string("binary") + op);
  assert(fn && "binary operator not found!");

  Value *ops[2] = {l, r};
  return builder.CreateCall(fn, ops, "binop");
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *callee_fn = get_function(callee);
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
  // Transfer ownership of the prototype to the fn_protos map, but keep a
  // reference to it for use below.
  auto &p = *proto;
  fn_protos[proto->get_name()] = std::move(proto);
  Function *fn = get_function(p.get_name());
  if (!fn) {
    return nullptr;
  }
  if (!fn->empty()) {
    return (Function*) log_error_v("Function cannot be redefined.");
  }

  // If this is an operator, install it.
  if (p.is_binary_op()) {
    binop_precedence[p.get_operator_name()] = p.get_binary_precedence();
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

    // Optimize the function.
    fpm->run(*fn);

    return fn;
  }

  // Error reading body, remove function.
  fn->eraseFromParent();
  return nullptr;
}

Value *IfExprAST::codegen() {
  Value *condv = cond->codegen();
  if (!condv) {
    return nullptr;
  }

  // Convert condition to a bool by comparing to 0.0.
  condv = builder.CreateFCmpONE(
      condv, ConstantFP::get(context, APFloat(0.0)), "ifcond");

  Function *fn = builder.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases, insert 'then' block at the end
  // of the function.
  BasicBlock *thenbb = BasicBlock::Create(context, "then", fn);
  BasicBlock *elsebb = BasicBlock::Create(context, "else");
  BasicBlock *mergebb = BasicBlock::Create(context, "ifcont");
  builder.CreateCondBr(condv, thenbb, elsebb);

  // Emit then value.
  builder.SetInsertPoint(thenbb);

  Value *thenv = thenexpr->codegen();
  if (!thenv) {
    return nullptr;
  }

  builder.CreateBr(mergebb);
  // Codegen of the 'then' can change the current block, update thenbb for PHI.
  thenbb = builder.GetInsertBlock();

  // Emit else block.
  fn->getBasicBlockList().push_back(elsebb);
  builder.SetInsertPoint(elsebb);

  Value *elsev = elseexpr->codegen();
  if (!elsev) {
    return nullptr;
  }

  builder.CreateBr(mergebb);
  // codegen of 'Else' can change the current block, update elsebb for PHI.
  elsebb = builder.GetInsertBlock();

  // Emit merge block.
  fn->getBasicBlockList().push_back(mergebb);
  builder.SetInsertPoint(mergebb);
  PHINode *pn = builder.CreatePHI(Type::getDoubleTy(context), 2, "iftmp");

  pn->addIncoming(thenv, thenbb);
  pn->addIncoming(elsev, elsebb);
  return pn;
}

Value *ForExprAST::codegen() {
  // Emit the start code first, without 'variable' in scope.
  Value *startv = start->codegen();
  if (!startv) {
    return nullptr;
  }

  // Make the new basic block for the loop header, inserting after cur block.
  Function *fn = builder.GetInsertBlock()->getParent();
  BasicBlock *preheaderbb = builder.GetInsertBlock();
  BasicBlock *loopbb = BasicBlock::Create(context, "loop", fn);

  // Insert an explicit fall through from the current block to the loopbb.
  builder.CreateBr(loopbb);
  // Start insertion in loopbb.
  builder.SetInsertPoint(loopbb);
  // Start PHI nmode with an entry for start.
  PHINode *variable = builder.CreatePHI(Type::getDoubleTy(context), 2,
      var_name.c_str());
  variable->addIncoming(startv, preheaderbb);

  // Within the loop, the variable if defined equal to the PHI node. If it
  // shadows an existing variable, we have to restore it, so save it now.
  Value *old_val = named_values[var_name];
  named_values[var_name] = variable;

  // Emit the body of the loop. This, like any expr, can change the current
  // BB. Note that we ignore the value computed by the body, but don't allow
  // an error.
  if (!body->codegen()) {
    return nullptr;
  }

  // Emit the step value.
  Value *stepv = nullptr;
  if (step) {
    stepv = step->codegen();
    if (!stepv) {
      return nullptr;
    }
  } else {
    // If not specified, use 1.0.
    stepv = ConstantFP::get(context, APFloat(1.0));
  }

  Value *next_var = builder.CreateFAdd(variable, stepv, "nextvar");

  // Compute the end condition.
  Value *end_cond = end->codegen();
  if (!end_cond) {
    return nullptr;
  }

  // Convert condition to a bool by comparing non-equal to 0.0.
  end_cond = builder.CreateFCmpONE(
      end_cond, ConstantFP::get(context, APFloat(0.0)), "loopcond");
  
  // Create the "after loop" block and insert it.
  BasicBlock *loop_endbb = builder.GetInsertBlock();
  BasicBlock *afterbb = BasicBlock::Create(context, "afterloop", fn);
  // Insert conditional branch into the end of loop_endbb.
  builder.CreateCondBr(end_cond, loopbb, afterbb);
  // Any new code will be inserted in afterbb.
  builder.SetInsertPoint(afterbb);

  // Add a new entry to the PHI node for the backedge.
  variable->addIncoming(next_var, loop_endbb);

  // Restore the unshadowed variable.
  if (old_val) {
    named_values[var_name] = old_val;
  } else {
    named_values.erase(var_name);
  }

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(context));
}

//----- OPTIMIZATION
void init_module_and_pass_manager() {
  // Open a new module.
  module = llvm::make_unique<Module>("my cool jit", context);
  module->setDataLayout(jit->getTargetMachine().createDataLayout());

  // Create a new pass manager attached to it.
  fpm = llvm::make_unique<legacy::FunctionPassManager>(module.get());

  // Do simple "peephole" optimizations and bit-twiddling optzns.
  fpm->add(createInstructionCombiningPass());
  // Reassociate expressions.
  fpm->add(createReassociatePass());
  // Eliminate common subexpressions.
  fpm->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  fpm->add(createCFGSimplificationPass());

  fpm->doInitialization();
}

//----- JIT DRIVER

static void handle_definition() {
  if (auto fn_ast = parse_definition()) {
    if (auto *fn_ir = fn_ast->codegen()) {
      std::fprintf(stderr, "Read function definition:");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
      jit->addModule(std::move(module));
      init_module_and_pass_manager();
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
      fn_protos[proto_ast->get_name()] = std::move(proto_ast);
    }
  } else {
    get_next_token();
  }
}

static void handle_top_level_expression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto fn_ast = parse_top_level_expr()) {
    if (fn_ast->codegen()) {
      // JIT the module containing the anon expr, keeping a handle so
      // we can free it later.
      auto h = jit->addModule(std::move(module));
      init_module_and_pass_manager();

      // Search the JIT for the __anon_expr symbol.
      auto expr_symbol = jit->findSymbol("__anon_expr");
      assert(expr_symbol && "Function not found");

      // Get the symbol's addr and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native fn.
      double (*fp)() = (double (*)())(intptr_t)cantFail(expr_symbol.getAddress());
      std::fprintf(stderr, "Evaluated to %f\n", fp());

      // Delete anon expr module from the JIT.
      jit->removeModule(h);
    }
  } else {
    get_next_token();
  }
}

/// top ::= definition | external | expression | ';'
static void main_loop(bool print) {
  for (;;) {
    if (print) {
      std::fprintf(stderr, "ready> ");
    }
    get_next_token();
    switch (cur_tok) {
    case tok_eof:
      return;
    case ';':
      get_next_token();
      break;
    case tok_def:
      handle_definition();
      break;
    case tok_extern:
      handle_extern();
      break;
    default:
      handle_top_level_expression();
      break;
    }
  }
}

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT double putchard(double x) {
  fputc((char)x, stderr);
  return 0;
}

extern "C" DLLEXPORT double printd(double x) {
  fprintf(stderr, "%f\n", x);
  return 0;
}

int main(int argc, char **argv) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  binop_precedence['<'] = 10;
  binop_precedence['+'] = 20;
  binop_precedence['-'] = 20;
  binop_precedence['*'] = 40;

  jit = llvm::make_unique<KaleidoscopeJIT>();

  init_module_and_pass_manager();

  if (argc > 1 && std::string(argv[1]) == "-file") {
    main_loop(false);
  } else {
    main_loop(true);
  }

  return 0;
}
