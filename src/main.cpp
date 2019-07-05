#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

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
