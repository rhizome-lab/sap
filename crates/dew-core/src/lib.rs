//! sap-core: minimal expression language.
//!
//! A simple expression parser that compiles string expressions into evaluable ASTs.
//! Variables and functions are provided by the caller - nothing is hardcoded.
//!
//! # Syntax
//!
//! ```text
//! // Operators (precedence low to high)
//! a + b, a - b     // Addition, subtraction
//! a * b, a / b     // Multiplication, division
//! a ^ b            // Exponentiation
//! -a               // Negation
//!
//! // Variables (resolved at eval time)
//! x, y, time       // Any identifier
//!
//! // Functions (registered via ExprFn trait)
//! sin(x), noise(x, y), etc.
//! ```
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::{Expr, FunctionRegistry};
//! use std::collections::HashMap;
//!
//! let registry = FunctionRegistry::new();
//! let expr = Expr::parse("x * 2 + y").unwrap();
//!
//! let mut vars = HashMap::new();
//! vars.insert("x".to_string(), 3.0);
//! vars.insert("y".to_string(), 1.0);
//!
//! let value = expr.eval(&vars, &registry).unwrap();
//! assert_eq!(value, 7.0);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// ExprFn trait and registry
// ============================================================================

/// A function that can be called from expressions.
///
/// Implement this trait to add custom functions.
/// Constants (like `pi`) can be 0-arg functions.
pub trait ExprFn: Send + Sync {
    /// Function name (e.g., "sin", "pi").
    fn name(&self) -> &str;

    /// Number of arguments this function expects.
    fn arg_count(&self) -> usize;

    /// Evaluate the function with the given arguments.
    fn call(&self, args: &[f32]) -> f32;

    /// Express as simpler expressions (enables automatic backend support).
    /// If this returns Some, backends can compile without knowing about this function.
    fn decompose(&self, _args: &[Ast]) -> Option<Ast> {
        None
    }
}

/// Registry of expression functions.
#[derive(Clone, Default)]
pub struct FunctionRegistry {
    funcs: HashMap<String, Arc<dyn ExprFn>>,
}

impl FunctionRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a function.
    pub fn register<F: ExprFn + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    /// Gets a function by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ExprFn>> {
        self.funcs.get(name)
    }

    /// Returns an iterator over all registered function names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Expression parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedChar(char),
    UnexpectedEnd,
    UnexpectedToken(String),
    InvalidNumber(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedChar(c) => write!(f, "unexpected character: '{}'", c),
            ParseError::UnexpectedEnd => write!(f, "unexpected end of expression"),
            ParseError::UnexpectedToken(t) => write!(f, "unexpected token: '{}'", t),
            ParseError::InvalidNumber(s) => write!(f, "invalid number: '{}'", s),
        }
    }
}

impl std::error::Error for ParseError {}

/// Expression evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    UnknownVariable(String),
    UnknownFunction(String),
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::UnknownVariable(name) => write!(f, "unknown variable: '{}'", name),
            EvalError::UnknownFunction(name) => write!(f, "unknown function: '{}'", name),
            EvalError::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{}' expects {} args, got {}",
                    func, expected, got
                )
            }
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Lexer
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f32),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.next_char();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Result<f32, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '.' {
                self.next_char();
            } else {
                break;
            }
        }
        let s = &self.input[start..self.pos];
        s.parse()
            .map_err(|_| ParseError::InvalidNumber(s.to_string()))
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.next_char();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        let Some(c) = self.peek_char() else {
            return Ok(Token::Eof);
        };

        match c {
            '+' => {
                self.next_char();
                Ok(Token::Plus)
            }
            '-' => {
                self.next_char();
                Ok(Token::Minus)
            }
            '*' => {
                self.next_char();
                Ok(Token::Star)
            }
            '/' => {
                self.next_char();
                Ok(Token::Slash)
            }
            '^' => {
                self.next_char();
                Ok(Token::Caret)
            }
            '(' => {
                self.next_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.next_char();
                Ok(Token::RParen)
            }
            ',' => {
                self.next_char();
                Ok(Token::Comma)
            }
            '0'..='9' | '.' => Ok(Token::Number(self.read_number()?)),
            'a'..='z' | 'A'..='Z' | '_' => Ok(Token::Ident(self.read_ident())),
            _ => Err(ParseError::UnexpectedChar(c)),
        }
    }
}

// ============================================================================
// AST
// ============================================================================

/// AST node for expressions.
#[derive(Debug, Clone)]
pub enum Ast {
    /// Numeric literal.
    Num(f32),
    /// Variable reference (resolved at eval time).
    Var(String),
    /// Binary operation.
    BinOp(BinOp, Box<Ast>, Box<Ast>),
    /// Unary operation.
    UnaryOp(UnaryOp, Box<Ast>),
    /// Function call.
    Call(String, Vec<Ast>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
}

// ============================================================================
// Parser
// ============================================================================

struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if self.current == expected {
            self.advance()
        } else {
            Err(ParseError::UnexpectedToken(format!("{:?}", self.current)))
        }
    }

    fn parse_expr(&mut self) -> Result<Ast, ParseError> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_mul_div()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    self.advance()?;
                    let right = self.parse_mul_div()?;
                    left = Ast::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_mul_div(&mut self) -> Result<Ast, ParseError> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Token::Slash => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = Ast::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Ast, ParseError> {
        let base = self.parse_unary()?;

        if self.current == Token::Caret {
            self.advance()?;
            let exp = self.parse_power()?; // Right associative
            Ok(Ast::BinOp(BinOp::Pow, Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Ast, ParseError> {
        if self.current == Token::Minus {
            self.advance()?;
            let inner = self.parse_unary()?;
            Ok(Ast::UnaryOp(UnaryOp::Neg, Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Ast, ParseError> {
        match &self.current {
            Token::Number(n) => {
                let n = *n;
                self.advance()?;
                Ok(Ast::Num(n))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;

                // Check if it's a function call
                if self.current == Token::LParen {
                    self.advance()?;
                    let mut args = Vec::new();
                    if self.current != Token::RParen {
                        args.push(self.parse_expr()?);
                        while self.current == Token::Comma {
                            self.advance()?;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Ast::Call(name, args))
                } else {
                    // It's a variable
                    Ok(Ast::Var(name))
                }
            }
            Token::LParen => {
                self.advance()?;
                let inner = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(inner)
            }
            Token::Eof => Err(ParseError::UnexpectedEnd),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", self.current))),
        }
    }
}

// ============================================================================
// Expression
// ============================================================================

/// A compiled expression that can be evaluated.
#[derive(Debug, Clone)]
pub struct Expr {
    ast: Ast,
}

impl Expr {
    /// Parses an expression from a string.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut parser = Parser::new(input)?;
        let ast = parser.parse_expr()?;
        if parser.current != Token::Eof {
            return Err(ParseError::UnexpectedToken(format!("{:?}", parser.current)));
        }
        Ok(Self { ast })
    }

    /// Returns a reference to the AST.
    pub fn ast(&self) -> &Ast {
        &self.ast
    }

    /// Evaluates the expression with given variables and functions.
    pub fn eval(
        &self,
        vars: &HashMap<String, f32>,
        funcs: &FunctionRegistry,
    ) -> Result<f32, EvalError> {
        eval_ast(&self.ast, vars, funcs)
    }
}

fn eval_ast(
    ast: &Ast,
    vars: &HashMap<String, f32>,
    funcs: &FunctionRegistry,
) -> Result<f32, EvalError> {
    match ast {
        Ast::Num(n) => Ok(*n),
        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| EvalError::UnknownVariable(name.clone())),
        Ast::BinOp(op, l, r) => {
            let l = eval_ast(l, vars, funcs)?;
            let r = eval_ast(r, vars, funcs)?;
            Ok(match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => l / r,
                BinOp::Pow => l.powf(r),
            })
        }
        Ast::UnaryOp(op, inner) => {
            let v = eval_ast(inner, vars, funcs)?;
            Ok(match op {
                UnaryOp::Neg => -v,
            })
        }
        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| EvalError::UnknownFunction(name.clone()))?;

            if args.len() != func.arg_count() {
                return Err(EvalError::WrongArgCount {
                    func: name.clone(),
                    expected: func.arg_count(),
                    got: args.len(),
                });
            }

            let arg_values: Vec<f32> = args
                .iter()
                .map(|a| eval_ast(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            Ok(func.call(&arg_values))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(expr_str: &str, vars: &[(&str, f32)]) -> f32 {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse(expr_str).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        expr.eval(&var_map, &registry).unwrap()
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(eval("42", &[]), 42.0);
    }

    #[test]
    fn test_parse_float() {
        assert!((eval("1.234", &[]) - 1.234).abs() < 0.001);
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(eval("x", &[("x", 5.0)]), 5.0);
        assert_eq!(eval("foo", &[("foo", 3.0)]), 3.0);
    }

    #[test]
    fn test_parse_add() {
        assert_eq!(eval("1 + 2", &[]), 3.0);
    }

    #[test]
    fn test_parse_mul() {
        assert_eq!(eval("3 * 4", &[]), 12.0);
    }

    #[test]
    fn test_precedence() {
        assert_eq!(eval("2 + 3 * 4", &[]), 14.0);
    }

    #[test]
    fn test_parentheses() {
        assert_eq!(eval("(2 + 3) * 4", &[]), 20.0);
    }

    #[test]
    fn test_negation() {
        assert_eq!(eval("-5", &[]), -5.0);
    }

    #[test]
    fn test_power() {
        assert_eq!(eval("2 ^ 3", &[]), 8.0);
    }

    #[test]
    fn test_unknown_variable() {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse("unknown").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::UnknownVariable(_))));
    }

    #[test]
    fn test_unknown_function() {
        let registry = FunctionRegistry::new();
        let expr = Expr::parse("unknown(1)").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::UnknownFunction(_))));
    }

    #[test]
    fn test_custom_function() {
        struct Double;
        impl ExprFn for Double {
            fn name(&self) -> &str {
                "double"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0] * 2.0
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Double);

        let expr = Expr::parse("double(5)").unwrap();
        let vars = HashMap::new();
        assert_eq!(expr.eval(&vars, &registry).unwrap(), 10.0);
    }

    #[test]
    fn test_zero_arg_function() {
        struct Pi;
        impl ExprFn for Pi {
            fn name(&self) -> &str {
                "pi"
            }
            fn arg_count(&self) -> usize {
                0
            }
            fn call(&self, _args: &[f32]) -> f32 {
                std::f32::consts::PI
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Pi);

        let expr = Expr::parse("pi()").unwrap();
        let vars = HashMap::new();
        assert!((expr.eval(&vars, &registry).unwrap() - std::f32::consts::PI).abs() < 0.001);
    }

    #[test]
    fn test_wrong_arg_count() {
        struct OneArg;
        impl ExprFn for OneArg {
            fn name(&self) -> &str {
                "one_arg"
            }
            fn arg_count(&self) -> usize {
                1
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0]
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(OneArg);

        let expr = Expr::parse("one_arg(1, 2)").unwrap();
        let vars = HashMap::new();
        let result = expr.eval(&vars, &registry);
        assert!(matches!(result, Err(EvalError::WrongArgCount { .. })));
    }

    #[test]
    fn test_complex_expression() {
        struct Add;
        impl ExprFn for Add {
            fn name(&self) -> &str {
                "add"
            }
            fn arg_count(&self) -> usize {
                2
            }
            fn call(&self, args: &[f32]) -> f32 {
                args[0] + args[1]
            }
        }

        let mut registry = FunctionRegistry::new();
        registry.register(Add);

        let expr = Expr::parse("add(x * 2, y + 1)").unwrap();
        let vars: HashMap<String, f32> = [("x".to_string(), 3.0), ("y".to_string(), 4.0)].into();
        assert_eq!(expr.eval(&vars, &registry).unwrap(), 11.0); // (3*2) + (4+1) = 11
    }
}
