//! Linear algebra types and operations for sap expressions.
//!
//! This crate provides vector and matrix types that work with sap-core's AST.
//! Types propagate during evaluation/emission - no separate type inference pass.
//!
//! # Features
//!
//! - `3d` (default): Enables Vec3, Mat3
//! - `4d`: Enables Vec4, Mat4 (implies 3d)
//!
//! # Composability
//!
//! For composing multiple domain crates (e.g., linalg + rotors), the [`LinalgValue`]
//! trait allows defining a combined value type that works with both crates.
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_linalg::{Value, Type, eval, FunctionRegistry};
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("a + b").unwrap();
//!
//! let mut vars: HashMap<String, Value<f32>> = HashMap::new();
//! vars.insert("a".to_string(), Value::Vec2([1.0, 2.0]));
//! vars.insert("b".to_string(), Value::Vec2([3.0, 4.0]));
//!
//! let registry = FunctionRegistry::new();
//! let result = eval(expr.ast(), &vars, &registry).unwrap();
//! assert_eq!(result, Value::Vec2([4.0, 6.0]));
//! ```

use num_traits::Float;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;
use std::sync::Arc;

mod funcs;
mod ops;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "lua")]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;

#[cfg(feature = "3d")]
pub use funcs::Cross;
pub use funcs::{
    Distance, Dot, Hadamard, Length, Lerp, Mix, Normalize, Reflect, linalg_registry,
    register_linalg,
};

// ============================================================================
// Types
// ============================================================================

/// Type of a linalg value (shape only, not numeric type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar,
    Vec2,
    #[cfg(feature = "3d")]
    Vec3,
    #[cfg(feature = "4d")]
    Vec4,
    Mat2,
    #[cfg(feature = "3d")]
    Mat3,
    #[cfg(feature = "4d")]
    Mat4,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar => write!(f, "scalar"),
            Type::Vec2 => write!(f, "vec2"),
            #[cfg(feature = "3d")]
            Type::Vec3 => write!(f, "vec3"),
            #[cfg(feature = "4d")]
            Type::Vec4 => write!(f, "vec4"),
            Type::Mat2 => write!(f, "mat2"),
            #[cfg(feature = "3d")]
            Type::Mat3 => write!(f, "mat3"),
            #[cfg(feature = "4d")]
            Type::Mat4 => write!(f, "mat4"),
        }
    }
}

// ============================================================================
// LinalgValue trait (Option 3: generic over value type)
// ============================================================================

/// Trait for values that support linalg operations.
///
/// This enables composing multiple domain crates. Users can define their own
/// combined enum implementing traits from each domain, then use both crates
/// with zero conversion.
///
/// # Example (composing linalg + rotors)
///
/// ```ignore
/// enum CombinedValue<T> {
///     Scalar(T),
///     Vec2([T; 2]),
///     Vec3([T; 3]),
///     Rotor2(Rotor2<T>),
/// }
///
/// impl<T: Float> LinalgValue<T> for CombinedValue<T> { ... }
/// impl<T: Float> RotorValue<T> for CombinedValue<T> { ... }
/// ```
pub trait LinalgValue<T: Float>: Clone + PartialEq + Sized + std::fmt::Debug {
    /// Returns the type of this value.
    fn typ(&self) -> Type;

    // Construction
    fn from_scalar(v: T) -> Self;
    fn from_vec2(v: [T; 2]) -> Self;
    #[cfg(feature = "3d")]
    fn from_vec3(v: [T; 3]) -> Self;
    #[cfg(feature = "4d")]
    fn from_vec4(v: [T; 4]) -> Self;
    fn from_mat2(v: [T; 4]) -> Self;
    #[cfg(feature = "3d")]
    fn from_mat3(v: [T; 9]) -> Self;
    #[cfg(feature = "4d")]
    fn from_mat4(v: [T; 16]) -> Self;

    // Extraction (returns None if wrong type)
    fn as_scalar(&self) -> Option<T>;
    fn as_vec2(&self) -> Option<[T; 2]>;
    #[cfg(feature = "3d")]
    fn as_vec3(&self) -> Option<[T; 3]>;
    #[cfg(feature = "4d")]
    fn as_vec4(&self) -> Option<[T; 4]>;
    fn as_mat2(&self) -> Option<[T; 4]>;
    #[cfg(feature = "3d")]
    fn as_mat3(&self) -> Option<[T; 9]>;
    #[cfg(feature = "4d")]
    fn as_mat4(&self) -> Option<[T; 16]>;
}

// ============================================================================
// Values
// ============================================================================

/// A linalg value, generic over numeric type.
///
/// This is the default concrete type for standalone use of sap-linalg.
/// For composing with other domain crates, implement `LinalgValue<T>` for
/// your own combined enum.
#[derive(Debug, Clone, PartialEq)]
pub enum Value<T> {
    Scalar(T),
    Vec2([T; 2]),
    #[cfg(feature = "3d")]
    Vec3([T; 3]),
    #[cfg(feature = "4d")]
    Vec4([T; 4]),
    Mat2([T; 4]), // column-major: [c0r0, c0r1, c1r0, c1r1]
    #[cfg(feature = "3d")]
    Mat3([T; 9]), // column-major
    #[cfg(feature = "4d")]
    Mat4([T; 16]), // column-major
}

// Inherent methods for backwards compatibility (don't require Debug bound)
impl<T> Value<T> {
    /// Returns the type of this value.
    pub fn typ(&self) -> Type {
        match self {
            Value::Scalar(_) => Type::Scalar,
            Value::Vec2(_) => Type::Vec2,
            #[cfg(feature = "3d")]
            Value::Vec3(_) => Type::Vec3,
            #[cfg(feature = "4d")]
            Value::Vec4(_) => Type::Vec4,
            Value::Mat2(_) => Type::Mat2,
            #[cfg(feature = "3d")]
            Value::Mat3(_) => Type::Mat3,
            #[cfg(feature = "4d")]
            Value::Mat4(_) => Type::Mat4,
        }
    }
}

impl<T: Copy> Value<T> {
    /// Try to get as scalar.
    pub fn as_scalar(&self) -> Option<T> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

impl<T: Float + std::fmt::Debug> LinalgValue<T> for Value<T> {
    fn typ(&self) -> Type {
        // Delegate to inherent method
        Value::typ(self)
    }

    fn from_scalar(v: T) -> Self {
        Value::Scalar(v)
    }
    fn from_vec2(v: [T; 2]) -> Self {
        Value::Vec2(v)
    }
    #[cfg(feature = "3d")]
    fn from_vec3(v: [T; 3]) -> Self {
        Value::Vec3(v)
    }
    #[cfg(feature = "4d")]
    fn from_vec4(v: [T; 4]) -> Self {
        Value::Vec4(v)
    }
    fn from_mat2(v: [T; 4]) -> Self {
        Value::Mat2(v)
    }
    #[cfg(feature = "3d")]
    fn from_mat3(v: [T; 9]) -> Self {
        Value::Mat3(v)
    }
    #[cfg(feature = "4d")]
    fn from_mat4(v: [T; 16]) -> Self {
        Value::Mat4(v)
    }

    fn as_scalar(&self) -> Option<T> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }
    fn as_vec2(&self) -> Option<[T; 2]> {
        match self {
            Value::Vec2(v) => Some(*v),
            _ => None,
        }
    }
    #[cfg(feature = "3d")]
    fn as_vec3(&self) -> Option<[T; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            _ => None,
        }
    }
    #[cfg(feature = "4d")]
    fn as_vec4(&self) -> Option<[T; 4]> {
        match self {
            Value::Vec4(v) => Some(*v),
            _ => None,
        }
    }
    fn as_mat2(&self) -> Option<[T; 4]> {
        match self {
            Value::Mat2(v) => Some(*v),
            _ => None,
        }
    }
    #[cfg(feature = "3d")]
    fn as_mat3(&self) -> Option<[T; 9]> {
        match self {
            Value::Mat3(v) => Some(*v),
            _ => None,
        }
    }
    #[cfg(feature = "4d")]
    fn as_mat4(&self) -> Option<[T; 16]> {
        match self {
            Value::Mat4(v) => Some(*v),
            _ => None,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Linalg evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Unknown variable.
    UnknownVariable(String),
    /// Unknown function.
    UnknownFunction(String),
    /// Type mismatch for binary operation.
    BinaryTypeMismatch { op: BinOp, left: Type, right: Type },
    /// Type mismatch for unary operation.
    UnaryTypeMismatch { op: UnaryOp, operand: Type },
    /// Wrong number of arguments to function.
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    /// Type mismatch in function arguments.
    FunctionTypeMismatch {
        func: String,
        expected: Vec<Type>,
        got: Vec<Type>,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            Error::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            Error::BinaryTypeMismatch { op, left, right } => {
                write!(f, "cannot apply {op:?} to {left} and {right}")
            }
            Error::UnaryTypeMismatch { op, operand } => {
                write!(f, "cannot apply {op:?} to {operand}")
            }
            Error::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(f, "function '{func}' expects {expected} args, got {got}")
            }
            Error::FunctionTypeMismatch {
                func,
                expected,
                got,
            } => {
                write!(
                    f,
                    "function '{func}' expects types {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

// ============================================================================
// Function Registry
// ============================================================================

/// A function signature: argument types and return type.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub args: Vec<Type>,
    pub ret: Type,
}

/// A function that can be called from linalg expressions.
pub trait LinalgFn<T>: Send + Sync {
    /// Function name.
    fn name(&self) -> &str;

    /// Available signatures for this function.
    fn signatures(&self) -> Vec<Signature>;

    /// Call the function with typed arguments.
    /// Caller guarantees args match one of the signatures.
    fn call(&self, args: &[Value<T>]) -> Value<T>;
}

/// Registry of linalg functions.
#[derive(Clone)]
pub struct FunctionRegistry<T> {
    funcs: HashMap<String, Arc<dyn LinalgFn<T>>>,
}

impl<T> Default for FunctionRegistry<T> {
    fn default() -> Self {
        Self {
            funcs: HashMap::new(),
        }
    }
}

impl<T> FunctionRegistry<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F: LinalgFn<T> + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn LinalgFn<T>>> {
        self.funcs.get(name)
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an AST with linalg values.
///
/// Literals from the AST (f32) are converted to T via `T::from(f32)`.
pub fn eval<T: Float>(
    ast: &Ast,
    vars: &HashMap<String, Value<T>>,
    funcs: &FunctionRegistry<T>,
) -> Result<Value<T>, Error> {
    match ast {
        Ast::Num(n) => {
            // Convert f32 literal to T
            Ok(Value::Scalar(T::from(*n).unwrap()))
        }

        Ast::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| Error::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let left_val = eval(left, vars, funcs)?;
            let right_val = eval(right, vars, funcs)?;
            ops::apply_binop(*op, left_val, right_val)
        }

        Ast::UnaryOp(op, inner) => {
            let val = eval(inner, vars, funcs)?;
            ops::apply_unaryop(*op, val)
        }

        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| Error::UnknownFunction(name.clone()))?;

            let arg_vals: Vec<Value<T>> = args
                .iter()
                .map(|a| eval(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            let arg_types: Vec<Type> = arg_vals.iter().map(|v| v.typ()).collect();

            // Find matching signature
            let matched = func.signatures().iter().any(|sig| sig.args == arg_types);
            if !matched {
                return Err(Error::FunctionTypeMismatch {
                    func: name.clone(),
                    expected: func
                        .signatures()
                        .first()
                        .map(|s| s.args.clone())
                        .unwrap_or_default(),
                    got: arg_types,
                });
            }

            Ok(func.call(&arg_vals))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod exhaustive_tests;

#[cfg(test)]
mod parity_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Result<Value<f32>, Error> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = FunctionRegistry::new();
        eval(expr.ast(), &var_map, &registry)
    }

    #[test]
    fn test_scalar_add() {
        let result = eval_expr(
            "a + b",
            &[("a", Value::Scalar(1.0)), ("b", Value::Scalar(2.0))],
        );
        assert_eq!(result.unwrap(), Value::Scalar(3.0));
    }

    #[test]
    fn test_vec2_add() {
        let result = eval_expr(
            "a + b",
            &[
                ("a", Value::Vec2([1.0, 2.0])),
                ("b", Value::Vec2([3.0, 4.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Vec2([4.0, 6.0]));
    }

    #[test]
    fn test_vec2_scalar_mul() {
        let result = eval_expr(
            "v * s",
            &[("v", Value::Vec2([2.0, 3.0])), ("s", Value::Scalar(2.0))],
        );
        assert_eq!(result.unwrap(), Value::Vec2([4.0, 6.0]));
    }

    #[test]
    fn test_scalar_vec2_mul() {
        let result = eval_expr(
            "s * v",
            &[("s", Value::Scalar(2.0)), ("v", Value::Vec2([2.0, 3.0]))],
        );
        assert_eq!(result.unwrap(), Value::Vec2([4.0, 6.0]));
    }

    #[test]
    fn test_vec2_neg() {
        let result = eval_expr("-v", &[("v", Value::Vec2([1.0, -2.0]))]);
        assert_eq!(result.unwrap(), Value::Vec2([-1.0, 2.0]));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_vec3_add() {
        let result = eval_expr(
            "a + b",
            &[
                ("a", Value::Vec3([1.0, 2.0, 3.0])),
                ("b", Value::Vec3([4.0, 5.0, 6.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Vec3([5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_type_mismatch() {
        let result = eval_expr(
            "a + b",
            &[("a", Value::Scalar(1.0)), ("b", Value::Vec2([1.0, 2.0]))],
        );
        assert!(matches!(result, Err(Error::BinaryTypeMismatch { .. })));
    }

    #[test]
    fn test_literal_conversion() {
        // Test that f32 literals work with f64 values
        let expr = Expr::parse("a + 1.5").unwrap();
        let mut vars: HashMap<String, Value<f64>> = HashMap::new();
        vars.insert("a".to_string(), Value::Scalar(2.5));
        let registry = FunctionRegistry::new();
        let result = eval(expr.ast(), &vars, &registry).unwrap();
        assert_eq!(result, Value::Scalar(4.0));
    }
}
