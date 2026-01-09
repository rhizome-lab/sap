//! Quaternion support for sap expressions.
//!
//! Provides quaternion types and operations for 3D rotations.
//! Uses [x, y, z, w] component order (scalar last, GLM/glTF convention).
//!
//! # Example
//!
//! ```
//! use rhizome_dew_core::Expr;
//! use rhizome_dew_quaternion::{Value, eval, quaternion_registry};
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("normalize(q)").unwrap();
//!
//! let mut vars: HashMap<String, Value<f32>> = HashMap::new();
//! vars.insert("q".to_string(), Value::Quaternion([0.0, 0.0, 0.0, 2.0]));
//!
//! let registry = quaternion_registry();
//! let result = eval(expr.ast(), &vars, &registry).unwrap();
//! assert_eq!(result, Value::Quaternion([0.0, 0.0, 0.0, 1.0]));
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

pub use funcs::{
    AxisAngle, Conj, Dot, Inverse, Length, Lerp, Normalize, Rotate, Slerp, quaternion_registry,
    register_quaternion,
};

// ============================================================================
// Types
// ============================================================================

/// Type of a quaternion value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    /// Real scalar.
    Scalar,
    /// 3D vector [x, y, z].
    Vec3,
    /// Quaternion [x, y, z, w].
    Quaternion,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar => write!(f, "scalar"),
            Type::Vec3 => write!(f, "vec3"),
            Type::Quaternion => write!(f, "quaternion"),
        }
    }
}

// ============================================================================
// QuaternionValue trait (for composability)
// ============================================================================

/// Trait for values that support quaternion operations.
///
/// Implement this for combined value types when composing multiple domain crates.
pub trait QuaternionValue<T: Float>: Clone + PartialEq + Sized + std::fmt::Debug {
    /// Returns the type of this value.
    fn typ(&self) -> Type;

    // Construction
    fn from_scalar(v: T) -> Self;
    fn from_vec3(v: [T; 3]) -> Self;
    fn from_quaternion(q: [T; 4]) -> Self;

    // Extraction
    fn as_scalar(&self) -> Option<T>;
    fn as_vec3(&self) -> Option<[T; 3]>;
    fn as_quaternion(&self) -> Option<[T; 4]>;
}

// ============================================================================
// Values
// ============================================================================

/// A quaternion value, generic over numeric type.
///
/// Quaternion uses [x, y, z, w] order (scalar last).
#[derive(Debug, Clone, PartialEq)]
pub enum Value<T> {
    /// Real scalar.
    Scalar(T),
    /// 3D vector [x, y, z].
    Vec3([T; 3]),
    /// Quaternion [x, y, z, w] (scalar last).
    Quaternion([T; 4]),
}

impl<T> Value<T> {
    /// Returns the type of this value.
    pub fn typ(&self) -> Type {
        match self {
            Value::Scalar(_) => Type::Scalar,
            Value::Vec3(_) => Type::Vec3,
            Value::Quaternion(_) => Type::Quaternion,
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

    /// Try to get as vec3.
    pub fn as_vec3(&self) -> Option<[T; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as quaternion.
    pub fn as_quaternion(&self) -> Option<[T; 4]> {
        match self {
            Value::Quaternion(q) => Some(*q),
            _ => None,
        }
    }
}

impl<T: Float + std::fmt::Debug> QuaternionValue<T> for Value<T> {
    fn typ(&self) -> Type {
        Value::typ(self)
    }

    fn from_scalar(v: T) -> Self {
        Value::Scalar(v)
    }

    fn from_vec3(v: [T; 3]) -> Self {
        Value::Vec3(v)
    }

    fn from_quaternion(q: [T; 4]) -> Self {
        Value::Quaternion(q)
    }

    fn as_scalar(&self) -> Option<T> {
        Value::as_scalar(self)
    }

    fn as_vec3(&self) -> Option<[T; 3]> {
        Value::as_vec3(self)
    }

    fn as_quaternion(&self) -> Option<[T; 4]> {
        Value::as_quaternion(self)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Quaternion evaluation error.
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

/// A function signature.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub args: Vec<Type>,
    pub ret: Type,
}

/// A function that can be called from quaternion expressions.
pub trait QuaternionFn<T>: Send + Sync {
    /// Function name.
    fn name(&self) -> &str;

    /// Available signatures for this function.
    fn signatures(&self) -> Vec<Signature>;

    /// Call the function with typed arguments.
    fn call(&self, args: &[Value<T>]) -> Value<T>;
}

/// Registry of quaternion functions.
#[derive(Clone)]
pub struct FunctionRegistry<T> {
    funcs: HashMap<String, Arc<dyn QuaternionFn<T>>>,
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

    pub fn register<F: QuaternionFn<T> + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn QuaternionFn<T>>> {
        self.funcs.get(name)
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an AST with quaternion values.
pub fn eval<T: Float>(
    ast: &Ast,
    vars: &HashMap<String, Value<T>>,
    funcs: &FunctionRegistry<T>,
) -> Result<Value<T>, Error> {
    match ast {
        Ast::Num(n) => Ok(Value::Scalar(T::from(*n).unwrap())),

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
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Result<Value<f32>, Error> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = quaternion_registry();
        eval(expr.ast(), &var_map, &registry)
    }

    #[test]
    fn test_quaternion_add() {
        let result = eval_expr(
            "a + b",
            &[
                ("a", Value::Quaternion([1.0, 2.0, 3.0, 4.0])),
                ("b", Value::Quaternion([5.0, 6.0, 7.0, 8.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Quaternion([6.0, 8.0, 10.0, 12.0]));
    }

    #[test]
    fn test_quaternion_mul() {
        // Identity quaternion: [0, 0, 0, 1]
        // q * identity = q
        let result = eval_expr(
            "a * b",
            &[
                ("a", Value::Quaternion([1.0, 2.0, 3.0, 4.0])),
                ("b", Value::Quaternion([0.0, 0.0, 0.0, 1.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Quaternion([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_quaternion_neg() {
        let result = eval_expr("-q", &[("q", Value::Quaternion([1.0, 2.0, 3.0, 4.0]))]);
        assert_eq!(result.unwrap(), Value::Quaternion([-1.0, -2.0, -3.0, -4.0]));
    }

    #[test]
    fn test_quaternion_scalar_mul() {
        let result = eval_expr(
            "s * q",
            &[
                ("s", Value::Scalar(2.0)),
                ("q", Value::Quaternion([1.0, 2.0, 3.0, 4.0])),
            ],
        );
        assert_eq!(result.unwrap(), Value::Quaternion([2.0, 4.0, 6.0, 8.0]));
    }
}
