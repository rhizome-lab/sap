//! Standard scalar function library for sap expressions.
//!
//! Provides common math functions (sin, cos, sqrt, etc.) and constants (pi, e)
//! with a generic numeric type `T: Float`.
//!
//! # Usage
//!
//! ```
//! use rhizome_sap_core::Expr;
//! use rhizome_sap_scalar::{eval, scalar_registry};
//! use std::collections::HashMap;
//!
//! let expr = Expr::parse("sin(x) + pi()").unwrap();
//! let vars: HashMap<String, f32> = [("x".to_string(), 0.0)].into();
//! let registry = scalar_registry();
//! let value = eval(expr.ast(), &vars, &registry).unwrap();
//! ```

use num_traits::Float;
use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(feature = "lua")]
pub mod lua;

#[cfg(feature = "cranelift")]
pub mod cranelift;

// ============================================================================
// Errors
// ============================================================================

/// Scalar evaluation error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Unknown variable.
    UnknownVariable(String),
    /// Unknown function.
    UnknownFunction(String),
    /// Wrong number of arguments to function.
    WrongArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            Error::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            Error::WrongArgCount {
                func,
                expected,
                got,
            } => {
                write!(f, "function '{func}' expects {expected} args, got {got}")
            }
        }
    }
}

impl std::error::Error for Error {}

// ============================================================================
// Function Registry
// ============================================================================

/// A scalar function that can be called from expressions.
pub trait ScalarFn<T>: Send + Sync {
    /// Function name.
    fn name(&self) -> &str;

    /// Number of arguments.
    fn arg_count(&self) -> usize;

    /// Call the function with arguments.
    fn call(&self, args: &[T]) -> T;
}

/// Registry of scalar functions.
#[derive(Clone)]
pub struct FunctionRegistry<T> {
    funcs: HashMap<String, Arc<dyn ScalarFn<T>>>,
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

    pub fn register<F: ScalarFn<T> + 'static>(&mut self, func: F) {
        self.funcs.insert(func.name().to_string(), Arc::new(func));
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn ScalarFn<T>>> {
        self.funcs.get(name)
    }

    /// Returns an iterator over all function names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an AST with scalar values.
pub fn eval<T: Float>(
    ast: &Ast,
    vars: &HashMap<String, T>,
    funcs: &FunctionRegistry<T>,
) -> Result<T, Error> {
    match ast {
        Ast::Num(n) => Ok(T::from(*n).unwrap()),

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| Error::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let l = eval(left, vars, funcs)?;
            let r = eval(right, vars, funcs)?;
            Ok(match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => l / r,
                BinOp::Pow => l.powf(r),
            })
        }

        Ast::UnaryOp(op, inner) => {
            let v = eval(inner, vars, funcs)?;
            Ok(match op {
                UnaryOp::Neg => -v,
            })
        }

        Ast::Call(name, args) => {
            let func = funcs
                .get(name)
                .ok_or_else(|| Error::UnknownFunction(name.clone()))?;

            if args.len() != func.arg_count() {
                return Err(Error::WrongArgCount {
                    func: name.clone(),
                    expected: func.arg_count(),
                    got: args.len(),
                });
            }

            let arg_vals: Vec<T> = args
                .iter()
                .map(|a| eval(a, vars, funcs))
                .collect::<Result<_, _>>()?;

            Ok(func.call(&arg_vals))
        }
    }
}

// ============================================================================
// Standard Functions - Constants
// ============================================================================

/// Pi constant: pi() = 3.14159...
pub struct Pi;
impl<T: Float> ScalarFn<T> for Pi {
    fn name(&self) -> &str {
        "pi"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::PI).unwrap()
    }
}

/// Euler's number: e() = 2.71828...
pub struct E;
impl<T: Float> ScalarFn<T> for E {
    fn name(&self) -> &str {
        "e"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::E).unwrap()
    }
}

/// Tau constant: tau() = 2*pi = 6.28318...
pub struct Tau;
impl<T: Float> ScalarFn<T> for Tau {
    fn name(&self) -> &str {
        "tau"
    }
    fn arg_count(&self) -> usize {
        0
    }
    fn call(&self, _args: &[T]) -> T {
        T::from(std::f64::consts::TAU).unwrap()
    }
}

// ============================================================================
// Standard Functions - Trigonometric
// ============================================================================

pub struct Sin;
impl<T: Float> ScalarFn<T> for Sin {
    fn name(&self) -> &str {
        "sin"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sin()
    }
}

pub struct Cos;
impl<T: Float> ScalarFn<T> for Cos {
    fn name(&self) -> &str {
        "cos"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].cos()
    }
}

pub struct Tan;
impl<T: Float> ScalarFn<T> for Tan {
    fn name(&self) -> &str {
        "tan"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].tan()
    }
}

pub struct Asin;
impl<T: Float> ScalarFn<T> for Asin {
    fn name(&self) -> &str {
        "asin"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].asin()
    }
}

pub struct Acos;
impl<T: Float> ScalarFn<T> for Acos {
    fn name(&self) -> &str {
        "acos"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].acos()
    }
}

pub struct Atan;
impl<T: Float> ScalarFn<T> for Atan {
    fn name(&self) -> &str {
        "atan"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].atan()
    }
}

pub struct Atan2;
impl<T: Float> ScalarFn<T> for Atan2 {
    fn name(&self) -> &str {
        "atan2"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].atan2(args[1])
    }
}

pub struct Sinh;
impl<T: Float> ScalarFn<T> for Sinh {
    fn name(&self) -> &str {
        "sinh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sinh()
    }
}

pub struct Cosh;
impl<T: Float> ScalarFn<T> for Cosh {
    fn name(&self) -> &str {
        "cosh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].cosh()
    }
}

pub struct Tanh;
impl<T: Float> ScalarFn<T> for Tanh {
    fn name(&self) -> &str {
        "tanh"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].tanh()
    }
}

// ============================================================================
// Standard Functions - Exponential / Logarithmic
// ============================================================================

pub struct Exp;
impl<T: Float> ScalarFn<T> for Exp {
    fn name(&self) -> &str {
        "exp"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].exp()
    }
}

pub struct Exp2;
impl<T: Float> ScalarFn<T> for Exp2 {
    fn name(&self) -> &str {
        "exp2"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].exp2()
    }
}

pub struct Log;
impl<T: Float> ScalarFn<T> for Log {
    fn name(&self) -> &str {
        "log"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ln()
    }
}

pub struct Ln;
impl<T: Float> ScalarFn<T> for Ln {
    fn name(&self) -> &str {
        "ln"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ln()
    }
}

pub struct Log2;
impl<T: Float> ScalarFn<T> for Log2 {
    fn name(&self) -> &str {
        "log2"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].log2()
    }
}

pub struct Log10;
impl<T: Float> ScalarFn<T> for Log10 {
    fn name(&self) -> &str {
        "log10"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].log10()
    }
}

pub struct Pow;
impl<T: Float> ScalarFn<T> for Pow {
    fn name(&self) -> &str {
        "pow"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].powf(args[1])
    }
}

pub struct Sqrt;
impl<T: Float> ScalarFn<T> for Sqrt {
    fn name(&self) -> &str {
        "sqrt"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].sqrt()
    }
}

pub struct InverseSqrt;
impl<T: Float> ScalarFn<T> for InverseSqrt {
    fn name(&self) -> &str {
        "inversesqrt"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        T::one() / args[0].sqrt()
    }
}

// ============================================================================
// Standard Functions - Common Math
// ============================================================================

pub struct Abs;
impl<T: Float> ScalarFn<T> for Abs {
    fn name(&self) -> &str {
        "abs"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].abs()
    }
}

pub struct Sign;
impl<T: Float> ScalarFn<T> for Sign {
    fn name(&self) -> &str {
        "sign"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        let x = args[0];
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            -T::one()
        } else {
            T::zero()
        }
    }
}

pub struct Floor;
impl<T: Float> ScalarFn<T> for Floor {
    fn name(&self) -> &str {
        "floor"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].floor()
    }
}

pub struct Ceil;
impl<T: Float> ScalarFn<T> for Ceil {
    fn name(&self) -> &str {
        "ceil"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].ceil()
    }
}

pub struct Round;
impl<T: Float> ScalarFn<T> for Round {
    fn name(&self) -> &str {
        "round"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].round()
    }
}

pub struct Trunc;
impl<T: Float> ScalarFn<T> for Trunc {
    fn name(&self) -> &str {
        "trunc"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].trunc()
    }
}

pub struct Fract;
impl<T: Float> ScalarFn<T> for Fract {
    fn name(&self) -> &str {
        "fract"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].fract()
    }
}

pub struct Min;
impl<T: Float> ScalarFn<T> for Min {
    fn name(&self) -> &str {
        "min"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].min(args[1])
    }
}

pub struct Max;
impl<T: Float> ScalarFn<T> for Max {
    fn name(&self) -> &str {
        "max"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(args[1])
    }
}

pub struct Clamp;
impl<T: Float> ScalarFn<T> for Clamp {
    fn name(&self) -> &str {
        "clamp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(args[1]).min(args[2])
    }
}

pub struct Saturate;
impl<T: Float> ScalarFn<T> for Saturate {
    fn name(&self) -> &str {
        "saturate"
    }
    fn arg_count(&self) -> usize {
        1
    }
    fn call(&self, args: &[T]) -> T {
        args[0].max(T::zero()).min(T::one())
    }
}

// ============================================================================
// Standard Functions - Interpolation
// ============================================================================

/// Linear interpolation: lerp(a, b, t) = a + (b - a) * t
pub struct Lerp;
impl<T: Float> ScalarFn<T> for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, t) = (args[0], args[1], args[2]);
        a + (b - a) * t
    }
}

/// Alias for lerp (GLSL naming)
pub struct Mix;
impl<T: Float> ScalarFn<T> for Mix {
    fn name(&self) -> &str {
        "mix"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, t) = (args[0], args[1], args[2]);
        a + (b - a) * t
    }
}

/// Step function: step(edge, x) = x < edge ? 0.0 : 1.0
pub struct Step;
impl<T: Float> ScalarFn<T> for Step {
    fn name(&self) -> &str {
        "step"
    }
    fn arg_count(&self) -> usize {
        2
    }
    fn call(&self, args: &[T]) -> T {
        if args[1] < args[0] {
            T::zero()
        } else {
            T::one()
        }
    }
}

/// Smooth Hermite interpolation
pub struct Smoothstep;
impl<T: Float> ScalarFn<T> for Smoothstep {
    fn name(&self) -> &str {
        "smoothstep"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (edge0, edge1, x) = (args[0], args[1], args[2]);
        let t = ((x - edge0) / (edge1 - edge0)).max(T::zero()).min(T::one());
        let three = T::from(3.0).unwrap();
        let two = T::from(2.0).unwrap();
        t * t * (three - two * t)
    }
}

/// Inverse lerp: inverse_lerp(a, b, v) = (v - a) / (b - a)
pub struct InverseLerp;
impl<T: Float> ScalarFn<T> for InverseLerp {
    fn name(&self) -> &str {
        "inverse_lerp"
    }
    fn arg_count(&self) -> usize {
        3
    }
    fn call(&self, args: &[T]) -> T {
        let (a, b, v) = (args[0], args[1], args[2]);
        (v - a) / (b - a)
    }
}

/// Remap: remap(x, in_lo, in_hi, out_lo, out_hi)
pub struct Remap;
impl<T: Float> ScalarFn<T> for Remap {
    fn name(&self) -> &str {
        "remap"
    }
    fn arg_count(&self) -> usize {
        5
    }
    fn call(&self, args: &[T]) -> T {
        let (x, in_lo, in_hi, out_lo, out_hi) = (args[0], args[1], args[2], args[3], args[4]);
        let t = (x - in_lo) / (in_hi - in_lo);
        out_lo + (out_hi - out_lo) * t
    }
}

// ============================================================================
// Registry
// ============================================================================

/// Registers all standard scalar functions into the given registry.
pub fn register_scalar<T: Float + 'static>(registry: &mut FunctionRegistry<T>) {
    // Constants
    registry.register(Pi);
    registry.register(E);
    registry.register(Tau);

    // Trigonometric
    registry.register(Sin);
    registry.register(Cos);
    registry.register(Tan);
    registry.register(Asin);
    registry.register(Acos);
    registry.register(Atan);
    registry.register(Atan2);
    registry.register(Sinh);
    registry.register(Cosh);
    registry.register(Tanh);

    // Exponential / logarithmic
    registry.register(Exp);
    registry.register(Exp2);
    registry.register(Log);
    registry.register(Ln);
    registry.register(Log2);
    registry.register(Log10);
    registry.register(Pow);
    registry.register(Sqrt);
    registry.register(InverseSqrt);

    // Common math
    registry.register(Abs);
    registry.register(Sign);
    registry.register(Floor);
    registry.register(Ceil);
    registry.register(Round);
    registry.register(Trunc);
    registry.register(Fract);
    registry.register(Min);
    registry.register(Max);
    registry.register(Clamp);
    registry.register(Saturate);

    // Interpolation
    registry.register(Lerp);
    registry.register(Mix);
    registry.register(Step);
    registry.register(Smoothstep);
    registry.register(InverseLerp);
    registry.register(Remap);
}

/// Creates a new registry with all standard scalar functions.
pub fn scalar_registry<T: Float + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_scalar(&mut registry);
    registry
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn eval_expr(expr: &str, vars: &[(&str, f32)]) -> f32 {
        let registry = scalar_registry();
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_constants() {
        assert!((eval_expr("pi()", &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval_expr("e()", &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval_expr("tau()", &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_trig() {
        assert!(eval_expr("sin(0)", &[]).abs() < 0.001);
        assert!((eval_expr("cos(0)", &[]) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval_expr("exp(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval_expr("ln(1)", &[]) - 0.0).abs() < 0.001);
        assert!((eval_expr("sqrt(16)", &[]) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval_expr("abs(-5)", &[]), 5.0);
        assert_eq!(eval_expr("floor(3.7)", &[]), 3.0);
        assert_eq!(eval_expr("ceil(3.2)", &[]), 4.0);
        assert_eq!(eval_expr("min(3, 7)", &[]), 3.0);
        assert_eq!(eval_expr("max(3, 7)", &[]), 7.0);
        assert_eq!(eval_expr("clamp(5, 0, 3)", &[]), 3.0);
        assert_eq!(eval_expr("saturate(1.5)", &[]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval_expr("lerp(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval_expr("mix(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval_expr("step(0.5, 0.3)", &[]), 0.0);
        assert_eq!(eval_expr("step(0.5, 0.7)", &[]), 1.0);
        assert!((eval_expr("smoothstep(0, 1, 0.5)", &[]) - 0.5).abs() < 0.1);
        assert_eq!(eval_expr("inverse_lerp(0, 10, 5)", &[]), 0.5);
    }

    #[test]
    fn test_remap() {
        assert_eq!(eval_expr("remap(5, 0, 10, 0, 100)", &[]), 50.0);
    }

    #[test]
    fn test_with_variables() {
        let v = eval_expr("sin(x * pi())", &[("x", 0.5)]);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f64() {
        let registry: FunctionRegistry<f64> = scalar_registry();
        let expr = Expr::parse("sin(x) + 1").unwrap();
        let vars: HashMap<String, f64> = [("x".to_string(), 0.0)].into();
        let result = eval(expr.ast(), &vars, &registry).unwrap();
        assert!((result - 1.0).abs() < 0.001);
    }
}
