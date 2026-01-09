//! Binary and unary operations for complex numbers.

use crate::{Error, Value};
use num_traits::Float;
use rhizome_dew_core::{BinOp, UnaryOp};

/// Apply a binary operation to two values.
pub fn apply_binop<T: Float>(
    op: BinOp,
    left: Value<T>,
    right: Value<T>,
) -> Result<Value<T>, Error> {
    match op {
        BinOp::Add => apply_add(left, right),
        BinOp::Sub => apply_sub(left, right),
        BinOp::Mul => apply_mul(left, right),
        BinOp::Div => apply_div(left, right),
        BinOp::Pow => apply_pow(left, right),
    }
}

/// Apply a unary operation to a value.
pub fn apply_unaryop<T: Float>(op: UnaryOp, val: Value<T>) -> Result<Value<T>, Error> {
    match op {
        UnaryOp::Neg => apply_neg(val),
    }
}

// ============================================================================
// Addition
// ============================================================================

fn apply_add<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar + Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a + *b)),

        // Complex + Complex
        (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex([a[0] + b[0], a[1] + b[1]])),

        // Scalar + Complex (promote scalar to complex)
        (Value::Scalar(s), Value::Complex(c)) => Ok(Value::Complex([*s + c[0], c[1]])),

        // Complex + Scalar
        (Value::Complex(c), Value::Scalar(s)) => Ok(Value::Complex([c[0] + *s, c[1]])),
    }
}

// ============================================================================
// Subtraction
// ============================================================================

fn apply_sub<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a - *b)),

        (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex([a[0] - b[0], a[1] - b[1]])),

        (Value::Scalar(s), Value::Complex(c)) => Ok(Value::Complex([*s - c[0], -c[1]])),

        (Value::Complex(c), Value::Scalar(s)) => Ok(Value::Complex([c[0] - *s, c[1]])),
    }
}

// ============================================================================
// Multiplication
// ============================================================================

fn apply_mul<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar * Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a * *b)),

        // Complex * Complex: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        (Value::Complex(a), Value::Complex(b)) => {
            let re = a[0] * b[0] - a[1] * b[1];
            let im = a[0] * b[1] + a[1] * b[0];
            Ok(Value::Complex([re, im]))
        }

        // Scalar * Complex
        (Value::Scalar(s), Value::Complex(c)) => Ok(Value::Complex([*s * c[0], *s * c[1]])),

        // Complex * Scalar
        (Value::Complex(c), Value::Scalar(s)) => Ok(Value::Complex([c[0] * *s, c[1] * *s])),
    }
}

// ============================================================================
// Division
// ============================================================================

fn apply_div<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar / Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a / *b)),

        // Complex / Complex: (a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
        (Value::Complex(a), Value::Complex(b)) => {
            let denom = b[0] * b[0] + b[1] * b[1];
            let re = (a[0] * b[0] + a[1] * b[1]) / denom;
            let im = (a[1] * b[0] - a[0] * b[1]) / denom;
            Ok(Value::Complex([re, im]))
        }

        // Scalar / Complex
        (Value::Scalar(s), Value::Complex(c)) => {
            // s / (a + bi) = s(a - bi) / (a² + b²)
            let denom = c[0] * c[0] + c[1] * c[1];
            let re = *s * c[0] / denom;
            let im = -*s * c[1] / denom;
            Ok(Value::Complex([re, im]))
        }

        // Complex / Scalar
        (Value::Complex(c), Value::Scalar(s)) => Ok(Value::Complex([c[0] / *s, c[1] / *s])),
    }
}

// ============================================================================
// Power
// ============================================================================

fn apply_pow<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar ^ Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.powf(*b))),

        // Complex ^ Scalar (integer-ish powers common)
        // z^n = r^n * e^(i*n*θ) where z = r*e^(iθ)
        (Value::Complex(c), Value::Scalar(n)) => {
            let r = (c[0] * c[0] + c[1] * c[1]).sqrt();
            let theta = c[1].atan2(c[0]);
            let r_n = r.powf(*n);
            let theta_n = theta * *n;
            Ok(Value::Complex([r_n * theta_n.cos(), r_n * theta_n.sin()]))
        }

        // Other cases not supported for now
        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Pow,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Negation
// ============================================================================

fn apply_neg<T: Float>(val: Value<T>) -> Result<Value<T>, Error> {
    match val {
        Value::Scalar(v) => Ok(Value::Scalar(-v)),
        Value::Complex(c) => Ok(Value::Complex([-c[0], -c[1]])),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_mul() {
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let a = Value::Complex([1.0_f32, 2.0]);
        let b = Value::Complex([3.0, 4.0]);
        let result = apply_binop(BinOp::Mul, a, b).unwrap();
        assert_eq!(result, Value::Complex([-5.0, 10.0]));
    }

    #[test]
    fn test_complex_div() {
        // (1+2i) / (1+2i) = 1
        let a = Value::Complex([1.0_f32, 2.0]);
        let b = Value::Complex([1.0, 2.0]);
        let result = apply_binop(BinOp::Div, a, b).unwrap();
        if let Value::Complex(c) = result {
            assert!((c[0] - 1.0).abs() < 0.0001);
            assert!(c[1].abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }

    #[test]
    fn test_complex_pow() {
        // i^2 = -1
        let i = Value::Complex([0.0_f32, 1.0]);
        let two = Value::Scalar(2.0);
        let result = apply_binop(BinOp::Pow, i, two).unwrap();
        if let Value::Complex(c) = result {
            assert!((c[0] - (-1.0)).abs() < 0.0001);
            assert!(c[1].abs() < 0.0001);
        } else {
            panic!("expected complex");
        }
    }
}
