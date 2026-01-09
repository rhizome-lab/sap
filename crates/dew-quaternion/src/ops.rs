//! Binary and unary operations for quaternions.
//!
//! Quaternion uses [x, y, z, w] order (scalar last).

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
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a + *b)),

        (Value::Vec3(a), Value::Vec3(b)) => {
            Ok(Value::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]]))
        }

        (Value::Quaternion(a), Value::Quaternion(b)) => Ok(Value::Quaternion([
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
        ])),

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Add,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Subtraction
// ============================================================================

fn apply_sub<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a - *b)),

        (Value::Vec3(a), Value::Vec3(b)) => {
            Ok(Value::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]))
        }

        (Value::Quaternion(a), Value::Quaternion(b)) => Ok(Value::Quaternion([
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
            a[3] - b[3],
        ])),

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Sub,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Multiplication
// ============================================================================

fn apply_mul<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar * Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a * *b)),

        // Scalar * Vec3
        (Value::Scalar(s), Value::Vec3(v)) => Ok(Value::Vec3([*s * v[0], *s * v[1], *s * v[2]])),
        (Value::Vec3(v), Value::Scalar(s)) => Ok(Value::Vec3([v[0] * *s, v[1] * *s, v[2] * *s])),

        // Scalar * Quaternion
        (Value::Scalar(s), Value::Quaternion(q)) => Ok(Value::Quaternion([
            *s * q[0],
            *s * q[1],
            *s * q[2],
            *s * q[3],
        ])),
        (Value::Quaternion(q), Value::Scalar(s)) => Ok(Value::Quaternion([
            q[0] * *s,
            q[1] * *s,
            q[2] * *s,
            q[3] * *s,
        ])),

        // Quaternion * Quaternion (Hamilton product)
        // q1 = [x1, y1, z1, w1], q2 = [x2, y2, z2, w2]
        // result.w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        // result.x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        // result.y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        // result.z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        (Value::Quaternion(a), Value::Quaternion(b)) => {
            let (x1, y1, z1, w1) = (a[0], a[1], a[2], a[3]);
            let (x2, y2, z2, w2) = (b[0], b[1], b[2], b[3]);

            Ok(Value::Quaternion([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, // x
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, // y
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, // z
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, // w
            ]))
        }

        // Quaternion * Vec3 (rotate vector)
        // v' = q * v * q^(-1), but we use the optimized formula
        (Value::Quaternion(q), Value::Vec3(v)) => Ok(Value::Vec3(rotate_vec3_by_quat(v, q))),

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Mul,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

/// Rotate a vec3 by a quaternion using the optimized formula.
/// v' = v + 2 * w * (q_xyz × v) + 2 * (q_xyz × (q_xyz × v))
fn rotate_vec3_by_quat<T: Float>(v: &[T; 3], q: &[T; 4]) -> [T; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    let two = T::from(2.0).unwrap();

    // t = 2 * (q_xyz × v)
    let tx = two * (qy * v[2] - qz * v[1]);
    let ty = two * (qz * v[0] - qx * v[2]);
    let tz = two * (qx * v[1] - qy * v[0]);

    // v' = v + w * t + (q_xyz × t)
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

// ============================================================================
// Division
// ============================================================================

fn apply_div<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a / *b)),

        (Value::Vec3(v), Value::Scalar(s)) => Ok(Value::Vec3([v[0] / *s, v[1] / *s, v[2] / *s])),

        (Value::Quaternion(q), Value::Scalar(s)) => Ok(Value::Quaternion([
            q[0] / *s,
            q[1] / *s,
            q[2] / *s,
            q[3] / *s,
        ])),

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Div,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Power
// ============================================================================

fn apply_pow<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.powf(*b))),

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
        Value::Vec3(v) => Ok(Value::Vec3([-v[0], -v[1], -v[2]])),
        Value::Quaternion(q) => Ok(Value::Quaternion([-q[0], -q[1], -q[2], -q[3]])),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn test_quaternion_mul_identity() {
        // q * identity = q
        let q = Value::Quaternion([1.0_f32, 2.0, 3.0, 4.0]);
        let identity = Value::Quaternion([0.0, 0.0, 0.0, 1.0]);
        let result = apply_binop(BinOp::Mul, q, identity).unwrap();
        assert_eq!(result, Value::Quaternion([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_quaternion_mul_inverse() {
        // For unit quaternion q, q * conj(q) = identity (up to normalization)
        // Let's test with 90° rotation around Z axis
        let angle = std::f32::consts::FRAC_PI_4; // half angle
        let q = Value::Quaternion([0.0, 0.0, angle.sin(), angle.cos()]);
        let q_conj = Value::Quaternion([0.0, 0.0, -angle.sin(), angle.cos()]);
        let result = apply_binop(BinOp::Mul, q, q_conj).unwrap();
        if let Value::Quaternion(r) = result {
            assert!(approx_eq(r[0], 0.0));
            assert!(approx_eq(r[1], 0.0));
            assert!(approx_eq(r[2], 0.0));
            assert!(approx_eq(r[3], 1.0));
        } else {
            panic!("expected quaternion");
        }
    }

    #[test]
    fn test_rotate_vec3() {
        // Rotate [1, 0, 0] by 90° around Z axis -> [0, 1, 0]
        let angle = std::f32::consts::FRAC_PI_4; // half angle for quaternion
        let q = Value::Quaternion([0.0, 0.0, angle.sin(), angle.cos()]);
        let v = Value::Vec3([1.0, 0.0, 0.0]);
        let result = apply_binop(BinOp::Mul, q, v).unwrap();
        if let Value::Vec3(r) = result {
            assert!(approx_eq(r[0], 0.0));
            assert!(approx_eq(r[1], 1.0));
            assert!(approx_eq(r[2], 0.0));
        } else {
            panic!("expected vec3");
        }
    }
}
