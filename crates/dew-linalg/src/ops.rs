//! Binary and unary operations with type dispatch.

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

        // Vec2 + Vec2
        (Value::Vec2(a), Value::Vec2(b)) => Ok(Value::Vec2([a[0] + b[0], a[1] + b[1]])),

        // Vec3 + Vec3
        #[cfg(feature = "3d")]
        (Value::Vec3(a), Value::Vec3(b)) => {
            Ok(Value::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]]))
        }

        // Vec4 + Vec4
        #[cfg(feature = "4d")]
        (Value::Vec4(a), Value::Vec4(b)) => Ok(Value::Vec4([
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
        (Value::Vec2(a), Value::Vec2(b)) => Ok(Value::Vec2([a[0] - b[0], a[1] - b[1]])),
        #[cfg(feature = "3d")]
        (Value::Vec3(a), Value::Vec3(b)) => {
            Ok(Value::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]))
        }
        #[cfg(feature = "4d")]
        (Value::Vec4(a), Value::Vec4(b)) => Ok(Value::Vec4([
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
// Multiplication (scalar * scalar, vec * scalar, scalar * vec, mat * vec, mat * mat)
// ============================================================================

fn apply_mul<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        // Scalar * Scalar
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a * *b)),

        // Vec * Scalar (scaling)
        (Value::Vec2(v), Value::Scalar(s)) => Ok(Value::Vec2([v[0] * *s, v[1] * *s])),
        #[cfg(feature = "3d")]
        (Value::Vec3(v), Value::Scalar(s)) => Ok(Value::Vec3([v[0] * *s, v[1] * *s, v[2] * *s])),
        #[cfg(feature = "4d")]
        (Value::Vec4(v), Value::Scalar(s)) => {
            Ok(Value::Vec4([v[0] * *s, v[1] * *s, v[2] * *s, v[3] * *s]))
        }

        // Scalar * Vec (scaling, commutative)
        (Value::Scalar(s), Value::Vec2(v)) => Ok(Value::Vec2([*s * v[0], *s * v[1]])),
        #[cfg(feature = "3d")]
        (Value::Scalar(s), Value::Vec3(v)) => Ok(Value::Vec3([*s * v[0], *s * v[1], *s * v[2]])),
        #[cfg(feature = "4d")]
        (Value::Scalar(s), Value::Vec4(v)) => {
            Ok(Value::Vec4([*s * v[0], *s * v[1], *s * v[2], *s * v[3]]))
        }

        // Mat * Vec (column vector)
        (Value::Mat2(m), Value::Vec2(v)) => Ok(Value::Vec2(mat2_mul_vec2(m, v))),
        #[cfg(feature = "3d")]
        (Value::Mat3(m), Value::Vec3(v)) => Ok(Value::Vec3(mat3_mul_vec3(m, v))),
        #[cfg(feature = "4d")]
        (Value::Mat4(m), Value::Vec4(v)) => Ok(Value::Vec4(mat4_mul_vec4(m, v))),

        // Vec * Mat (row vector)
        (Value::Vec2(v), Value::Mat2(m)) => Ok(Value::Vec2(vec2_mul_mat2(v, m))),
        #[cfg(feature = "3d")]
        (Value::Vec3(v), Value::Mat3(m)) => Ok(Value::Vec3(vec3_mul_mat3(v, m))),
        #[cfg(feature = "4d")]
        (Value::Vec4(v), Value::Mat4(m)) => Ok(Value::Vec4(vec4_mul_mat4(v, m))),

        // Mat * Scalar (scaling)
        (Value::Mat2(m), Value::Scalar(s)) => {
            Ok(Value::Mat2([m[0] * *s, m[1] * *s, m[2] * *s, m[3] * *s]))
        }
        #[cfg(feature = "3d")]
        (Value::Mat3(m), Value::Scalar(s)) => Ok(Value::Mat3(std::array::from_fn(|i| m[i] * *s))),
        #[cfg(feature = "4d")]
        (Value::Mat4(m), Value::Scalar(s)) => Ok(Value::Mat4(std::array::from_fn(|i| m[i] * *s))),

        // Scalar * Mat (scaling, commutative)
        (Value::Scalar(s), Value::Mat2(m)) => {
            Ok(Value::Mat2([*s * m[0], *s * m[1], *s * m[2], *s * m[3]]))
        }
        #[cfg(feature = "3d")]
        (Value::Scalar(s), Value::Mat3(m)) => Ok(Value::Mat3(std::array::from_fn(|i| *s * m[i]))),
        #[cfg(feature = "4d")]
        (Value::Scalar(s), Value::Mat4(m)) => Ok(Value::Mat4(std::array::from_fn(|i| *s * m[i]))),

        // Mat * Mat
        (Value::Mat2(a), Value::Mat2(b)) => Ok(Value::Mat2(mat2_mul_mat2(a, b))),
        #[cfg(feature = "3d")]
        (Value::Mat3(a), Value::Mat3(b)) => Ok(Value::Mat3(mat3_mul_mat3(a, b))),
        #[cfg(feature = "4d")]
        (Value::Mat4(a), Value::Mat4(b)) => Ok(Value::Mat4(mat4_mul_mat4(a, b))),

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Mul,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Division (element-wise for vectors, scalar only for matrices)
// ============================================================================

fn apply_div<T: Float>(left: Value<T>, right: Value<T>) -> Result<Value<T>, Error> {
    match (&left, &right) {
        (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(*a / *b)),

        // Vec / Scalar
        (Value::Vec2(v), Value::Scalar(s)) => Ok(Value::Vec2([v[0] / *s, v[1] / *s])),
        #[cfg(feature = "3d")]
        (Value::Vec3(v), Value::Scalar(s)) => Ok(Value::Vec3([v[0] / *s, v[1] / *s, v[2] / *s])),
        #[cfg(feature = "4d")]
        (Value::Vec4(v), Value::Scalar(s)) => {
            Ok(Value::Vec4([v[0] / *s, v[1] / *s, v[2] / *s, v[3] / *s]))
        }

        _ => Err(Error::BinaryTypeMismatch {
            op: BinOp::Div,
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

// ============================================================================
// Power (scalar only)
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
        Value::Vec2(v) => Ok(Value::Vec2([-v[0], -v[1]])),
        #[cfg(feature = "3d")]
        Value::Vec3(v) => Ok(Value::Vec3([-v[0], -v[1], -v[2]])),
        #[cfg(feature = "4d")]
        Value::Vec4(v) => Ok(Value::Vec4([-v[0], -v[1], -v[2], -v[3]])),
        Value::Mat2(m) => Ok(Value::Mat2([-m[0], -m[1], -m[2], -m[3]])),
        #[cfg(feature = "3d")]
        Value::Mat3(m) => Ok(Value::Mat3(std::array::from_fn(|i| -m[i]))),
        #[cfg(feature = "4d")]
        Value::Mat4(m) => Ok(Value::Mat4(std::array::from_fn(|i| -m[i]))),
    }
}

// ============================================================================
// Matrix-vector multiplication helpers
// ============================================================================

/// Mat2 * Vec2 (column vector, column-major storage)
fn mat2_mul_vec2<T: Float>(m: &[T; 4], v: &[T; 2]) -> [T; 2] {
    // m = [c0r0, c0r1, c1r0, c1r1]
    // result[i] = sum_j(m[j*2+i] * v[j])
    [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]
}

/// Vec2 * Mat2 (row vector, column-major storage)
fn vec2_mul_mat2<T: Float>(v: &[T; 2], m: &[T; 4]) -> [T; 2] {
    // result[j] = sum_i(v[i] * m[j*2+i])
    [v[0] * m[0] + v[1] * m[1], v[0] * m[2] + v[1] * m[3]]
}

/// Mat3 * Vec3 (column vector, column-major storage)
#[cfg(feature = "3d")]
fn mat3_mul_vec3<T: Float>(m: &[T; 9], v: &[T; 3]) -> [T; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

/// Vec3 * Mat3 (row vector, column-major storage)
#[cfg(feature = "3d")]
fn vec3_mul_mat3<T: Float>(v: &[T; 3], m: &[T; 9]) -> [T; 3] {
    [
        v[0] * m[0] + v[1] * m[1] + v[2] * m[2],
        v[0] * m[3] + v[1] * m[4] + v[2] * m[5],
        v[0] * m[6] + v[1] * m[7] + v[2] * m[8],
    ]
}

/// Mat4 * Vec4 (column vector, column-major storage)
#[cfg(feature = "4d")]
fn mat4_mul_vec4<T: Float>(m: &[T; 16], v: &[T; 4]) -> [T; 4] {
    [
        m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
        m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
        m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
        m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
    ]
}

/// Vec4 * Mat4 (row vector, column-major storage)
#[cfg(feature = "4d")]
fn vec4_mul_mat4<T: Float>(v: &[T; 4], m: &[T; 16]) -> [T; 4] {
    [
        v[0] * m[0] + v[1] * m[1] + v[2] * m[2] + v[3] * m[3],
        v[0] * m[4] + v[1] * m[5] + v[2] * m[6] + v[3] * m[7],
        v[0] * m[8] + v[1] * m[9] + v[2] * m[10] + v[3] * m[11],
        v[0] * m[12] + v[1] * m[13] + v[2] * m[14] + v[3] * m[15],
    ]
}

// ============================================================================
// Matrix-matrix multiplication helpers
// ============================================================================

/// Mat2 * Mat2 (column-major)
fn mat2_mul_mat2<T: Float>(a: &[T; 4], b: &[T; 4]) -> [T; 4] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
    ]
}

/// Mat3 * Mat3 (column-major)
#[cfg(feature = "3d")]
fn mat3_mul_mat3<T: Float>(a: &[T; 9], b: &[T; 9]) -> [T; 9] {
    let mut result = [T::zero(); 9];
    for col in 0..3 {
        for row in 0..3 {
            result[col * 3 + row] =
                a[row] * b[col * 3] + a[3 + row] * b[col * 3 + 1] + a[6 + row] * b[col * 3 + 2];
        }
    }
    result
}

/// Mat4 * Mat4 (column-major)
#[cfg(feature = "4d")]
fn mat4_mul_mat4<T: Float>(a: &[T; 16], b: &[T; 16]) -> [T; 16] {
    let mut result = [T::zero(); 16];
    for col in 0..4 {
        for row in 0..4 {
            result[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat2_mul_vec2() {
        // Identity matrix
        let identity = [1.0, 0.0, 0.0, 1.0];
        let v = [3.0, 4.0];
        assert_eq!(mat2_mul_vec2(&identity, &v), [3.0, 4.0]);

        // Rotation 90 degrees (column-major)
        let rot90 = [0.0, 1.0, -1.0, 0.0];
        let v = [1.0, 0.0];
        let result = mat2_mul_vec2(&rot90, &v);
        assert!((result[0] - 0.0).abs() < 0.001);
        assert!((result[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_vec2_mul_mat2() {
        // Identity
        let identity = [1.0, 0.0, 0.0, 1.0];
        let v = [3.0, 4.0];
        assert_eq!(vec2_mul_mat2(&v, &identity), [3.0, 4.0]);

        // vec * mat should differ from mat * vec for non-symmetric matrices
        let m = [1.0, 2.0, 3.0, 4.0]; // column-major: [[1,2], [3,4]]
        let v = [1.0, 0.0];
        // mat * vec: [1*1 + 3*0, 2*1 + 4*0] = [1, 2]
        assert_eq!(mat2_mul_vec2(&m, &v), [1.0, 2.0]);
        // vec * mat: [1*1 + 0*2, 1*3 + 0*4] = [1, 3]
        assert_eq!(vec2_mul_mat2(&v, &m), [1.0, 3.0]);
    }

    #[test]
    fn test_mat2_mul_mat2() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mat2_mul_mat2(&identity, &a), a);
        assert_eq!(mat2_mul_mat2(&a, &identity), a);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_mat3_mul_vec3() {
        // Identity
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v = [1.0, 2.0, 3.0];
        assert_eq!(mat3_mul_vec3(&identity, &v), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_f64_operations() {
        // Ensure it works with f64 too
        let a: Value<f64> = Value::Vec2([1.0, 2.0]);
        let b: Value<f64> = Value::Vec2([3.0, 4.0]);
        let result = apply_binop(BinOp::Add, a, b).unwrap();
        assert_eq!(result, Value::Vec2([4.0, 6.0]));
    }
}
