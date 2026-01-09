//! Quaternion functions: normalize, conjugate, slerp, rotate, etc.
//!
//! Quaternion uses [x, y, z, w] order (scalar last).

use crate::{FunctionRegistry, QuaternionFn, Signature, Type, Value};
use num_traits::Float;

// ============================================================================
// Conjugate
// ============================================================================

/// Quaternion conjugate: conj([x, y, z, w]) = [-x, -y, -z, w]
pub struct Conj;

impl<T: Float> QuaternionFn<T> for Conj {
    fn name(&self) -> &str {
        "conj"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Quaternion],
            ret: Type::Quaternion,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Quaternion(q) => Value::Quaternion([-q[0], -q[1], -q[2], q[3]]),
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Length
// ============================================================================

/// Quaternion magnitude: length(q) = sqrt(x² + y² + z² + w²)
pub struct Length;

impl<T: Float> QuaternionFn<T> for Length {
    fn name(&self) -> &str {
        "length"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Vec3],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Quaternion],
                ret: Type::Scalar,
            },
        ]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec3(v) => Value::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()),
            Value::Quaternion(q) => {
                Value::Scalar((q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt())
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Normalize
// ============================================================================

/// Normalize to unit quaternion/vector
pub struct Normalize;

impl<T: Float> QuaternionFn<T> for Normalize {
    fn name(&self) -> &str {
        "normalize"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Vec3],
                ret: Type::Vec3,
            },
            Signature {
                args: vec![Type::Quaternion],
                ret: Type::Quaternion,
            },
        ]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec3(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                Value::Vec3([v[0] / len, v[1] / len, v[2] / len])
            }
            Value::Quaternion(q) => {
                let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                Value::Quaternion([q[0] / len, q[1] / len, q[2] / len, q[3] / len])
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Inverse
// ============================================================================

/// Quaternion inverse: inverse(q) = conj(q) / |q|²
/// For unit quaternions, inverse = conjugate
pub struct Inverse;

impl<T: Float> QuaternionFn<T> for Inverse {
    fn name(&self) -> &str {
        "inverse"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Quaternion],
            ret: Type::Quaternion,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Quaternion(q) => {
                let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                Value::Quaternion([
                    -q[0] / norm_sq,
                    -q[1] / norm_sq,
                    -q[2] / norm_sq,
                    q[3] / norm_sq,
                ])
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Dot product
// ============================================================================

/// Dot product (4D for quaternions, 3D for vectors)
pub struct Dot;

impl<T: Float> QuaternionFn<T> for Dot {
    fn name(&self) -> &str {
        "dot"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Vec3, Type::Vec3],
                ret: Type::Scalar,
            },
            Signature {
                args: vec![Type::Quaternion, Type::Quaternion],
                ret: Type::Scalar,
            },
        ]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            }
            (Value::Quaternion(a), Value::Quaternion(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Lerp (linear interpolation)
// ============================================================================

/// Linear interpolation (use slerp for rotations)
pub struct Lerp;

impl<T: Float> QuaternionFn<T> for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![
            Signature {
                args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
                ret: Type::Vec3,
            },
            Signature {
                args: vec![Type::Quaternion, Type::Quaternion, Type::Scalar],
                ret: Type::Quaternion,
            },
        ]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec3(a), Value::Vec3(b), Value::Scalar(t)) => Value::Vec3([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
            ]),
            (Value::Quaternion(a), Value::Quaternion(b), Value::Scalar(t)) => Value::Quaternion([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
                a[3] + (b[3] - a[3]) * *t,
            ]),
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Slerp (spherical linear interpolation)
// ============================================================================

/// Spherical linear interpolation for quaternions
pub struct Slerp;

impl<T: Float> QuaternionFn<T> for Slerp {
    fn name(&self) -> &str {
        "slerp"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Quaternion, Type::Quaternion, Type::Scalar],
            ret: Type::Quaternion,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Quaternion(a), Value::Quaternion(b), Value::Scalar(t)) => {
                Value::Quaternion(slerp_impl(a, b, *t))
            }
            _ => unreachable!(),
        }
    }
}

fn slerp_impl<T: Float>(a: &[T; 4], b: &[T; 4], t: T) -> [T; 4] {
    // Compute cosine of angle between quaternions
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If dot < 0, negate one quaternion to take shorter path
    let mut b = *b;
    if dot < T::zero() {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }

    // Clamp dot to valid range for acos
    let one = T::one();
    if dot > one {
        dot = one;
    }

    // If quaternions are very close, use linear interpolation
    let threshold = T::from(0.9995).unwrap();
    if dot > threshold {
        // Linear interpolation
        let result = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
            a[3] + (b[3] - a[3]) * t,
        ];
        // Normalize
        let len = (result[0] * result[0]
            + result[1] * result[1]
            + result[2] * result[2]
            + result[3] * result[3])
            .sqrt();
        return [
            result[0] / len,
            result[1] / len,
            result[2] / len,
            result[3] / len,
        ];
    }

    // Spherical interpolation
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let s0 = ((one - t) * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;

    [
        a[0] * s0 + b[0] * s1,
        a[1] * s0 + b[1] * s1,
        a[2] * s0 + b[2] * s1,
        a[3] * s0 + b[3] * s1,
    ]
}

// ============================================================================
// Axis-Angle construction
// ============================================================================

/// Create quaternion from axis and angle: axis_angle(axis, angle)
pub struct AxisAngle;

impl<T: Float> QuaternionFn<T> for AxisAngle {
    fn name(&self) -> &str {
        "axis_angle"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Scalar],
            ret: Type::Quaternion,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(axis), Value::Scalar(angle)) => {
                let half_angle = *angle / T::from(2.0).unwrap();
                let s = half_angle.sin();
                let c = half_angle.cos();
                // Normalize axis
                let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
                Value::Quaternion([axis[0] / len * s, axis[1] / len * s, axis[2] / len * s, c])
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Rotate vector
// ============================================================================

/// Rotate a vector by a quaternion: rotate(vec, quat)
pub struct Rotate;

impl<T: Float> QuaternionFn<T> for Rotate {
    fn name(&self) -> &str {
        "rotate"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Quaternion],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(v), Value::Quaternion(q)) => Value::Vec3(rotate_vec3_by_quat(v, q)),
            _ => unreachable!(),
        }
    }
}

/// Rotate a vec3 by a quaternion using the optimized formula.
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
// Registry helper
// ============================================================================

/// Register all standard quaternion functions.
pub fn register_quaternion<T: Float + 'static>(registry: &mut FunctionRegistry<T>) {
    registry.register(Conj);
    registry.register(Length);
    registry.register(Normalize);
    registry.register(Inverse);
    registry.register(Dot);
    registry.register(Lerp);
    registry.register(Slerp);
    registry.register(AxisAngle);
    registry.register(Rotate);
}

/// Create a new registry with all standard quaternion functions.
pub fn quaternion_registry<T: Float + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_quaternion(&mut registry);
    registry
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;
    use std::collections::HashMap;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Value<f32> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = quaternion_registry();
        crate::eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_conj() {
        let result = eval_expr("conj(q)", &[("q", Value::Quaternion([1.0, 2.0, 3.0, 4.0]))]);
        assert_eq!(result, Value::Quaternion([-1.0, -2.0, -3.0, 4.0]));
    }

    #[test]
    fn test_normalize() {
        let result = eval_expr(
            "normalize(q)",
            &[("q", Value::Quaternion([0.0, 0.0, 0.0, 2.0]))],
        );
        assert_eq!(result, Value::Quaternion([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_length() {
        let result = eval_expr(
            "length(q)",
            &[("q", Value::Quaternion([0.0, 0.0, 3.0, 4.0]))],
        );
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_dot() {
        let result = eval_expr(
            "dot(a, b)",
            &[
                ("a", Value::Quaternion([1.0, 0.0, 0.0, 0.0])),
                ("b", Value::Quaternion([1.0, 0.0, 0.0, 0.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(1.0));
    }

    #[test]
    fn test_axis_angle() {
        // 90° rotation around Z axis
        let result = eval_expr(
            "axis_angle(axis, angle)",
            &[
                ("axis", Value::Vec3([0.0, 0.0, 1.0])),
                ("angle", Value::Scalar(std::f32::consts::FRAC_PI_2)),
            ],
        );
        if let Value::Quaternion(q) = result {
            // half angle = 45°, sin(45°) ≈ 0.707, cos(45°) ≈ 0.707
            assert!(approx_eq(q[0], 0.0));
            assert!(approx_eq(q[1], 0.0));
            assert!(approx_eq(q[2], std::f32::consts::FRAC_PI_4.sin()));
            assert!(approx_eq(q[3], std::f32::consts::FRAC_PI_4.cos()));
        } else {
            panic!("expected quaternion");
        }
    }

    #[test]
    fn test_rotate() {
        // Rotate [1, 0, 0] by 90° around Z axis -> [0, 1, 0]
        let half_angle = std::f32::consts::FRAC_PI_4;
        let result = eval_expr(
            "rotate(v, q)",
            &[
                ("v", Value::Vec3([1.0, 0.0, 0.0])),
                (
                    "q",
                    Value::Quaternion([0.0, 0.0, half_angle.sin(), half_angle.cos()]),
                ),
            ],
        );
        if let Value::Vec3(v) = result {
            assert!(approx_eq(v[0], 0.0));
            assert!(approx_eq(v[1], 1.0));
            assert!(approx_eq(v[2], 0.0));
        } else {
            panic!("expected vec3");
        }
    }

    #[test]
    fn test_slerp() {
        // Slerp between identity and 180° rotation should give 90° at t=0.5
        let identity = Value::Quaternion([0.0, 0.0, 0.0, 1.0]);
        // 180° around Z = [0, 0, 1, 0]
        let half_turn = Value::Quaternion([0.0, 0.0, 1.0, 0.0]);
        let result = eval_expr(
            "slerp(a, b, t)",
            &[("a", identity), ("b", half_turn), ("t", Value::Scalar(0.5))],
        );
        if let Value::Quaternion(q) = result {
            // Should be 90° rotation: [0, 0, sin(45°), cos(45°)]
            assert!(approx_eq(q[0], 0.0));
            assert!(approx_eq(q[1], 0.0));
            assert!(approx_eq(q[2], std::f32::consts::FRAC_PI_4.sin()));
            assert!(approx_eq(q[3], std::f32::consts::FRAC_PI_4.cos()));
        } else {
            panic!("expected quaternion");
        }
    }
}
