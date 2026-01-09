//! Standard linalg functions: dot, cross, normalize, length, etc.

use crate::{LinalgFn, Signature, Type, Value};
use num_traits::Float;

// ============================================================================
// Dot product
// ============================================================================

/// Dot product: dot(a, b) -> scalar
pub struct Dot;

impl<T: Float> LinalgFn<T> for Dot {
    fn name(&self) -> &str {
        "dot"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Scalar(a[0] * b[0] + a[1] * b[1]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Cross product (3D only)
// ============================================================================

/// Cross product: cross(a, b) -> vec3
#[cfg(feature = "3d")]
pub struct Cross;

#[cfg(feature = "3d")]
impl<T: Float> LinalgFn<T> for Cross {
    fn name(&self) -> &str {
        "cross"
    }

    fn signatures(&self) -> Vec<Signature> {
        vec![Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        }]
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec3(a), Value::Vec3(b)) => Value::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Length
// ============================================================================

/// Vector length: length(v) -> scalar
pub struct Length;

impl<T: Float> LinalgFn<T> for Length {
    fn name(&self) -> &str {
        "length"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => Value::Scalar((v[0] * v[0] + v[1] * v[1]).sqrt()),
            #[cfg(feature = "3d")]
            Value::Vec3(v) => Value::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()),
            #[cfg(feature = "4d")]
            Value::Vec4(v) => {
                Value::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt())
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Normalize
// ============================================================================

/// Normalize vector: normalize(v) -> vec (same type, unit length)
pub struct Normalize;

impl<T: Float> LinalgFn<T> for Normalize {
    fn name(&self) -> &str {
        "normalize"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match &args[0] {
            Value::Vec2(v) => {
                let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
                Value::Vec2([v[0] / len, v[1] / len])
            }
            #[cfg(feature = "3d")]
            Value::Vec3(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                Value::Vec3([v[0] / len, v[1] / len, v[2] / len])
            }
            #[cfg(feature = "4d")]
            Value::Vec4(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                Value::Vec4([v[0] / len, v[1] / len, v[2] / len, v[3] / len])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Distance
// ============================================================================

/// Distance between two points: distance(a, b) -> scalar
pub struct Distance;

impl<T: Float> LinalgFn<T> for Distance {
    fn name(&self) -> &str {
        "distance"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Scalar,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Scalar,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Scalar,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                Value::Scalar((dx * dx + dy * dy).sqrt())
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                Value::Scalar((dx * dx + dy * dy + dz * dz).sqrt())
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let dz = a[2] - b[2];
                let dw = a[3] - b[3];
                Value::Scalar((dx * dx + dy * dy + dz * dz + dw * dw).sqrt())
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Reflect
// ============================================================================

/// Reflect vector: reflect(incident, normal) -> vec
/// Returns incident - 2 * dot(normal, incident) * normal
pub struct Reflect;

impl<T: Float> LinalgFn<T> for Reflect {
    fn name(&self) -> &str {
        "reflect"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(i), Value::Vec2(n)) => {
                let d = i[0] * n[0] + i[1] * n[1];
                let two = T::from(2.0).unwrap();
                Value::Vec2([i[0] - two * d * n[0], i[1] - two * d * n[1]])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(i), Value::Vec3(n)) => {
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2];
                let two = T::from(2.0).unwrap();
                Value::Vec3([
                    i[0] - two * d * n[0],
                    i[1] - two * d * n[1],
                    i[2] - two * d * n[2],
                ])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(i), Value::Vec4(n)) => {
                let d = i[0] * n[0] + i[1] * n[1] + i[2] * n[2] + i[3] * n[3];
                let two = T::from(2.0).unwrap();
                Value::Vec4([
                    i[0] - two * d * n[0],
                    i[1] - two * d * n[1],
                    i[2] - two * d * n[2],
                    i[3] - two * d * n[3],
                ])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Hadamard (element-wise multiply)
// ============================================================================

/// Element-wise vector multiply: hadamard(a, b) -> vec
pub struct Hadamard;

impl<T: Float> LinalgFn<T> for Hadamard {
    fn name(&self) -> &str {
        "hadamard"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1]) {
            (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] * b[0], a[1] * b[1]]),
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b)) => {
                Value::Vec3([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
            }
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b)) => {
                Value::Vec4([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
            }
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Lerp (linear interpolation for vectors)
// ============================================================================

/// Linear interpolation: lerp(a, b, t) -> vec
/// Returns a + (b - a) * t
pub struct Lerp;

impl<T: Float> LinalgFn<T> for Lerp {
    fn name(&self) -> &str {
        "lerp"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Scalar],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(a), Value::Vec2(b), Value::Scalar(t)) => {
                Value::Vec2([a[0] + (b[0] - a[0]) * *t, a[1] + (b[1] - a[1]) * *t])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b), Value::Scalar(t)) => Value::Vec3([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b), Value::Scalar(t)) => Value::Vec4([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
                a[3] + (b[3] - a[3]) * *t,
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Mix (alias for lerp, GLSL naming)
// ============================================================================

/// Linear interpolation (GLSL naming): mix(a, b, t) -> vec
pub struct Mix;

impl<T: Float> LinalgFn<T> for Mix {
    fn name(&self) -> &str {
        "mix"
    }

    fn signatures(&self) -> Vec<Signature> {
        let mut sigs = vec![Signature {
            args: vec![Type::Vec2, Type::Vec2, Type::Scalar],
            ret: Type::Vec2,
        }];
        #[cfg(feature = "3d")]
        sigs.push(Signature {
            args: vec![Type::Vec3, Type::Vec3, Type::Scalar],
            ret: Type::Vec3,
        });
        #[cfg(feature = "4d")]
        sigs.push(Signature {
            args: vec![Type::Vec4, Type::Vec4, Type::Scalar],
            ret: Type::Vec4,
        });
        sigs
    }

    fn call(&self, args: &[Value<T>]) -> Value<T> {
        match (&args[0], &args[1], &args[2]) {
            (Value::Vec2(a), Value::Vec2(b), Value::Scalar(t)) => {
                Value::Vec2([a[0] + (b[0] - a[0]) * *t, a[1] + (b[1] - a[1]) * *t])
            }
            #[cfg(feature = "3d")]
            (Value::Vec3(a), Value::Vec3(b), Value::Scalar(t)) => Value::Vec3([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
            ]),
            #[cfg(feature = "4d")]
            (Value::Vec4(a), Value::Vec4(b), Value::Scalar(t)) => Value::Vec4([
                a[0] + (b[0] - a[0]) * *t,
                a[1] + (b[1] - a[1]) * *t,
                a[2] + (b[2] - a[2]) * *t,
                a[3] + (b[3] - a[3]) * *t,
            ]),
            _ => unreachable!("signature mismatch"),
        }
    }
}

// ============================================================================
// Registry helper
// ============================================================================

use crate::FunctionRegistry;

/// Register all standard linalg functions.
pub fn register_linalg<T: Float + 'static>(registry: &mut FunctionRegistry<T>) {
    registry.register(Dot);
    #[cfg(feature = "3d")]
    registry.register(Cross);
    registry.register(Length);
    registry.register(Normalize);
    registry.register(Distance);
    registry.register(Reflect);
    registry.register(Hadamard);
    registry.register(Lerp);
    registry.register(Mix);
}

/// Create a new registry with all standard linalg functions.
pub fn linalg_registry<T: Float + 'static>() -> FunctionRegistry<T> {
    let mut registry = FunctionRegistry::new();
    register_linalg(&mut registry);
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

    fn eval_expr(expr: &str, vars: &[(&str, Value<f32>)]) -> Value<f32> {
        let expr = Expr::parse(expr).unwrap();
        let var_map: HashMap<String, Value<f32>> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let registry = linalg_registry();
        crate::eval(expr.ast(), &var_map, &registry).unwrap()
    }

    #[test]
    fn test_dot_vec2() {
        let result = eval_expr(
            "dot(a, b)",
            &[
                ("a", Value::Vec2([1.0, 2.0])),
                ("b", Value::Vec2([3.0, 4.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(11.0)); // 1*3 + 2*4 = 11
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_dot_vec3() {
        let result = eval_expr(
            "dot(a, b)",
            &[
                ("a", Value::Vec3([1.0, 2.0, 3.0])),
                ("b", Value::Vec3([4.0, 5.0, 6.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(32.0)); // 1*4 + 2*5 + 3*6 = 32
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = eval_expr(
            "cross(a, b)",
            &[
                ("a", Value::Vec3([1.0, 0.0, 0.0])),
                ("b", Value::Vec3([0.0, 1.0, 0.0])),
            ],
        );
        assert_eq!(result, Value::Vec3([0.0, 0.0, 1.0])); // x cross y = z
    }

    #[test]
    fn test_length_vec2() {
        let result = eval_expr("length(v)", &[("v", Value::Vec2([3.0, 4.0]))]);
        assert_eq!(result, Value::Scalar(5.0)); // 3-4-5 triangle
    }

    #[test]
    fn test_normalize_vec2() {
        let result = eval_expr("normalize(v)", &[("v", Value::Vec2([3.0, 4.0]))]);
        if let Value::Vec2(v) = result {
            assert!((v[0] - 0.6).abs() < 0.001);
            assert!((v[1] - 0.8).abs() < 0.001);
        } else {
            panic!("expected Vec2");
        }
    }

    #[test]
    fn test_distance_vec2() {
        let result = eval_expr(
            "distance(a, b)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([3.0, 4.0])),
            ],
        );
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_reflect_vec2() {
        // Reflect (1, -1) off horizontal surface with normal (0, 1)
        let result = eval_expr(
            "reflect(i, n)",
            &[
                ("i", Value::Vec2([1.0, -1.0])),
                ("n", Value::Vec2([0.0, 1.0])),
            ],
        );
        if let Value::Vec2(v) = result {
            assert!((v[0] - 1.0).abs() < 0.001);
            assert!((v[1] - 1.0).abs() < 0.001);
        } else {
            panic!("expected Vec2");
        }
    }

    #[test]
    fn test_hadamard_vec2() {
        let result = eval_expr(
            "hadamard(a, b)",
            &[
                ("a", Value::Vec2([2.0, 3.0])),
                ("b", Value::Vec2([4.0, 5.0])),
            ],
        );
        assert_eq!(result, Value::Vec2([8.0, 15.0]));
    }

    #[test]
    fn test_lerp_vec2() {
        let result = eval_expr(
            "lerp(a, b, t)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([10.0, 20.0])),
                ("t", Value::Scalar(0.5)),
            ],
        );
        assert_eq!(result, Value::Vec2([5.0, 10.0]));
    }

    #[test]
    fn test_mix_vec2() {
        let result = eval_expr(
            "mix(a, b, t)",
            &[
                ("a", Value::Vec2([0.0, 0.0])),
                ("b", Value::Vec2([10.0, 20.0])),
                ("t", Value::Scalar(0.25)),
            ],
        );
        assert_eq!(result, Value::Vec2([2.5, 5.0]));
    }
}
