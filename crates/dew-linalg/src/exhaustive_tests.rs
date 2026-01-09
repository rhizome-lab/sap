//! Exhaustive tests for linalg functions across all backends.
//!
//! Tests eval, Lua execution, and Cranelift JIT compilation.
//! WGSL only generates code (no runtime), so we test code patterns.

use crate::{Value, eval, linalg_registry};
#[cfg(any(feature = "lua", feature = "wgsl"))]
use crate::Type;
use rhizome_dew_core::Expr;
use std::collections::HashMap;

const EPSILON: f32 = 0.0001;

fn assert_close(actual: f32, expected: f32, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < EPSILON || (expected.abs() > 1.0 && diff / expected.abs() < EPSILON),
        "{}: expected {}, got {} (diff: {})",
        context,
        expected,
        actual,
        diff
    );
}

fn assert_vec2_close(actual: &[f32; 2], expected: &[f32; 2], context: &str) {
    for i in 0..2 {
        assert_close(actual[i], expected[i], &format!("{} [{}]", context, i));
    }
}

#[cfg(feature = "3d")]
fn assert_vec3_close(actual: &[f32; 3], expected: &[f32; 3], context: &str) {
    for i in 0..3 {
        assert_close(actual[i], expected[i], &format!("{} [{}]", context, i));
    }
}

// ============================================================================
// Eval helpers
// ============================================================================

fn eval_linalg(expr_str: &str, vars: &[(&str, Value<f32>)]) -> Result<Value<f32>, crate::Error> {
    let expr = Expr::parse(expr_str).unwrap();
    let var_map: HashMap<String, Value<f32>> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let registry = linalg_registry();
    eval(expr.ast(), &var_map, &registry)
}

// ============================================================================
// Scalar operations
// ============================================================================

#[test]
fn test_scalar_add() {
    let result = eval_linalg(
        "a + b",
        &[("a", Value::Scalar(3.0)), ("b", Value::Scalar(4.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(7.0));
}

#[test]
fn test_scalar_sub() {
    let result = eval_linalg(
        "a - b",
        &[("a", Value::Scalar(10.0)), ("b", Value::Scalar(3.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(7.0));
}

#[test]
fn test_scalar_mul() {
    let result = eval_linalg(
        "a * b",
        &[("a", Value::Scalar(3.0)), ("b", Value::Scalar(4.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(12.0));
}

#[test]
fn test_scalar_div() {
    let result = eval_linalg(
        "a / b",
        &[("a", Value::Scalar(12.0)), ("b", Value::Scalar(3.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(4.0));
}

#[test]
fn test_scalar_pow() {
    let result = eval_linalg(
        "a ^ b",
        &[("a", Value::Scalar(2.0)), ("b", Value::Scalar(3.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(8.0));
}

#[test]
fn test_scalar_neg() {
    let result = eval_linalg("-a", &[("a", Value::Scalar(5.0))]).unwrap();
    assert_eq!(result, Value::Scalar(-5.0));
}

// ============================================================================
// Vec2 operations
// ============================================================================

#[test]
fn test_vec2_add() {
    let result = eval_linalg(
        "a + b",
        &[
            ("a", Value::Vec2([1.0, 2.0])),
            ("b", Value::Vec2([3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([4.0, 6.0]));
}

#[test]
fn test_vec2_sub() {
    let result = eval_linalg(
        "a - b",
        &[
            ("a", Value::Vec2([5.0, 7.0])),
            ("b", Value::Vec2([2.0, 3.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([3.0, 4.0]));
}

#[test]
fn test_vec2_scalar_mul() {
    let result = eval_linalg(
        "v * s",
        &[("v", Value::Vec2([2.0, 3.0])), ("s", Value::Scalar(2.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([4.0, 6.0]));
}

#[test]
fn test_scalar_vec2_mul() {
    let result = eval_linalg(
        "s * v",
        &[("s", Value::Scalar(2.0)), ("v", Value::Vec2([2.0, 3.0]))],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([4.0, 6.0]));
}

#[test]
fn test_vec2_scalar_div() {
    let result = eval_linalg(
        "v / s",
        &[("v", Value::Vec2([4.0, 6.0])), ("s", Value::Scalar(2.0))],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([2.0, 3.0]));
}

#[test]
fn test_vec2_neg() {
    let result = eval_linalg("-v", &[("v", Value::Vec2([1.0, -2.0]))]).unwrap();
    assert_eq!(result, Value::Vec2([-1.0, 2.0]));
}

// ============================================================================
// Vec3 operations (feature = "3d")
// ============================================================================

#[cfg(feature = "3d")]
#[test]
fn test_vec3_add() {
    let result = eval_linalg(
        "a + b",
        &[
            ("a", Value::Vec3([1.0, 2.0, 3.0])),
            ("b", Value::Vec3([4.0, 5.0, 6.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec3([5.0, 7.0, 9.0]));
}

#[cfg(feature = "3d")]
#[test]
fn test_vec3_sub() {
    let result = eval_linalg(
        "a - b",
        &[
            ("a", Value::Vec3([5.0, 7.0, 9.0])),
            ("b", Value::Vec3([1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec3([4.0, 5.0, 6.0]));
}

#[cfg(feature = "3d")]
#[test]
fn test_vec3_scalar_mul() {
    let result = eval_linalg(
        "v * s",
        &[
            ("v", Value::Vec3([1.0, 2.0, 3.0])),
            ("s", Value::Scalar(2.0)),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec3([2.0, 4.0, 6.0]));
}

#[cfg(feature = "3d")]
#[test]
fn test_vec3_neg() {
    let result = eval_linalg("-v", &[("v", Value::Vec3([1.0, -2.0, 3.0]))]).unwrap();
    assert_eq!(result, Value::Vec3([-1.0, 2.0, -3.0]));
}

// ============================================================================
// Matrix operations
// ============================================================================

#[test]
fn test_mat2_vec2_mul() {
    // Identity matrix times vector
    // [1 0]   [3]   [3]
    // [0 1] * [4] = [4]
    // Column-major: [1, 0, 0, 1]
    let result = eval_linalg(
        "m * v",
        &[
            ("m", Value::Mat2([1.0, 0.0, 0.0, 1.0])),
            ("v", Value::Vec2([3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([3.0, 4.0]));
}

#[test]
fn test_mat2_vec2_mul_rotation() {
    // 90 degree rotation
    // [0 -1]   [1]   [0]
    // [1  0] * [0] = [1]
    // Column-major: [0, 1, -1, 0]
    let result = eval_linalg(
        "m * v",
        &[
            ("m", Value::Mat2([0.0, 1.0, -1.0, 0.0])),
            ("v", Value::Vec2([1.0, 0.0])),
        ],
    )
    .unwrap();
    if let Value::Vec2(v) = result {
        assert_close(v[0], 0.0, "mat2 rotation x");
        assert_close(v[1], 1.0, "mat2 rotation y");
    } else {
        panic!("expected Vec2");
    }
}

#[test]
fn test_vec2_mat2_mul() {
    // Row vector times identity
    let result = eval_linalg(
        "v * m",
        &[
            ("v", Value::Vec2([3.0, 4.0])),
            ("m", Value::Mat2([1.0, 0.0, 0.0, 1.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([3.0, 4.0]));
}

#[test]
fn test_mat2_mul_mat2() {
    // Identity times identity
    let result = eval_linalg(
        "a * b",
        &[
            ("a", Value::Mat2([1.0, 0.0, 0.0, 1.0])),
            ("b", Value::Mat2([1.0, 0.0, 0.0, 1.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Mat2([1.0, 0.0, 0.0, 1.0]));
}

#[cfg(feature = "3d")]
#[test]
fn test_mat3_vec3_mul() {
    // Identity
    let result = eval_linalg(
        "m * v",
        &[
            (
                "m",
                Value::Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            ),
            ("v", Value::Vec3([3.0, 4.0, 5.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec3([3.0, 4.0, 5.0]));
}

// ============================================================================
// Linalg functions - dot
// ============================================================================

#[test]
fn test_dot_vec2() {
    let result = eval_linalg(
        "dot(a, b)",
        &[
            ("a", Value::Vec2([1.0, 2.0])),
            ("b", Value::Vec2([3.0, 4.0])),
        ],
    )
    .unwrap();
    // 1*3 + 2*4 = 11
    assert_eq!(result, Value::Scalar(11.0));
}

#[test]
fn test_dot_vec2_orthogonal() {
    let result = eval_linalg(
        "dot(a, b)",
        &[
            ("a", Value::Vec2([1.0, 0.0])),
            ("b", Value::Vec2([0.0, 1.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(0.0));
}

#[cfg(feature = "3d")]
#[test]
fn test_dot_vec3() {
    let result = eval_linalg(
        "dot(a, b)",
        &[
            ("a", Value::Vec3([1.0, 2.0, 3.0])),
            ("b", Value::Vec3([4.0, 5.0, 6.0])),
        ],
    )
    .unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(result, Value::Scalar(32.0));
}

// ============================================================================
// Linalg functions - cross (3d only)
// ============================================================================

#[cfg(feature = "3d")]
#[test]
fn test_cross_basic() {
    let result = eval_linalg(
        "cross(a, b)",
        &[
            ("a", Value::Vec3([1.0, 0.0, 0.0])),
            ("b", Value::Vec3([0.0, 1.0, 0.0])),
        ],
    )
    .unwrap();
    // x cross y = z
    if let Value::Vec3(v) = result {
        assert_vec3_close(&v, &[0.0, 0.0, 1.0], "cross x*y");
    } else {
        panic!("expected Vec3");
    }
}

#[cfg(feature = "3d")]
#[test]
fn test_cross_anticommutative() {
    let a = Value::Vec3([1.0, 2.0, 3.0]);
    let b = Value::Vec3([4.0, 5.0, 6.0]);

    let result1 = eval_linalg("cross(a, b)", &[("a", a.clone()), ("b", b.clone())]).unwrap();
    let result2 = eval_linalg("cross(b, a)", &[("a", a), ("b", b)]).unwrap();

    if let (Value::Vec3(v1), Value::Vec3(v2)) = (result1, result2) {
        // cross(a, b) = -cross(b, a)
        assert_vec3_close(&v1, &[-v2[0], -v2[1], -v2[2]], "cross anticommutative");
    } else {
        panic!("expected Vec3");
    }
}

// ============================================================================
// Linalg functions - length
// ============================================================================

#[test]
fn test_length_vec2() {
    let result = eval_linalg("length(v)", &[("v", Value::Vec2([3.0, 4.0]))]).unwrap();
    // sqrt(9 + 16) = 5
    assert_eq!(result, Value::Scalar(5.0));
}

#[test]
fn test_length_vec2_unit() {
    let result = eval_linalg("length(v)", &[("v", Value::Vec2([1.0, 0.0]))]).unwrap();
    assert_eq!(result, Value::Scalar(1.0));
}

#[cfg(feature = "3d")]
#[test]
fn test_length_vec3() {
    let result = eval_linalg("length(v)", &[("v", Value::Vec3([2.0, 3.0, 6.0]))]).unwrap();
    // sqrt(4 + 9 + 36) = 7
    assert_eq!(result, Value::Scalar(7.0));
}

// ============================================================================
// Linalg functions - normalize
// ============================================================================

#[test]
fn test_normalize_vec2() {
    let result = eval_linalg("normalize(v)", &[("v", Value::Vec2([3.0, 4.0]))]).unwrap();
    if let Value::Vec2(v) = result {
        assert_vec2_close(&v, &[0.6, 0.8], "normalize");
        // Verify unit length
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert_close(len, 1.0, "normalized length");
    } else {
        panic!("expected Vec2");
    }
}

#[cfg(feature = "3d")]
#[test]
fn test_normalize_vec3() {
    let result = eval_linalg("normalize(v)", &[("v", Value::Vec3([2.0, 3.0, 6.0]))]).unwrap();
    if let Value::Vec3(v) = result {
        let expected = [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0];
        assert_vec3_close(&v, &expected, "normalize vec3");
    } else {
        panic!("expected Vec3");
    }
}

// ============================================================================
// Linalg functions - distance
// ============================================================================

#[test]
fn test_distance_vec2() {
    let result = eval_linalg(
        "distance(a, b)",
        &[
            ("a", Value::Vec2([0.0, 0.0])),
            ("b", Value::Vec2([3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(5.0));
}

#[test]
fn test_distance_vec2_same_point() {
    let result = eval_linalg(
        "distance(a, b)",
        &[
            ("a", Value::Vec2([5.0, 7.0])),
            ("b", Value::Vec2([5.0, 7.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(0.0));
}

#[cfg(feature = "3d")]
#[test]
fn test_distance_vec3() {
    let result = eval_linalg(
        "distance(a, b)",
        &[
            ("a", Value::Vec3([0.0, 0.0, 0.0])),
            ("b", Value::Vec3([2.0, 3.0, 6.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(7.0));
}

// ============================================================================
// Linalg functions - reflect
// ============================================================================

#[test]
fn test_reflect_vec2() {
    // Reflect horizontal vector off vertical normal
    let result = eval_linalg(
        "reflect(i, n)",
        &[
            ("i", Value::Vec2([1.0, 0.0])),
            ("n", Value::Vec2([-1.0, 0.0])), // pointing left
        ],
    )
    .unwrap();
    if let Value::Vec2(v) = result {
        assert_vec2_close(&v, &[-1.0, 0.0], "reflect horizontal");
    } else {
        panic!("expected Vec2");
    }
}

#[test]
fn test_reflect_vec2_45deg() {
    // Reflect down-right vector off horizontal surface (normal pointing up)
    let result = eval_linalg(
        "reflect(i, n)",
        &[
            ("i", Value::Vec2([1.0, -1.0])),
            ("n", Value::Vec2([0.0, 1.0])),
        ],
    )
    .unwrap();
    if let Value::Vec2(v) = result {
        // Should bounce to up-right
        assert_vec2_close(&v, &[1.0, 1.0], "reflect 45deg");
    } else {
        panic!("expected Vec2");
    }
}

// ============================================================================
// Linalg functions - hadamard
// ============================================================================

#[test]
fn test_hadamard_vec2() {
    let result = eval_linalg(
        "hadamard(a, b)",
        &[
            ("a", Value::Vec2([2.0, 3.0])),
            ("b", Value::Vec2([4.0, 5.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([8.0, 15.0]));
}

#[cfg(feature = "3d")]
#[test]
fn test_hadamard_vec3() {
    let result = eval_linalg(
        "hadamard(a, b)",
        &[
            ("a", Value::Vec3([1.0, 2.0, 3.0])),
            ("b", Value::Vec3([4.0, 5.0, 6.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec3([4.0, 10.0, 18.0]));
}

// ============================================================================
// Linalg functions - lerp / mix (vector types only, scalar lerp is in sap-std)
// ============================================================================

#[test]
fn test_lerp_vec2() {
    let result = eval_linalg(
        "lerp(a, b, t)",
        &[
            ("a", Value::Vec2([0.0, 0.0])),
            ("b", Value::Vec2([10.0, 20.0])),
            ("t", Value::Scalar(0.5)),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([5.0, 10.0]));
}

#[test]
fn test_lerp_vec2_endpoints() {
    let a = Value::Vec2([1.0, 2.0]);
    let b = Value::Vec2([5.0, 6.0]);

    let result0 = eval_linalg(
        "lerp(a, b, t)",
        &[
            ("a", a.clone()),
            ("b", b.clone()),
            ("t", Value::Scalar(0.0)),
        ],
    )
    .unwrap();
    assert_eq!(result0, Value::Vec2([1.0, 2.0]));

    let result1 = eval_linalg(
        "lerp(a, b, t)",
        &[("a", a), ("b", b), ("t", Value::Scalar(1.0))],
    )
    .unwrap();
    assert_eq!(result1, Value::Vec2([5.0, 6.0]));
}

#[test]
fn test_mix_vec2() {
    let result = eval_linalg(
        "mix(a, b, t)",
        &[
            ("a", Value::Vec2([0.0, 0.0])),
            ("b", Value::Vec2([10.0, 20.0])),
            ("t", Value::Scalar(0.25)),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Vec2([2.5, 5.0]));
}

// ============================================================================
// Complex expressions
// ============================================================================

#[test]
fn test_complex_normalize_scale() {
    let result = eval_linalg("normalize(v) * 5", &[("v", Value::Vec2([3.0, 4.0]))]).unwrap();
    if let Value::Vec2(v) = result {
        assert_vec2_close(&v, &[3.0, 4.0], "normalize then scale");
    } else {
        panic!("expected Vec2");
    }
}

#[test]
fn test_complex_length_sub() {
    // Distance is length of difference
    let result = eval_linalg(
        "length(a - b)",
        &[
            ("a", Value::Vec2([0.0, 0.0])),
            ("b", Value::Vec2([3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(result, Value::Scalar(5.0));
}

#[test]
fn test_complex_dot_self() {
    // dot(v, v) = length(v)^2
    let v = Value::Vec2([3.0, 4.0]);
    let result = eval_linalg("dot(v, v)", &[("v", v)]).unwrap();
    assert_eq!(result, Value::Scalar(25.0)); // 3^2 + 4^2
}

// ============================================================================
// Type error tests
// ============================================================================

#[test]
fn test_type_mismatch_add() {
    let result = eval_linalg(
        "a + b",
        &[("a", Value::Scalar(1.0)), ("b", Value::Vec2([1.0, 2.0]))],
    );
    assert!(result.is_err());
}

#[test]
fn test_type_mismatch_vec_sizes() {
    #[cfg(feature = "3d")]
    {
        let result = eval_linalg(
            "a + b",
            &[
                ("a", Value::Vec2([1.0, 2.0])),
                ("b", Value::Vec3([1.0, 2.0, 3.0])),
            ],
        );
        assert!(result.is_err());
    }
}

// ============================================================================
// Lua backend tests
// ============================================================================

#[cfg(feature = "lua")]
mod lua_tests {
    use super::*;
    use crate::lua::{LuaExpr, emit_lua};

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<LuaExpr, crate::lua::LuaError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_lua(expr.ast(), &types)
    }

    #[test]
    fn test_lua_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("[1]"));
        assert!(result.code.contains("[2]"));
    }

    #[test]
    fn test_lua_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("math.sqrt"));
    }

    #[test]
    fn test_lua_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("math.sqrt"));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_lua_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
    }

    #[test]
    fn test_lua_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_lua_mat_vec_mul() {
        let result = emit("m * v", &[("m", Type::Mat2), ("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }
}

// ============================================================================
// WGSL backend tests
// ============================================================================

#[cfg(feature = "wgsl")]
mod wgsl_tests {
    use super::*;
    use crate::wgsl::{WgslExpr, emit_wgsl};

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<WgslExpr, crate::wgsl::WgslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_wgsl(expr.ast(), &types)
    }

    #[test]
    fn test_wgsl_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("dot("));
    }

    #[test]
    fn test_wgsl_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("length("));
    }

    #[test]
    fn test_wgsl_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("normalize("));
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_wgsl_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains("cross("));
    }

    #[test]
    fn test_wgsl_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("mix(")); // WGSL uses mix
    }

    #[test]
    fn test_wgsl_mat_vec_mul() {
        let result = emit("m * v", &[("m", Type::Mat2), ("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }
}

// ============================================================================
// Cranelift backend tests (scalar return only)
// ============================================================================

#[cfg(feature = "cranelift")]
mod cranelift_tests {
    use super::*;
    use crate::cranelift::{LinalgJit, VarSpec};

    fn compile_and_run(expr_str: &str, vars: &[VarSpec], args: &[f32]) -> f32 {
        let expr = Expr::parse(expr_str).unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit.compile_scalar(expr.ast(), vars).unwrap();
        func.call(args)
    }

    #[test]
    fn test_cranelift_scalar_ops() {
        let result = compile_and_run(
            "a + b",
            &[
                VarSpec::new("a", Type::Scalar),
                VarSpec::new("b", Type::Scalar),
            ],
            &[3.0, 4.0],
        );
        assert_close(result, 7.0, "cranelift scalar add");
    }

    #[test]
    fn test_cranelift_dot_vec2() {
        let result = compile_and_run(
            "dot(a, b)",
            &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            &[1.0, 2.0, 3.0, 4.0],
        );
        // 1*3 + 2*4 = 11
        assert_close(result, 11.0, "cranelift dot vec2");
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cranelift_dot_vec3() {
        let result = compile_and_run(
            "dot(a, b)",
            &[VarSpec::new("a", Type::Vec3), VarSpec::new("b", Type::Vec3)],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        // 1*4 + 2*5 + 3*6 = 32
        assert_close(result, 32.0, "cranelift dot vec3");
    }

    #[test]
    fn test_cranelift_length_vec2() {
        let result = compile_and_run("length(v)", &[VarSpec::new("v", Type::Vec2)], &[3.0, 4.0]);
        assert_close(result, 5.0, "cranelift length vec2");
    }

    #[test]
    fn test_cranelift_distance_vec2() {
        let result = compile_and_run(
            "distance(a, b)",
            &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            &[0.0, 0.0, 3.0, 4.0],
        );
        assert_close(result, 5.0, "cranelift distance vec2");
    }

    #[test]
    fn test_cranelift_complex() {
        // length(a - b) should equal distance(a, b)
        let result = compile_and_run(
            "length(a - b)",
            &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            &[0.0, 0.0, 3.0, 4.0],
        );
        assert_close(result, 5.0, "cranelift length of difference");
    }

    #[test]
    fn test_cranelift_vec_scale_length() {
        let result = compile_and_run(
            "length(v * 2)",
            &[VarSpec::new("v", Type::Vec2)],
            &[3.0, 4.0],
        );
        assert_close(result, 10.0, "cranelift scaled vec length");
    }
}
