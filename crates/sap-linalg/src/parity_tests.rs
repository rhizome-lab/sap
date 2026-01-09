//! Parity tests to ensure eval and Cranelift backends produce consistent results.
//!
//! For linalg, we can compare eval vs Cranelift for scalar-returning functions
//! (dot, length, distance). WGSL/Lua generate code strings and don't have
//! built-in runtime execution for vector types.

use crate::{Type, Value, eval, linalg_registry};
use rhizome_sap_core::Expr;
use std::collections::HashMap;

#[cfg(feature = "cranelift")]
use crate::cranelift::{LinalgJit, VarSpec};

const EPSILON: f32 = 0.0001;

fn assert_close(a: f32, b: f32, context: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < EPSILON || (a.abs() > 1.0 && diff / a.abs() < EPSILON),
        "{}: values differ by {}: {} vs {}",
        context,
        diff,
        a,
        b
    );
}

// ============================================================================
// Eval helpers
// ============================================================================

fn eval_scalar(expr_str: &str, vars: &[(&str, Value<f32>)]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let var_map: HashMap<String, Value<f32>> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let registry = linalg_registry();
    let result = eval(expr.ast(), &var_map, &registry).unwrap();
    match result {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar result"),
    }
}

// ============================================================================
// Cranelift helpers
// ============================================================================

#[cfg(feature = "cranelift")]
fn cranelift_scalar(expr_str: &str, vars: &[VarSpec], args: &[f32]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let jit = LinalgJit::new().unwrap();
    let func = jit.compile_scalar(expr.ast(), vars).unwrap();
    func.call(args)
}

// ============================================================================
// Parity check functions
// ============================================================================

#[cfg(feature = "cranelift")]
fn check_vec2_parity(
    expr_str: &str,
    v1_name: &str,
    v1: [f32; 2],
    v2_name: Option<&str>,
    v2: Option<[f32; 2]>,
) {
    // Build eval variables
    let mut eval_vars: Vec<(&str, Value<f32>)> = vec![(v1_name, Value::Vec2(v1))];
    if let (Some(name), Some(val)) = (v2_name, v2) {
        eval_vars.push((name, Value::Vec2(val)));
    }

    let eval_result = eval_scalar(expr_str, &eval_vars);

    // Build Cranelift variables and args
    let mut cranelift_vars = vec![VarSpec::new(v1_name, Type::Vec2)];
    let mut cranelift_args: Vec<f32> = v1.to_vec();
    if let (Some(name), Some(val)) = (v2_name, v2) {
        cranelift_vars.push(VarSpec::new(name, Type::Vec2));
        cranelift_args.extend(val);
    }

    let cranelift_result = cranelift_scalar(expr_str, &cranelift_vars, &cranelift_args);

    let context = format!(
        "expr='{}', {}={:?}{}",
        expr_str,
        v1_name,
        v1,
        v2_name
            .map(|n| format!(", {}={:?}", n, v2.unwrap()))
            .unwrap_or_default()
    );
    assert_close(eval_result, cranelift_result, &context);
}

#[cfg(all(feature = "cranelift", feature = "3d"))]
fn check_vec3_parity(
    expr_str: &str,
    v1_name: &str,
    v1: [f32; 3],
    v2_name: Option<&str>,
    v2: Option<[f32; 3]>,
) {
    let mut eval_vars: Vec<(&str, Value<f32>)> = vec![(v1_name, Value::Vec3(v1))];
    if let (Some(name), Some(val)) = (v2_name, v2) {
        eval_vars.push((name, Value::Vec3(val)));
    }

    let eval_result = eval_scalar(expr_str, &eval_vars);

    let mut cranelift_vars = vec![VarSpec::new(v1_name, Type::Vec3)];
    let mut cranelift_args: Vec<f32> = v1.to_vec();
    if let (Some(name), Some(val)) = (v2_name, v2) {
        cranelift_vars.push(VarSpec::new(name, Type::Vec3));
        cranelift_args.extend(val);
    }

    let cranelift_result = cranelift_scalar(expr_str, &cranelift_vars, &cranelift_args);

    let context = format!(
        "expr='{}', {}={:?}{}",
        expr_str,
        v1_name,
        v1,
        v2_name
            .map(|n| format!(", {}={:?}", n, v2.unwrap()))
            .unwrap_or_default()
    );
    assert_close(eval_result, cranelift_result, &context);
}

// ============================================================================
// Parity tests - dot product
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_dot_vec2() {
    check_vec2_parity("dot(a, b)", "a", [1.0, 2.0], Some("b"), Some([3.0, 4.0]));
    check_vec2_parity("dot(a, b)", "a", [0.0, 1.0], Some("b"), Some([1.0, 0.0]));
    check_vec2_parity("dot(a, b)", "a", [1.0, 0.0], Some("b"), Some([1.0, 0.0]));
    check_vec2_parity("dot(a, b)", "a", [-1.0, 2.0], Some("b"), Some([3.0, -4.0]));
}

#[cfg(all(feature = "cranelift", feature = "3d"))]
#[test]
fn test_parity_dot_vec3() {
    check_vec3_parity(
        "dot(a, b)",
        "a",
        [1.0, 2.0, 3.0],
        Some("b"),
        Some([4.0, 5.0, 6.0]),
    );
    check_vec3_parity(
        "dot(a, b)",
        "a",
        [1.0, 0.0, 0.0],
        Some("b"),
        Some([0.0, 1.0, 0.0]),
    );
}

// ============================================================================
// Parity tests - length
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_length_vec2() {
    check_vec2_parity("length(v)", "v", [3.0, 4.0], None, None);
    check_vec2_parity("length(v)", "v", [1.0, 0.0], None, None);
    check_vec2_parity("length(v)", "v", [0.0, 1.0], None, None);
    check_vec2_parity("length(v)", "v", [1.0, 1.0], None, None);
    check_vec2_parity("length(v)", "v", [5.0, 12.0], None, None);
}

#[cfg(all(feature = "cranelift", feature = "3d"))]
#[test]
fn test_parity_length_vec3() {
    check_vec3_parity("length(v)", "v", [2.0, 3.0, 6.0], None, None);
    check_vec3_parity("length(v)", "v", [1.0, 0.0, 0.0], None, None);
    check_vec3_parity("length(v)", "v", [1.0, 1.0, 1.0], None, None);
}

// ============================================================================
// Parity tests - distance
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_distance_vec2() {
    check_vec2_parity(
        "distance(a, b)",
        "a",
        [0.0, 0.0],
        Some("b"),
        Some([3.0, 4.0]),
    );
    check_vec2_parity(
        "distance(a, b)",
        "a",
        [1.0, 1.0],
        Some("b"),
        Some([4.0, 5.0]),
    );
    check_vec2_parity(
        "distance(a, b)",
        "a",
        [5.0, 5.0],
        Some("b"),
        Some([5.0, 5.0]),
    );
}

#[cfg(all(feature = "cranelift", feature = "3d"))]
#[test]
fn test_parity_distance_vec3() {
    check_vec3_parity(
        "distance(a, b)",
        "a",
        [0.0, 0.0, 0.0],
        Some("b"),
        Some([2.0, 3.0, 6.0]),
    );
}

// ============================================================================
// Parity tests - complex expressions
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_length_difference() {
    // length(a - b) should equal distance(a, b)
    check_vec2_parity(
        "length(a - b)",
        "a",
        [0.0, 0.0],
        Some("b"),
        Some([3.0, 4.0]),
    );
    check_vec2_parity(
        "length(a - b)",
        "a",
        [10.0, 10.0],
        Some("b"),
        Some([13.0, 14.0]),
    );
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_dot_self() {
    // dot(v, v) should equal length(v)^2
    let v = [3.0, 4.0];
    let expr = Expr::parse("dot(v, v)").unwrap();

    // Eval
    let var_map: HashMap<String, Value<f32>> =
        [("v".to_string(), Value::Vec2(v))].into_iter().collect();
    let registry = linalg_registry();
    let eval_result = match eval(expr.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    // Cranelift
    let jit = LinalgJit::new().unwrap();
    let func = jit
        .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
        .unwrap();
    let cranelift_result = func.call(&v);

    // Both should be 25 (3^2 + 4^2)
    assert_close(eval_result, cranelift_result, "dot(v, v)");
    assert_close(eval_result, 25.0, "dot(v, v) expected value");
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_scaled_vector_length() {
    // length(v * s) should equal length(v) * s (for positive s)
    check_vec2_parity("length(v * 2)", "v", [3.0, 4.0], None, None);
    check_vec2_parity("length(v * 0.5)", "v", [6.0, 8.0], None, None);
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_vector_operations() {
    // length(a + b) - various vector operations
    check_vec2_parity(
        "length(a + b)",
        "a",
        [1.0, 0.0],
        Some("b"),
        Some([0.0, 1.0]),
    );
    check_vec2_parity(
        "dot(a + b, a - b)",
        "a",
        [2.0, 3.0],
        Some("b"),
        Some([1.0, 1.0]),
    );
}

// ============================================================================
// Parity tests - scalar operations (sanity check)
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_scalar_ops() {
    let expr = Expr::parse("a + b * 2").unwrap();

    // Eval
    let var_map: HashMap<String, Value<f32>> = [
        ("a".to_string(), Value::Scalar(3.0)),
        ("b".to_string(), Value::Scalar(4.0)),
    ]
    .into_iter()
    .collect();
    let registry = linalg_registry();
    let eval_result = match eval(expr.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    // Cranelift
    let jit = LinalgJit::new().unwrap();
    let func = jit
        .compile_scalar(
            expr.ast(),
            &[
                VarSpec::new("a", Type::Scalar),
                VarSpec::new("b", Type::Scalar),
            ],
        )
        .unwrap();
    let cranelift_result = func.call(&[3.0, 4.0]);

    assert_close(eval_result, cranelift_result, "scalar ops");
    assert_close(eval_result, 11.0, "scalar ops expected value");
}

// ============================================================================
// Randomized parity tests
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_vec2_dot() {
    // Test with various "random" values
    let test_vectors: &[([f32; 2], [f32; 2])] = &[
        ([0.5, 0.7], [0.3, 0.9]),
        ([1.23, 4.56], [7.89, 0.12]),
        ([-3.0, 2.5], [1.5, -4.0]),
        ([100.0, 200.0], [0.001, 0.002]),
    ];

    for (v1, v2) in test_vectors {
        check_vec2_parity("dot(a, b)", "a", *v1, Some("b"), Some(*v2));
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_vec2_length() {
    let test_vectors: &[[f32; 2]] = &[[0.5, 0.7], [1.23, 4.56], [100.0, 0.0], [0.001, 0.001]];

    for v in test_vectors {
        check_vec2_parity("length(v)", "v", *v, None, None);
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_vec2_distance() {
    let test_pairs: &[([f32; 2], [f32; 2])] = &[
        ([0.0, 0.0], [1.0, 1.0]),
        ([5.5, 3.3], [7.7, 9.9]),
        ([-10.0, -10.0], [10.0, 10.0]),
    ];

    for (v1, v2) in test_pairs {
        check_vec2_parity("distance(a, b)", "a", *v1, Some("b"), Some(*v2));
    }
}
