//! Parity tests to ensure eval and Cranelift backends produce consistent results.

#[cfg(feature = "cranelift")]
use crate::{Type, Value, eval, quaternion_registry};
#[cfg(feature = "cranelift")]
use rhizome_dew_core::Expr;
#[cfg(feature = "cranelift")]
use std::collections::HashMap;

#[cfg(feature = "cranelift")]
use crate::cranelift::{QuaternionJit, VarSpec};

#[cfg(feature = "cranelift")]
const EPSILON: f32 = 0.0001;

#[cfg(feature = "cranelift")]
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

#[cfg(feature = "cranelift")]
fn eval_scalar(expr_str: &str, vars: &[(&str, Value<f32>)]) -> f32 {
    let expr = Expr::parse(expr_str).unwrap();
    let var_map: HashMap<String, Value<f32>> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let registry = quaternion_registry();
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
    let jit = QuaternionJit::new().unwrap();
    let func = jit.compile_scalar(expr.ast(), vars).unwrap();
    func.call(args)
}

// ============================================================================
// Parity check functions
// ============================================================================

#[cfg(feature = "cranelift")]
fn check_quaternion_parity(
    expr_str: &str,
    q1_name: &str,
    q1: [f32; 4],
    q2_name: Option<&str>,
    q2: Option<[f32; 4]>,
) {
    // Build eval variables
    let mut eval_vars: Vec<(&str, Value<f32>)> = vec![(q1_name, Value::Quaternion(q1))];
    if let (Some(name), Some(val)) = (q2_name, q2) {
        eval_vars.push((name, Value::Quaternion(val)));
    }

    let eval_result = eval_scalar(expr_str, &eval_vars);

    // Build Cranelift variables and args
    let mut cranelift_vars = vec![VarSpec::new(q1_name, Type::Quaternion)];
    let mut cranelift_args: Vec<f32> = q1.to_vec();
    if let (Some(name), Some(val)) = (q2_name, q2) {
        cranelift_vars.push(VarSpec::new(name, Type::Quaternion));
        cranelift_args.extend(val);
    }

    let cranelift_result = cranelift_scalar(expr_str, &cranelift_vars, &cranelift_args);

    let context = format!(
        "expr='{}', {}={:?}{}",
        expr_str,
        q1_name,
        q1,
        q2_name
            .map(|n| format!(", {}={:?}", n, q2.unwrap()))
            .unwrap_or_default()
    );
    assert_close(eval_result, cranelift_result, &context);
}

#[cfg(feature = "cranelift")]
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
// Parity tests - quaternion length
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_quaternion_length() {
    // Identity quaternion
    check_quaternion_parity("length(q)", "q", [0.0, 0.0, 0.0, 1.0], None, None);
    // Various quaternions
    check_quaternion_parity("length(q)", "q", [1.0, 0.0, 0.0, 0.0], None, None);
    check_quaternion_parity("length(q)", "q", [1.0, 2.0, 2.0, 0.0], None, None); // |q| = 3
    check_quaternion_parity("length(q)", "q", [0.5, 0.5, 0.5, 0.5], None, None); // |q| = 1
}

// ============================================================================
// Parity tests - vec3 length
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_vec3_length() {
    check_vec3_parity("length(v)", "v", [3.0, 4.0, 0.0], None, None); // 5
    check_vec3_parity("length(v)", "v", [1.0, 0.0, 0.0], None, None); // 1
    check_vec3_parity("length(v)", "v", [2.0, 3.0, 6.0], None, None); // 7
    check_vec3_parity("length(v)", "v", [1.0, 1.0, 1.0], None, None); // sqrt(3)
}

// ============================================================================
// Parity tests - dot product
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_quaternion_dot() {
    check_quaternion_parity(
        "dot(a, b)",
        "a",
        [0.0, 0.0, 0.0, 1.0],
        Some("b"),
        Some([0.0, 0.0, 0.0, 1.0]),
    ); // 1
    check_quaternion_parity(
        "dot(a, b)",
        "a",
        [1.0, 0.0, 0.0, 0.0],
        Some("b"),
        Some([0.0, 1.0, 0.0, 0.0]),
    ); // 0
    check_quaternion_parity(
        "dot(a, b)",
        "a",
        [1.0, 2.0, 3.0, 4.0],
        Some("b"),
        Some([4.0, 3.0, 2.0, 1.0]),
    ); // 20
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_vec3_dot() {
    check_vec3_parity(
        "dot(a, b)",
        "a",
        [1.0, 2.0, 3.0],
        Some("b"),
        Some([4.0, 5.0, 6.0]),
    ); // 32
    check_vec3_parity(
        "dot(a, b)",
        "a",
        [1.0, 0.0, 0.0],
        Some("b"),
        Some([0.0, 1.0, 0.0]),
    ); // 0 (perpendicular)
}

// ============================================================================
// Parity tests - quaternion multiplication (via length)
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_quaternion_mul_length() {
    // |q1 * q2| = |q1| * |q2| for unit quaternions
    // For non-unit: verify eval and cranelift agree
    let expr_str = "length(a * b)";

    // Identity * identity = identity (length 1)
    check_quaternion_parity(
        expr_str,
        "a",
        [0.0, 0.0, 0.0, 1.0],
        Some("b"),
        Some([0.0, 0.0, 0.0, 1.0]),
    );

    // q * identity = q
    check_quaternion_parity(
        expr_str,
        "a",
        [1.0, 2.0, 2.0, 0.0],
        Some("b"),
        Some([0.0, 0.0, 0.0, 1.0]),
    );

    // Two rotations
    let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
    check_quaternion_parity(
        expr_str,
        "a",
        [sqrt2_2, 0.0, 0.0, sqrt2_2], // 90° around X
        Some("b"),
        Some([0.0, sqrt2_2, 0.0, sqrt2_2]), // 90° around Y
    );
}

// ============================================================================
// Parity tests - mathematical identities
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_unit_quaternion_length() {
    // Normalized quaternion should have length 1
    let q = [0.5, 0.5, 0.5, 0.5]; // Already unit

    let expr = Expr::parse("length(q)").unwrap();
    let var_map: HashMap<String, Value<f32>> = [("q".to_string(), Value::Quaternion(q))]
        .into_iter()
        .collect();
    let registry = quaternion_registry();

    let eval_result = match eval(expr.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    let jit = QuaternionJit::new().unwrap();
    let func = jit
        .compile_scalar(expr.ast(), &[VarSpec::new("q", Type::Quaternion)])
        .unwrap();
    let cranelift_result = func.call(&q);

    assert_close(eval_result, cranelift_result, "unit quaternion length");
    assert_close(eval_result, 1.0, "unit quaternion length expected");
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_dot_self_equals_length_squared() {
    // dot(q, q) = |q|²
    let q = [1.0, 2.0, 2.0, 0.0]; // |q| = 3

    let expr1 = Expr::parse("dot(q, q)").unwrap();
    let var_map: HashMap<String, Value<f32>> = [("q".to_string(), Value::Quaternion(q))]
        .into_iter()
        .collect();
    let registry = quaternion_registry();

    let eval_dot = match eval(expr1.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    let jit = QuaternionJit::new().unwrap();
    let func = jit
        .compile_scalar(expr1.ast(), &[VarSpec::new("q", Type::Quaternion)])
        .unwrap();
    let cranelift_dot = func.call(&q);

    assert_close(eval_dot, cranelift_dot, "dot(q, q)");
    assert_close(eval_dot, 9.0, "dot(q, q) expected (|q|² = 9)");
}

// ============================================================================
// Randomized parity tests
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_quaternion_length() {
    let test_values: &[[f32; 4]] = &[
        [0.1, 0.2, 0.3, 0.4],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-0.5, 0.5, -0.5, 0.5],
    ];

    for q in test_values {
        check_quaternion_parity("length(q)", "q", *q, None, None);
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_vec3_length() {
    let test_values: &[[f32; 3]] = &[
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
        [5.0, 12.0, 0.0],
        [-3.0, -4.0, 0.0],
        [1.0, 1.0, 1.0],
    ];

    for v in test_values {
        check_vec3_parity("length(v)", "v", *v, None, None);
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_quaternion_dot() {
    let test_pairs: &[([f32; 4], [f32; 4])] = &[
        ([0.0, 0.0, 0.0, 1.0], [0.707, 0.0, 0.0, 0.707]),
        ([0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5]),
        ([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]),
    ];

    for (q1, q2) in test_pairs {
        check_quaternion_parity("dot(a, b)", "a", *q1, Some("b"), Some(*q2));
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_vec3_dot() {
    let test_pairs: &[([f32; 3], [f32; 3])] = &[
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        ([-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]),
    ];

    for (v1, v2) in test_pairs {
        check_vec3_parity("dot(a, b)", "a", *v1, Some("b"), Some(*v2));
    }
}
