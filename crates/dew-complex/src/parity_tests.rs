//! Parity tests to ensure eval and Cranelift backends produce consistent results.

#[cfg(feature = "cranelift")]
use crate::{Type, Value, complex_registry, eval};
#[cfg(feature = "cranelift")]
use rhizome_dew_core::Expr;
#[cfg(feature = "cranelift")]
use std::collections::HashMap;

#[cfg(feature = "cranelift")]
use crate::cranelift::{ComplexJit, VarSpec};

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
    let registry = complex_registry();
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
    let jit = ComplexJit::new().unwrap();
    let func = jit.compile_scalar(expr.ast(), vars).unwrap();
    func.call(args)
}

// ============================================================================
// Parity check functions
// ============================================================================

#[cfg(feature = "cranelift")]
fn check_complex_parity(
    expr_str: &str,
    z1_name: &str,
    z1: [f32; 2],
    z2_name: Option<&str>,
    z2: Option<[f32; 2]>,
) {
    // Build eval variables
    let mut eval_vars: Vec<(&str, Value<f32>)> = vec![(z1_name, Value::Complex(z1))];
    if let (Some(name), Some(val)) = (z2_name, z2) {
        eval_vars.push((name, Value::Complex(val)));
    }

    let eval_result = eval_scalar(expr_str, &eval_vars);

    // Build Cranelift variables and args
    let mut cranelift_vars = vec![VarSpec::new(z1_name, Type::Complex)];
    let mut cranelift_args: Vec<f32> = z1.to_vec();
    if let (Some(name), Some(val)) = (z2_name, z2) {
        cranelift_vars.push(VarSpec::new(name, Type::Complex));
        cranelift_args.extend(val);
    }

    let cranelift_result = cranelift_scalar(expr_str, &cranelift_vars, &cranelift_args);

    let context = format!(
        "expr='{}', {}={:?}{}",
        expr_str,
        z1_name,
        z1,
        z2_name
            .map(|n| format!(", {}={:?}", n, z2.unwrap()))
            .unwrap_or_default()
    );
    assert_close(eval_result, cranelift_result, &context);
}

// ============================================================================
// Parity tests - abs (magnitude)
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_abs() {
    check_complex_parity("abs(z)", "z", [3.0, 4.0], None, None);
    check_complex_parity("abs(z)", "z", [1.0, 0.0], None, None);
    check_complex_parity("abs(z)", "z", [0.0, 1.0], None, None);
    check_complex_parity("abs(z)", "z", [1.0, 1.0], None, None);
    check_complex_parity("abs(z)", "z", [5.0, 12.0], None, None);
}

// ============================================================================
// Parity tests - arg (phase)
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_arg() {
    check_complex_parity("arg(z)", "z", [1.0, 0.0], None, None); // 0
    check_complex_parity("arg(z)", "z", [0.0, 1.0], None, None); // pi/2
    check_complex_parity("arg(z)", "z", [1.0, 1.0], None, None); // pi/4
    check_complex_parity("arg(z)", "z", [-1.0, 0.0], None, None); // pi
    check_complex_parity("arg(z)", "z", [1.0, -1.0], None, None); // -pi/4
}

// ============================================================================
// Parity tests - norm (squared magnitude)
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_norm() {
    check_complex_parity("norm(z)", "z", [3.0, 4.0], None, None); // 25
    check_complex_parity("norm(z)", "z", [1.0, 0.0], None, None); // 1
    check_complex_parity("norm(z)", "z", [2.0, 3.0], None, None); // 13
}

// ============================================================================
// Parity tests - re/im
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_re_im() {
    check_complex_parity("re(z)", "z", [3.0, 4.0], None, None);
    check_complex_parity("im(z)", "z", [3.0, 4.0], None, None);
    check_complex_parity("re(z)", "z", [-5.0, 7.0], None, None);
    check_complex_parity("im(z)", "z", [-5.0, 7.0], None, None);
}

// ============================================================================
// Parity tests - complex arithmetic returning scalar
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_complex_mul_abs() {
    // |z1 * z2| = |z1| * |z2|
    check_complex_parity("abs(a * b)", "a", [1.0, 2.0], Some("b"), Some([3.0, 4.0]));
    check_complex_parity("abs(a * b)", "a", [3.0, 4.0], Some("b"), Some([3.0, 4.0]));
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_complex_div_abs() {
    // |z1 / z2| = |z1| / |z2|
    check_complex_parity("abs(a / b)", "a", [6.0, 8.0], Some("b"), Some([3.0, 4.0]));
    check_complex_parity("abs(a / b)", "a", [1.0, 0.0], Some("b"), Some([0.0, 1.0]));
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_complex_add_re_im() {
    // re(a + b) and im(a + b)
    check_complex_parity("re(a + b)", "a", [1.0, 2.0], Some("b"), Some([3.0, 4.0]));
    check_complex_parity("im(a + b)", "a", [1.0, 2.0], Some("b"), Some([3.0, 4.0]));
}

// ============================================================================
// Parity tests - mathematical identities
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_abs_squared_equals_norm() {
    // |z|² = norm(z)
    let z = [3.0, 4.0];

    let expr1 = Expr::parse("abs(z) * abs(z)").unwrap();
    let expr2 = Expr::parse("norm(z)").unwrap();

    let var_map: HashMap<String, Value<f32>> =
        [("z".to_string(), Value::Complex(z))].into_iter().collect();
    let registry = complex_registry();

    let result1 = match eval(expr1.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };
    let result2 = match eval(expr2.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    assert_close(result1, result2, "|z|² = norm(z)");
    assert_close(result1, 25.0, "|z|² expected value");
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_conj_mul_equals_norm() {
    // z * conj(z) = norm(z) (real number)
    let z = [3.0, 4.0];
    let expr = Expr::parse("re(z * conj(z))").unwrap();

    let var_map: HashMap<String, Value<f32>> =
        [("z".to_string(), Value::Complex(z))].into_iter().collect();
    let registry = complex_registry();

    let eval_result = match eval(expr.ast(), &var_map, &registry).unwrap() {
        Value::Scalar(s) => s,
        _ => panic!("expected scalar"),
    };

    let jit = ComplexJit::new().unwrap();
    let func = jit
        .compile_scalar(expr.ast(), &[VarSpec::new("z", Type::Complex)])
        .unwrap();
    let cranelift_result = func.call(&z);

    assert_close(eval_result, cranelift_result, "z * conj(z)");
    assert_close(eval_result, 25.0, "z * conj(z) expected value");
}

// ============================================================================
// Randomized parity tests
// ============================================================================

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_abs() {
    let test_values: &[[f32; 2]] = &[
        [0.5, 0.7],
        [1.23, 4.56],
        [100.0, 0.0],
        [0.0, 100.0],
        [-3.0, -4.0],
    ];

    for z in test_values {
        check_complex_parity("abs(z)", "z", *z, None, None);
    }
}

#[cfg(feature = "cranelift")]
#[test]
fn test_parity_random_complex_ops() {
    let test_pairs: &[([f32; 2], [f32; 2])] = &[
        ([1.0, 2.0], [3.0, 4.0]),
        ([0.5, 0.5], [0.5, -0.5]),
        ([-1.0, 1.0], [1.0, -1.0]),
        ([10.0, 0.0], [0.0, 10.0]),
    ];

    for (z1, z2) in test_pairs {
        check_complex_parity("abs(a * b)", "a", *z1, Some("b"), Some(*z2));
        check_complex_parity("abs(a + b)", "a", *z1, Some("b"), Some(*z2));
        check_complex_parity("abs(a - b)", "a", *z1, Some("b"), Some(*z2));
    }
}
