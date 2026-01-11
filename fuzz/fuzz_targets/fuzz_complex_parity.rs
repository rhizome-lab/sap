#![no_main]

//! Type-aware fuzzer for dew-complex backend parity.
//!
//! Generates expressions that are always well-typed, tracking types through
//! the expression tree to ensure valid operations.
//!
//! Tests eval (interpreter) vs Lua vs Cranelift JIT for both scalar and complex expressions.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rhizome_dew_complex::{complex_registry, cranelift::{ComplexJit, VarSpec}, lua::eval_lua, Value};
use rhizome_dew_core::Expr;
use std::collections::HashMap;

/// Types in the complex domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CType {
    Scalar,
    Complex,
}

/// A typed expression fragment.
#[derive(Debug, Clone)]
struct TypedExpr {
    expr: String,
    typ: CType,
}

impl TypedExpr {
    fn scalar(expr: String) -> Self {
        Self { expr, typ: CType::Scalar }
    }
    fn complex(expr: String) -> Self {
        Self { expr, typ: CType::Complex }
    }
}

/// Variable definitions with their types.
struct VarDefs {
    scalar_vars: Vec<String>,
    complex_vars: Vec<String>,
}

impl VarDefs {
    fn new() -> Self {
        Self {
            scalar_vars: vec!["a".into(), "b".into(), "c".into()],
            complex_vars: vec!["z".into(), "w".into()],
        }
    }

    fn pick_scalar(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.scalar_vars.len())?;
        Ok(TypedExpr::scalar(self.scalar_vars[idx].clone()))
    }

    fn pick_complex(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.complex_vars.len())?;
        Ok(TypedExpr::complex(self.complex_vars[idx].clone()))
    }
}

/// Generate a scalar-typed expression (for JIT testing).
fn generate_scalar(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        // Terminal: scalar variable or number
        if u.ratio(1, 2)? {
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-10.0, 10.0) } else { 1.0 };
            Ok(TypedExpr::scalar(format!("{:.4}", val)))
        } else {
            vars.pick_scalar(u)
        }
    } else {
        let choice: u8 = u.int_in_range(0..=6)?;
        match choice {
            0 => {
                // Scalar + Scalar
                let left = generate_scalar(u, vars, depth - 1)?;
                let right = generate_scalar(u, vars, depth - 1)?;
                let ops = ["+", "-", "*", "/"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::scalar(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Scalar ^ Scalar
                let base = generate_scalar(u, vars, depth - 1)?;
                let exp = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::scalar(format!("({} ^ {})", base.expr, exp.expr)))
            }
            2 => {
                // -Scalar
                let inner = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::scalar(format!("(-{})", inner.expr)))
            }
            3 => {
                // re(Complex) -> Scalar
                let z = generate_complex(u, vars, depth - 1)?;
                Ok(TypedExpr::scalar(format!("re({})", z.expr)))
            }
            4 => {
                // im(Complex) -> Scalar
                let z = generate_complex(u, vars, depth - 1)?;
                Ok(TypedExpr::scalar(format!("im({})", z.expr)))
            }
            5 => {
                // abs(Scalar|Complex) -> Scalar, arg(Complex) -> Scalar, norm(Complex) -> Scalar
                let funcs = ["abs", "arg", "norm"];
                let func = funcs[u.choose_index(funcs.len())?];
                if func == "abs" && u.ratio(1, 2)? {
                    let s = generate_scalar(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("abs({})", s.expr)))
                } else {
                    let z = generate_complex(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("{}({})", func, z.expr)))
                }
            }
            _ => {
                // Scalar function: exp, log, sqrt on scalar
                let funcs = ["exp", "log", "sqrt"];
                let func = funcs[u.choose_index(funcs.len())?];
                let s = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::scalar(format!("{}({})", func, s.expr)))
            }
        }
    }
}

/// Generate a complex-typed expression.
fn generate_complex(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        // Terminal: complex variable
        vars.pick_complex(u)
    } else {
        let choice: u8 = u.int_in_range(0..=5)?;
        match choice {
            0 => {
                // Complex op Complex or Scalar op Complex or Complex op Scalar
                let ops = ["+", "-", "*", "/"];
                let op = ops[u.choose_index(ops.len())?];
                let left = generate_any(u, vars, depth - 1)?;
                let right = generate_any(u, vars, depth - 1)?;
                // At least one must be complex for result to be complex
                let (l, r) = if left.typ == CType::Scalar && right.typ == CType::Scalar {
                    // Force one to be complex
                    if u.ratio(1, 2)? {
                        (generate_complex(u, vars, depth - 1)?, right)
                    } else {
                        (left, generate_complex(u, vars, depth - 1)?)
                    }
                } else {
                    (left, right)
                };
                Ok(TypedExpr::complex(format!("({} {} {})", l.expr, op, r.expr)))
            }
            1 => {
                // Complex ^ Scalar or Complex ^ Complex
                let base = generate_complex(u, vars, depth - 1)?;
                let exp = generate_any(u, vars, depth - 1)?;
                Ok(TypedExpr::complex(format!("({} ^ {})", base.expr, exp.expr)))
            }
            2 => {
                // -Complex
                let inner = generate_complex(u, vars, depth - 1)?;
                Ok(TypedExpr::complex(format!("(-{})", inner.expr)))
            }
            3 => {
                // conj(Complex) -> Complex
                let z = generate_complex(u, vars, depth - 1)?;
                Ok(TypedExpr::complex(format!("conj({})", z.expr)))
            }
            4 => {
                // exp/log/sqrt on Complex
                let funcs = ["exp", "log", "sqrt"];
                let func = funcs[u.choose_index(funcs.len())?];
                let z = generate_complex(u, vars, depth - 1)?;
                Ok(TypedExpr::complex(format!("{}({})", func, z.expr)))
            }
            _ => {
                // polar(Scalar, Scalar) -> Complex
                let r = generate_scalar(u, vars, depth - 1)?;
                let theta = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::complex(format!("polar({}, {})", r.expr, theta.expr)))
            }
        }
    }
}

/// Generate any type.
fn generate_any(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if u.ratio(1, 2)? {
        generate_scalar(u, vars, depth)
    } else {
        generate_complex(u, vars, depth)
    }
}

/// Structured input for complex parity fuzzing.
#[derive(Debug)]
struct ComplexParityInput {
    expr: String,
    output_type: CType,
    scalar_values: HashMap<String, f32>,
    complex_values: HashMap<String, [f32; 2]>,
}

impl<'a> Arbitrary<'a> for ComplexParityInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let vars = VarDefs::new();

        // Randomly choose scalar or complex output type
        let typed_expr = if u.ratio(1, 2)? {
            generate_scalar(u, &vars, 4)?
        } else {
            generate_complex(u, &vars, 4)?
        };

        // Generate variable values
        let mut scalar_values = HashMap::new();
        for var in &vars.scalar_vars {
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-10.0, 10.0) } else { 1.0 };
            scalar_values.insert(var.clone(), val);
        }

        let mut complex_values = HashMap::new();
        for var in &vars.complex_vars {
            let re: f32 = u.arbitrary()?;
            let im: f32 = u.arbitrary()?;
            let re = if re.is_finite() { re.clamp(-10.0, 10.0) } else { 1.0 };
            let im = if im.is_finite() { im.clamp(-10.0, 10.0) } else { 0.0 };
            complex_values.insert(var.clone(), [re, im]);
        }

        Ok(ComplexParityInput {
            expr: typed_expr.expr,
            output_type: typed_expr.typ,
            scalar_values,
            complex_values,
        })
    }
}

/// Compare two f32 values with tolerance for NaN/Inf.
fn approx_eq(a: f32, b: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return false;
    }
    let diff = (a - b).abs();
    diff < 1e-4 || diff / a.abs().max(b.abs()).max(1e-10) < 1e-4
}

/// Compare two complex values.
fn complex_approx_eq(a: &[f32; 2], b: &[f32; 2]) -> bool {
    approx_eq(a[0], b[0]) && approx_eq(a[1], b[1])
}

fuzz_target!(|input: ComplexParityInput| {
    // Parse expression
    let Ok(expr) = Expr::parse(&input.expr) else {
        return;
    };

    // Build variable map
    let mut var_map: HashMap<String, Value<f32>> = HashMap::new();
    for (name, val) in &input.scalar_values {
        var_map.insert(name.clone(), Value::Scalar(*val));
    }
    for (name, val) in &input.complex_values {
        var_map.insert(name.clone(), Value::Complex(*val));
    }

    let registry = complex_registry::<f32>();

    // 1. Direct eval (interpreter)
    let Ok(eval_val) = rhizome_dew_complex::eval(expr.ast(), &var_map, &registry) else {
        return;
    };

    // Verify output type matches expected
    let actual_type = match &eval_val {
        Value::Scalar(_) => CType::Scalar,
        Value::Complex(_) => CType::Complex,
    };
    if actual_type != input.output_type {
        // Type inference mismatch - skip this test case
        return;
    }

    // Build JIT args (shared between scalar and complex paths)
    let free_vars: Vec<&str> = expr.free_vars().into_iter().collect();
    let mut args: Vec<f32> = Vec::new();
    for var in &free_vars {
        match &var_map[*var] {
            Value::Scalar(s) => args.push(*s),
            Value::Complex(c) => {
                args.push(c[0]);
                args.push(c[1]);
            }
        }
    }

    match input.output_type {
        CType::Scalar => {
            let Value::Scalar(eval_scalar) = eval_val else { unreachable!() };

            // 2. Lua evaluation
            if let Ok(lua_val) = eval_lua(expr.ast(), &var_map) {
                let Value::Scalar(lua_scalar) = lua_val else {
                    panic!(
                        "LUA TYPE MISMATCH: expected Scalar, got {:?}\nExpr: {}",
                        lua_val, input.expr
                    );
                };

                if !approx_eq(eval_scalar, lua_scalar) {
                    panic!(
                        "PARITY MISMATCH: eval vs lua\nExpr: {}\nVars: {:?}\nEval: {}\nLua: {}",
                        input.expr, var_map, eval_scalar, lua_scalar
                    );
                }
            }

            // 3. Cranelift JIT (scalar)
            if let Ok(jit) = ComplexJit::new() {
                let var_specs: Vec<VarSpec> = free_vars
                    .iter()
                    .map(|v| VarSpec::new(*v, var_map[*v].typ()))
                    .collect();

                if let Ok(compiled) = jit.compile_scalar(expr.ast(), &var_specs) {
                    let jit_val = compiled.call(&args);

                    if !approx_eq(eval_scalar, jit_val) {
                        panic!(
                            "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {}\nJIT: {}",
                            input.expr, var_map, eval_scalar, jit_val
                        );
                    }
                }
            }
        }
        CType::Complex => {
            let Value::Complex(eval_complex) = eval_val else { unreachable!() };

            // 2. Lua evaluation
            if let Ok(lua_val) = eval_lua(expr.ast(), &var_map) {
                let Value::Complex(lua_complex) = lua_val else {
                    panic!(
                        "LUA TYPE MISMATCH: expected Complex, got {:?}\nExpr: {}",
                        lua_val, input.expr
                    );
                };

                if !complex_approx_eq(&eval_complex, &lua_complex) {
                    panic!(
                        "PARITY MISMATCH: eval vs lua\nExpr: {}\nVars: {:?}\nEval: {:?}\nLua: {:?}",
                        input.expr, var_map, eval_complex, lua_complex
                    );
                }
            }

            // 3. Cranelift JIT (complex)
            if let Ok(jit) = ComplexJit::new() {
                let var_specs: Vec<VarSpec> = free_vars
                    .iter()
                    .map(|v| VarSpec::new(*v, var_map[*v].typ()))
                    .collect();

                if let Ok(compiled) = jit.compile_complex(expr.ast(), &var_specs) {
                    let jit_val = compiled.call(&args);

                    if !complex_approx_eq(&eval_complex, &jit_val) {
                        panic!(
                            "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {:?}\nJIT: {:?}",
                            input.expr, var_map, eval_complex, jit_val
                        );
                    }
                }
            }
        }
    }
});
