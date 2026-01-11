#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{scalar_registry, cranelift::ScalarJit, lua::eval_lua, FunctionRegistry};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Get all scalar functions with their arities (cached).
fn scalar_funcs() -> &'static Vec<(String, usize)> {
    static FUNCS: OnceLock<Vec<(String, usize)>> = OnceLock::new();
    FUNCS.get_or_init(|| {
        let registry: FunctionRegistry<f32> = scalar_registry();
        registry
            .names()
            .map(|name| {
                let arity = registry.get(name).map(|f| f.arg_count()).unwrap_or(0);
                (name.to_string(), arity)
            })
            .collect()
    })
}

/// Generate a random expression string that uses scalar functions.
fn generate_expr(u: &mut Unstructured, depth: usize) -> arbitrary::Result<String> {
    if depth == 0 || u.ratio(1, 4)? {
        // Terminal: variable or number
        if u.ratio(1, 2)? {
            Ok(format!("{:.6}", u.arbitrary::<f32>()?))
        } else {
            let vars = ["x", "y", "z", "a", "b", "c"];
            Ok(vars[u.choose_index(vars.len())?].to_string())
        }
    } else {
        let choice: u8 = u.int_in_range(0..=4)?;
        match choice {
            0 => {
                // Binary op
                let ops = ["+", "-", "*", "/", "^"];
                let op = ops[u.choose_index(ops.len())?];
                let left = generate_expr(u, depth - 1)?;
                let right = generate_expr(u, depth - 1)?;
                Ok(format!("({} {} {})", left, op, right))
            }
            1 => {
                // Unary minus
                let inner = generate_expr(u, depth - 1)?;
                Ok(format!("(-{})", inner))
            }
            2 => {
                // Function call
                let funcs = scalar_funcs();
                let idx = u.choose_index(funcs.len())?;
                let (name, arity) = &funcs[idx];
                let args: Vec<String> = (0..*arity)
                    .map(|_| generate_expr(u, depth - 1))
                    .collect::<Result<_, _>>()?;
                Ok(format!("{}({})", name, args.join(", ")))
            }
            3 => {
                // Comparison (cond feature)
                let ops = ["<", "<=", ">", ">=", "==", "!="];
                let op = ops[u.choose_index(ops.len())?];
                let left = generate_expr(u, depth - 1)?;
                let right = generate_expr(u, depth - 1)?;
                Ok(format!("({} {} {})", left, op, right))
            }
            _ => {
                // If/then/else
                let cond = generate_expr(u, depth - 1)?;
                let then_expr = generate_expr(u, depth - 1)?;
                let else_expr = generate_expr(u, depth - 1)?;
                Ok(format!("(if {} then {} else {})", cond, then_expr, else_expr))
            }
        }
    }
}

/// Structured input for backend parity fuzzing.
#[derive(Debug)]
struct ParityInput {
    expr: String,
    var_values: HashMap<String, f32>,
}

impl<'a> Arbitrary<'a> for ParityInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let expr = generate_expr(u, 4)?;
        let mut var_values = HashMap::new();
        for var in ["x", "y", "z", "a", "b", "c"] {
            // Use values in reasonable range to avoid NaN/Inf in most cases
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-100.0, 100.0) } else { 1.0 };
            var_values.insert(var.to_string(), val);
        }
        Ok(ParityInput { expr, var_values })
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

fuzz_target!(|input: ParityInput| {
    // Parse expression
    let Ok(expr) = Expr::parse(&input.expr) else {
        return; // Generated invalid syntax (shouldn't happen often)
    };

    let registry = scalar_registry::<f32>();

    // 1. Direct eval (interpreter)
    let eval_result = rhizome_dew_scalar::eval(expr.ast(), &input.var_values, &registry);
    let Ok(eval_val) = eval_result else {
        return; // Unknown function/variable
    };

    // 2. Lua eval
    if let Ok(lua_val) = eval_lua(expr.ast(), &input.var_values) {
        if !approx_eq(eval_val, lua_val) {
            panic!(
                "PARITY MISMATCH: eval vs lua\nExpr: {}\nVars: {:?}\nEval: {}\nLua: {}",
                input.expr, input.var_values, eval_val, lua_val
            );
        }
    }

    // 3. Cranelift JIT
    let free_vars: Vec<&str> = expr.free_vars().into_iter().collect();
    if free_vars.len() <= 6 {
        if let Ok(jit) = ScalarJit::new() {
            if let Ok(compiled) = jit.compile(expr.ast(), &free_vars) {
                let args: Vec<f32> = free_vars.iter().map(|v| input.var_values[*v]).collect();
                let jit_val = compiled.call(&args);

                if !approx_eq(eval_val, jit_val) {
                    panic!(
                        "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {}\nJIT: {}",
                        input.expr, input.var_values, eval_val, jit_val
                    );
                }
            }
        }
    }
});
