#![no_main]

//! Type-aware fuzzer for dew-linalg backend parity.
//!
//! Generates expressions that are always well-typed, tracking types through
//! the expression tree to ensure valid operations.
//!
//! Tests eval (interpreter) vs Lua vs Cranelift JIT for scalar-returning expressions.
//! Supports: Scalar, Vec2, Vec3, Mat2, Mat3 (with 3d feature, which is default).

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rhizome_dew_core::Expr;
use rhizome_dew_linalg::{linalg_registry, cranelift::{LinalgJit, VarSpec}, lua::eval_lua, Value};
use std::collections::HashMap;

/// Types in the linalg domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LType {
    Scalar,
    Vec2,
    Vec3,
    Mat2,
    Mat3,
}

/// A typed expression fragment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TypedExpr {
    expr: String,
    typ: LType,
}

impl TypedExpr {
    fn scalar(expr: String) -> Self {
        Self { expr, typ: LType::Scalar }
    }
    fn vec2(expr: String) -> Self {
        Self { expr, typ: LType::Vec2 }
    }
    fn vec3(expr: String) -> Self {
        Self { expr, typ: LType::Vec3 }
    }
    fn mat2(expr: String) -> Self {
        Self { expr, typ: LType::Mat2 }
    }
    fn mat3(expr: String) -> Self {
        Self { expr, typ: LType::Mat3 }
    }
}

/// Variable definitions with their types.
struct VarDefs {
    scalar_vars: Vec<String>,
    vec2_vars: Vec<String>,
    vec3_vars: Vec<String>,
    mat2_vars: Vec<String>,
    mat3_vars: Vec<String>,
}

impl VarDefs {
    fn new() -> Self {
        Self {
            scalar_vars: vec!["s".into(), "t".into()],
            vec2_vars: vec!["a".into(), "b".into()],
            vec3_vars: vec!["p".into(), "q".into()],
            mat2_vars: vec!["M".into()],
            mat3_vars: vec!["N".into()],
        }
    }

    fn pick_scalar(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.scalar_vars.len())?;
        Ok(TypedExpr::scalar(self.scalar_vars[idx].clone()))
    }

    fn pick_vec2(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.vec2_vars.len())?;
        Ok(TypedExpr::vec2(self.vec2_vars[idx].clone()))
    }

    fn pick_vec3(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.vec3_vars.len())?;
        Ok(TypedExpr::vec3(self.vec3_vars[idx].clone()))
    }

    fn pick_mat2(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.mat2_vars.len())?;
        Ok(TypedExpr::mat2(self.mat2_vars[idx].clone()))
    }

    fn pick_mat3(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.mat3_vars.len())?;
        Ok(TypedExpr::mat3(self.mat3_vars[idx].clone()))
    }
}

/// Generate a scalar-typed expression (for JIT testing).
fn generate_scalar(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        if u.ratio(1, 2)? {
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-10.0, 10.0) } else { 1.0 };
            Ok(TypedExpr::scalar(format!("{:.4}", val)))
        } else {
            vars.pick_scalar(u)
        }
    } else {
        let choice: u8 = u.int_in_range(0..=4)?;
        match choice {
            0 => {
                // Scalar op Scalar
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
                // dot(Vec2, Vec2) or dot(Vec3, Vec3) -> Scalar
                if u.ratio(1, 2)? {
                    let a = generate_vec2(u, vars, depth - 1)?;
                    let b = generate_vec2(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("dot({}, {})", a.expr, b.expr)))
                } else {
                    let a = generate_vec3(u, vars, depth - 1)?;
                    let b = generate_vec3(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("dot({}, {})", a.expr, b.expr)))
                }
            }
            _ => {
                // length(Vec2|Vec3) or distance(Vec2, Vec2|Vec3, Vec3) -> Scalar
                if u.ratio(1, 2)? {
                    if u.ratio(1, 2)? {
                        let v = generate_vec2(u, vars, depth - 1)?;
                        Ok(TypedExpr::scalar(format!("length({})", v.expr)))
                    } else {
                        let v = generate_vec3(u, vars, depth - 1)?;
                        Ok(TypedExpr::scalar(format!("length({})", v.expr)))
                    }
                } else {
                    if u.ratio(1, 2)? {
                        let a = generate_vec2(u, vars, depth - 1)?;
                        let b = generate_vec2(u, vars, depth - 1)?;
                        Ok(TypedExpr::scalar(format!("distance({}, {})", a.expr, b.expr)))
                    } else {
                        let a = generate_vec3(u, vars, depth - 1)?;
                        let b = generate_vec3(u, vars, depth - 1)?;
                        Ok(TypedExpr::scalar(format!("distance({}, {})", a.expr, b.expr)))
                    }
                }
            }
        }
    }
}

/// Generate a Vec2-typed expression.
fn generate_vec2(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        vars.pick_vec2(u)
    } else {
        let choice: u8 = u.int_in_range(0..=6)?;
        match choice {
            0 => {
                // Vec2 +/- Vec2
                let left = generate_vec2(u, vars, depth - 1)?;
                let right = generate_vec2(u, vars, depth - 1)?;
                let ops = ["+", "-"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::vec2(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Vec2 * Scalar or Scalar * Vec2
                let v = generate_vec2(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::vec2(format!("({} * {})", v.expr, s.expr)))
                } else {
                    Ok(TypedExpr::vec2(format!("({} * {})", s.expr, v.expr)))
                }
            }
            2 => {
                // Vec2 / Scalar
                let v = generate_vec2(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::vec2(format!("({} / {})", v.expr, s.expr)))
            }
            3 => {
                // Mat2 * Vec2 or Vec2 * Mat2
                let m = generate_mat2(u, vars, depth - 1)?;
                let v = generate_vec2(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::vec2(format!("({} * {})", m.expr, v.expr)))
                } else {
                    Ok(TypedExpr::vec2(format!("({} * {})", v.expr, m.expr)))
                }
            }
            4 => {
                // -Vec2
                let inner = generate_vec2(u, vars, depth - 1)?;
                Ok(TypedExpr::vec2(format!("(-{})", inner.expr)))
            }
            5 => {
                // normalize(Vec2), hadamard(Vec2, Vec2), reflect(Vec2, Vec2)
                let funcs = ["normalize", "hadamard", "reflect"];
                let func = funcs[u.choose_index(funcs.len())?];
                if func == "normalize" {
                    let v = generate_vec2(u, vars, depth - 1)?;
                    Ok(TypedExpr::vec2(format!("normalize({})", v.expr)))
                } else {
                    let a = generate_vec2(u, vars, depth - 1)?;
                    let b = generate_vec2(u, vars, depth - 1)?;
                    Ok(TypedExpr::vec2(format!("{}({}, {})", func, a.expr, b.expr)))
                }
            }
            _ => {
                // lerp/mix(Vec2, Vec2, Scalar) -> Vec2
                let funcs = ["lerp", "mix"];
                let func = funcs[u.choose_index(funcs.len())?];
                let a = generate_vec2(u, vars, depth - 1)?;
                let b = generate_vec2(u, vars, depth - 1)?;
                let t = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::vec2(format!("{}({}, {}, {})", func, a.expr, b.expr, t.expr)))
            }
        }
    }
}

/// Generate a Vec3-typed expression.
fn generate_vec3(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        vars.pick_vec3(u)
    } else {
        let choice: u8 = u.int_in_range(0..=6)?;
        match choice {
            0 => {
                // Vec3 +/- Vec3
                let left = generate_vec3(u, vars, depth - 1)?;
                let right = generate_vec3(u, vars, depth - 1)?;
                let ops = ["+", "-"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::vec3(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Vec3 * Scalar or Scalar * Vec3
                let v = generate_vec3(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::vec3(format!("({} * {})", v.expr, s.expr)))
                } else {
                    Ok(TypedExpr::vec3(format!("({} * {})", s.expr, v.expr)))
                }
            }
            2 => {
                // Vec3 / Scalar
                let v = generate_vec3(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::vec3(format!("({} / {})", v.expr, s.expr)))
            }
            3 => {
                // Mat3 * Vec3 or Vec3 * Mat3
                let m = generate_mat3(u, vars, depth - 1)?;
                let v = generate_vec3(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::vec3(format!("({} * {})", m.expr, v.expr)))
                } else {
                    Ok(TypedExpr::vec3(format!("({} * {})", v.expr, m.expr)))
                }
            }
            4 => {
                // -Vec3
                let inner = generate_vec3(u, vars, depth - 1)?;
                Ok(TypedExpr::vec3(format!("(-{})", inner.expr)))
            }
            5 => {
                // cross(Vec3, Vec3) -> Vec3, normalize(Vec3), hadamard, reflect
                let funcs = ["cross", "normalize", "hadamard", "reflect"];
                let func = funcs[u.choose_index(funcs.len())?];
                if func == "normalize" {
                    let v = generate_vec3(u, vars, depth - 1)?;
                    Ok(TypedExpr::vec3(format!("normalize({})", v.expr)))
                } else {
                    let a = generate_vec3(u, vars, depth - 1)?;
                    let b = generate_vec3(u, vars, depth - 1)?;
                    Ok(TypedExpr::vec3(format!("{}({}, {})", func, a.expr, b.expr)))
                }
            }
            _ => {
                // lerp/mix(Vec3, Vec3, Scalar) -> Vec3
                let funcs = ["lerp", "mix"];
                let func = funcs[u.choose_index(funcs.len())?];
                let a = generate_vec3(u, vars, depth - 1)?;
                let b = generate_vec3(u, vars, depth - 1)?;
                let t = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::vec3(format!("{}({}, {}, {})", func, a.expr, b.expr, t.expr)))
            }
        }
    }
}

/// Generate a Mat2-typed expression.
fn generate_mat2(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 3)? {
        vars.pick_mat2(u)
    } else {
        let choice: u8 = u.int_in_range(0..=3)?;
        match choice {
            0 => {
                // Mat2 +/- Mat2
                let left = generate_mat2(u, vars, depth - 1)?;
                let right = generate_mat2(u, vars, depth - 1)?;
                let ops = ["+", "-"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::mat2(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Mat2 * Mat2
                let left = generate_mat2(u, vars, depth - 1)?;
                let right = generate_mat2(u, vars, depth - 1)?;
                Ok(TypedExpr::mat2(format!("({} * {})", left.expr, right.expr)))
            }
            2 => {
                // Mat2 * Scalar or Scalar * Mat2
                let m = generate_mat2(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::mat2(format!("({} * {})", m.expr, s.expr)))
                } else {
                    Ok(TypedExpr::mat2(format!("({} * {})", s.expr, m.expr)))
                }
            }
            _ => {
                // -Mat2
                let inner = generate_mat2(u, vars, depth - 1)?;
                Ok(TypedExpr::mat2(format!("(-{})", inner.expr)))
            }
        }
    }
}

/// Generate a Mat3-typed expression.
fn generate_mat3(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 3)? {
        vars.pick_mat3(u)
    } else {
        let choice: u8 = u.int_in_range(0..=3)?;
        match choice {
            0 => {
                // Mat3 +/- Mat3
                let left = generate_mat3(u, vars, depth - 1)?;
                let right = generate_mat3(u, vars, depth - 1)?;
                let ops = ["+", "-"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::mat3(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Mat3 * Mat3
                let left = generate_mat3(u, vars, depth - 1)?;
                let right = generate_mat3(u, vars, depth - 1)?;
                Ok(TypedExpr::mat3(format!("({} * {})", left.expr, right.expr)))
            }
            2 => {
                // Mat3 * Scalar or Scalar * Mat3
                let m = generate_mat3(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::mat3(format!("({} * {})", m.expr, s.expr)))
                } else {
                    Ok(TypedExpr::mat3(format!("({} * {})", s.expr, m.expr)))
                }
            }
            _ => {
                // -Mat3
                let inner = generate_mat3(u, vars, depth - 1)?;
                Ok(TypedExpr::mat3(format!("(-{})", inner.expr)))
            }
        }
    }
}

/// Structured input for linalg parity fuzzing.
#[derive(Debug)]
struct LinalgParityInput {
    expr: String,
    scalar_values: HashMap<String, f32>,
    vec2_values: HashMap<String, [f32; 2]>,
    vec3_values: HashMap<String, [f32; 3]>,
    mat2_values: HashMap<String, [f32; 4]>,
    mat3_values: HashMap<String, [f32; 9]>,
}

impl<'a> Arbitrary<'a> for LinalgParityInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let vars = VarDefs::new();

        // Generate scalar-returning expression (JIT only supports scalar output)
        let typed_expr = generate_scalar(u, &vars, 4)?;

        // Generate variable values
        let mut scalar_values = HashMap::new();
        for var in &vars.scalar_vars {
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-10.0, 10.0) } else { 1.0 };
            scalar_values.insert(var.clone(), val);
        }

        let mut vec2_values = HashMap::new();
        for var in &vars.vec2_vars {
            let vals: [f32; 2] = [u.arbitrary()?, u.arbitrary()?];
            let vals = vals.map(|v| if v.is_finite() { v.clamp(-10.0, 10.0) } else { 1.0 });
            vec2_values.insert(var.clone(), vals);
        }

        let mut vec3_values = HashMap::new();
        for var in &vars.vec3_vars {
            let vals: [f32; 3] = [u.arbitrary()?, u.arbitrary()?, u.arbitrary()?];
            let vals = vals.map(|v| if v.is_finite() { v.clamp(-10.0, 10.0) } else { 1.0 });
            vec3_values.insert(var.clone(), vals);
        }

        let mut mat2_values = HashMap::new();
        for var in &vars.mat2_vars {
            let mut vals = [0.0f32; 4];
            for v in &mut vals {
                *v = u.arbitrary()?;
                *v = if v.is_finite() { v.clamp(-5.0, 5.0) } else { 1.0 };
            }
            mat2_values.insert(var.clone(), vals);
        }

        let mut mat3_values = HashMap::new();
        for var in &vars.mat3_vars {
            let mut vals = [0.0f32; 9];
            for v in &mut vals {
                *v = u.arbitrary()?;
                *v = if v.is_finite() { v.clamp(-5.0, 5.0) } else { 1.0 };
            }
            mat3_values.insert(var.clone(), vals);
        }

        Ok(LinalgParityInput {
            expr: typed_expr.expr,
            scalar_values,
            vec2_values,
            vec3_values,
            mat2_values,
            mat3_values,
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

fuzz_target!(|input: LinalgParityInput| {
    // Parse expression
    let Ok(expr) = Expr::parse(&input.expr) else {
        return;
    };

    // Build variable map
    let mut var_map: HashMap<String, Value<f32>> = HashMap::new();
    for (name, val) in &input.scalar_values {
        var_map.insert(name.clone(), Value::Scalar(*val));
    }
    for (name, val) in &input.vec2_values {
        var_map.insert(name.clone(), Value::Vec2(*val));
    }
    for (name, val) in &input.vec3_values {
        var_map.insert(name.clone(), Value::Vec3(*val));
    }
    for (name, val) in &input.mat2_values {
        var_map.insert(name.clone(), Value::Mat2(*val));
    }
    for (name, val) in &input.mat3_values {
        var_map.insert(name.clone(), Value::Mat3(*val));
    }

    let registry = linalg_registry::<f32>();

    // 1. Direct eval (interpreter)
    let Ok(eval_val) = rhizome_dew_linalg::eval(expr.ast(), &var_map, &registry) else {
        return;
    };

    // Must be scalar for JIT comparison
    let Value::Scalar(eval_scalar) = eval_val else {
        return;
    };

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

    // 3. Cranelift JIT
    let free_vars: Vec<&str> = expr.free_vars().into_iter().collect();
    if let Ok(jit) = LinalgJit::new() {
        // Build VarSpec list
        let var_specs: Vec<VarSpec> = free_vars
            .iter()
            .map(|v| VarSpec::new(*v, var_map[*v].typ()))
            .collect();

        if let Ok(compiled) = jit.compile_scalar(expr.ast(), &var_specs) {
            // Flatten arguments
            let mut args: Vec<f32> = Vec::new();
            for var in &free_vars {
                match &var_map[*var] {
                    Value::Scalar(s) => args.push(*s),
                    Value::Vec2(v) => {
                        args.extend_from_slice(v);
                    }
                    Value::Vec3(v) => {
                        args.extend_from_slice(v);
                    }
                    Value::Mat2(m) => {
                        args.extend_from_slice(m);
                    }
                    Value::Mat3(m) => {
                        args.extend_from_slice(m);
                    }
                    #[allow(unreachable_patterns)]
                    _ => {}
                }
            }

            let jit_val = compiled.call(&args);

            if !approx_eq(eval_scalar, jit_val) {
                panic!(
                    "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {}\nJIT: {}",
                    input.expr, var_map, eval_scalar, jit_val
                );
            }
        }
    }
});
