#![no_main]

//! Type-aware fuzzer for dew-quaternion backend parity.
//!
//! Generates expressions that are always well-typed, tracking types through
//! the expression tree to ensure valid operations.
//!
//! Tests eval (interpreter) vs Lua vs Cranelift JIT for scalar, vec3, and quaternion expressions.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rhizome_dew_core::Expr;
use rhizome_dew_quaternion::{quaternion_registry, cranelift::{QuaternionJit, VarSpec}, lua::eval_lua, Value};
use std::collections::HashMap;

/// Types in the quaternion domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QType {
    Scalar,
    Vec3,
    Quaternion,
}

/// A typed expression fragment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TypedExpr {
    expr: String,
    typ: QType,
}

impl TypedExpr {
    fn scalar(expr: String) -> Self {
        Self { expr, typ: QType::Scalar }
    }
    fn vec3(expr: String) -> Self {
        Self { expr, typ: QType::Vec3 }
    }
    fn quat(expr: String) -> Self {
        Self { expr, typ: QType::Quaternion }
    }
}

/// Variable definitions with their types.
struct VarDefs {
    scalar_vars: Vec<String>,
    vec3_vars: Vec<String>,
    quat_vars: Vec<String>,
}

impl VarDefs {
    fn new() -> Self {
        Self {
            scalar_vars: vec!["t".into(), "s".into()],
            vec3_vars: vec!["v".into(), "u".into()],
            quat_vars: vec!["q".into(), "p".into()],
        }
    }

    fn pick_scalar(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.scalar_vars.len())?;
        Ok(TypedExpr::scalar(self.scalar_vars[idx].clone()))
    }

    fn pick_vec3(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.vec3_vars.len())?;
        Ok(TypedExpr::vec3(self.vec3_vars[idx].clone()))
    }

    fn pick_quat(&self, u: &mut Unstructured) -> arbitrary::Result<TypedExpr> {
        let idx = u.choose_index(self.quat_vars.len())?;
        Ok(TypedExpr::quat(self.quat_vars[idx].clone()))
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
                // length(Vec3|Quaternion) -> Scalar
                if u.ratio(1, 2)? {
                    let v = generate_vec3(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("length({})", v.expr)))
                } else {
                    let q = generate_quat(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("length({})", q.expr)))
                }
            }
            _ => {
                // dot(Vec3, Vec3) or dot(Quat, Quat) -> Scalar
                if u.ratio(1, 2)? {
                    let a = generate_vec3(u, vars, depth - 1)?;
                    let b = generate_vec3(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("dot({}, {})", a.expr, b.expr)))
                } else {
                    let a = generate_quat(u, vars, depth - 1)?;
                    let b = generate_quat(u, vars, depth - 1)?;
                    Ok(TypedExpr::scalar(format!("dot({}, {})", a.expr, b.expr)))
                }
            }
        }
    }
}

/// Generate a vec3-typed expression.
fn generate_vec3(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        vars.pick_vec3(u)
    } else {
        let choice: u8 = u.int_in_range(0..=5)?;
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
                // -Vec3
                let inner = generate_vec3(u, vars, depth - 1)?;
                Ok(TypedExpr::vec3(format!("(-{})", inner.expr)))
            }
            4 => {
                // normalize(Vec3) -> Vec3
                let v = generate_vec3(u, vars, depth - 1)?;
                Ok(TypedExpr::vec3(format!("normalize({})", v.expr)))
            }
            _ => {
                // rotate(Vec3, Quat) -> Vec3 or Quat * Vec3
                let v = generate_vec3(u, vars, depth - 1)?;
                let q = generate_quat(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::vec3(format!("rotate({}, {})", v.expr, q.expr)))
                } else {
                    Ok(TypedExpr::vec3(format!("({} * {})", q.expr, v.expr)))
                }
            }
        }
    }
}

/// Generate a quaternion-typed expression.
fn generate_quat(
    u: &mut Unstructured,
    vars: &VarDefs,
    depth: usize,
) -> arbitrary::Result<TypedExpr> {
    if depth == 0 || u.ratio(1, 4)? {
        vars.pick_quat(u)
    } else {
        let choice: u8 = u.int_in_range(0..=6)?;
        match choice {
            0 => {
                // Quat +/- Quat
                let left = generate_quat(u, vars, depth - 1)?;
                let right = generate_quat(u, vars, depth - 1)?;
                let ops = ["+", "-"];
                let op = ops[u.choose_index(ops.len())?];
                Ok(TypedExpr::quat(format!("({} {} {})", left.expr, op, right.expr)))
            }
            1 => {
                // Quat * Quat (Hamilton product)
                let left = generate_quat(u, vars, depth - 1)?;
                let right = generate_quat(u, vars, depth - 1)?;
                Ok(TypedExpr::quat(format!("({} * {})", left.expr, right.expr)))
            }
            2 => {
                // Quat * Scalar or Scalar * Quat
                let q = generate_quat(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                if u.ratio(1, 2)? {
                    Ok(TypedExpr::quat(format!("({} * {})", q.expr, s.expr)))
                } else {
                    Ok(TypedExpr::quat(format!("({} * {})", s.expr, q.expr)))
                }
            }
            3 => {
                // Quat / Scalar
                let q = generate_quat(u, vars, depth - 1)?;
                let s = generate_scalar(u, vars, depth - 1)?;
                Ok(TypedExpr::quat(format!("({} / {})", q.expr, s.expr)))
            }
            4 => {
                // -Quat
                let inner = generate_quat(u, vars, depth - 1)?;
                Ok(TypedExpr::quat(format!("(-{})", inner.expr)))
            }
            5 => {
                // conj/normalize/inverse(Quat) -> Quat
                let funcs = ["conj", "normalize", "inverse"];
                let func = funcs[u.choose_index(funcs.len())?];
                let q = generate_quat(u, vars, depth - 1)?;
                Ok(TypedExpr::quat(format!("{}({})", func, q.expr)))
            }
            _ => {
                // slerp/lerp(Quat, Quat, Scalar) -> Quat or axis_angle(Vec3, Scalar) -> Quat
                if u.ratio(2, 3)? {
                    let funcs = ["slerp", "lerp"];
                    let func = funcs[u.choose_index(funcs.len())?];
                    let a = generate_quat(u, vars, depth - 1)?;
                    let b = generate_quat(u, vars, depth - 1)?;
                    let t = generate_scalar(u, vars, depth - 1)?;
                    Ok(TypedExpr::quat(format!("{}({}, {}, {})", func, a.expr, b.expr, t.expr)))
                } else {
                    let axis = generate_vec3(u, vars, depth - 1)?;
                    let angle = generate_scalar(u, vars, depth - 1)?;
                    Ok(TypedExpr::quat(format!("axis_angle({}, {})", axis.expr, angle.expr)))
                }
            }
        }
    }
}

/// Structured input for quaternion parity fuzzing.
#[derive(Debug)]
struct QuaternionParityInput {
    expr: String,
    output_type: QType,
    scalar_values: HashMap<String, f32>,
    vec3_values: HashMap<String, [f32; 3]>,
    quat_values: HashMap<String, [f32; 4]>,
}

impl<'a> Arbitrary<'a> for QuaternionParityInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let vars = VarDefs::new();

        // Randomly choose output type
        let typed_expr = match u.int_in_range(0..=2)? {
            0 => generate_scalar(u, &vars, 4)?,
            1 => generate_vec3(u, &vars, 4)?,
            _ => generate_quat(u, &vars, 4)?,
        };

        // Generate variable values
        let mut scalar_values = HashMap::new();
        for var in &vars.scalar_vars {
            let val: f32 = u.arbitrary()?;
            let val = if val.is_finite() { val.clamp(-10.0, 10.0) } else { 1.0 };
            scalar_values.insert(var.clone(), val);
        }

        let mut vec3_values = HashMap::new();
        for var in &vars.vec3_vars {
            let x: f32 = u.arbitrary()?;
            let y: f32 = u.arbitrary()?;
            let z: f32 = u.arbitrary()?;
            let x = if x.is_finite() { x.clamp(-10.0, 10.0) } else { 1.0 };
            let y = if y.is_finite() { y.clamp(-10.0, 10.0) } else { 0.0 };
            let z = if z.is_finite() { z.clamp(-10.0, 10.0) } else { 0.0 };
            vec3_values.insert(var.clone(), [x, y, z]);
        }

        let mut quat_values = HashMap::new();
        for var in &vars.quat_vars {
            // Generate unit-ish quaternion for stability
            let x: f32 = u.arbitrary()?;
            let y: f32 = u.arbitrary()?;
            let z: f32 = u.arbitrary()?;
            let w: f32 = u.arbitrary()?;
            let x = if x.is_finite() { x.clamp(-1.0, 1.0) } else { 0.0 };
            let y = if y.is_finite() { y.clamp(-1.0, 1.0) } else { 0.0 };
            let z = if z.is_finite() { z.clamp(-1.0, 1.0) } else { 0.0 };
            let w = if w.is_finite() { w.clamp(-1.0, 1.0) } else { 1.0 };
            // Normalize
            let len = (x * x + y * y + z * z + w * w).sqrt();
            let len = if len > 0.0001 { len } else { 1.0 };
            quat_values.insert(var.clone(), [x / len, y / len, z / len, w / len]);
        }

        Ok(QuaternionParityInput {
            expr: typed_expr.expr,
            output_type: typed_expr.typ,
            scalar_values,
            vec3_values,
            quat_values,
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

/// Compare two vec3 values.
fn vec3_approx_eq(a: &[f32; 3], b: &[f32; 3]) -> bool {
    approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
}

/// Compare two quaternion values.
fn quat_approx_eq(a: &[f32; 4], b: &[f32; 4]) -> bool {
    approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2]) && approx_eq(a[3], b[3])
}

fuzz_target!(|input: QuaternionParityInput| {
    // Parse expression
    let Ok(expr) = Expr::parse(&input.expr) else {
        return;
    };

    // Build variable map
    let mut var_map: HashMap<String, Value<f32>> = HashMap::new();
    for (name, val) in &input.scalar_values {
        var_map.insert(name.clone(), Value::Scalar(*val));
    }
    for (name, val) in &input.vec3_values {
        var_map.insert(name.clone(), Value::Vec3(*val));
    }
    for (name, val) in &input.quat_values {
        var_map.insert(name.clone(), Value::Quaternion(*val));
    }

    let registry = quaternion_registry::<f32>();

    // 1. Direct eval (interpreter)
    let Ok(eval_val) = rhizome_dew_quaternion::eval(expr.ast(), &var_map, &registry) else {
        return;
    };

    // Verify output type matches expected
    let actual_type = match &eval_val {
        Value::Scalar(_) => QType::Scalar,
        Value::Vec3(_) => QType::Vec3,
        Value::Quaternion(_) => QType::Quaternion,
    };
    if actual_type != input.output_type {
        // Type inference mismatch - skip this test case
        return;
    }

    // Build JIT args (shared between all paths)
    let free_vars: Vec<&str> = expr.free_vars().into_iter().collect();
    let mut args: Vec<f32> = Vec::new();
    for var in &free_vars {
        match &var_map[*var] {
            Value::Scalar(s) => args.push(*s),
            Value::Vec3(v) => {
                args.push(v[0]);
                args.push(v[1]);
                args.push(v[2]);
            }
            Value::Quaternion(q) => {
                args.push(q[0]);
                args.push(q[1]);
                args.push(q[2]);
                args.push(q[3]);
            }
        }
    }

    match input.output_type {
        QType::Scalar => {
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
            if let Ok(jit) = QuaternionJit::new() {
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
        QType::Vec3 => {
            let Value::Vec3(eval_vec3) = eval_val else { unreachable!() };

            // 2. Lua evaluation
            if let Ok(lua_val) = eval_lua(expr.ast(), &var_map) {
                let Value::Vec3(lua_vec3) = lua_val else {
                    panic!(
                        "LUA TYPE MISMATCH: expected Vec3, got {:?}\nExpr: {}",
                        lua_val, input.expr
                    );
                };

                if !vec3_approx_eq(&eval_vec3, &lua_vec3) {
                    panic!(
                        "PARITY MISMATCH: eval vs lua\nExpr: {}\nVars: {:?}\nEval: {:?}\nLua: {:?}",
                        input.expr, var_map, eval_vec3, lua_vec3
                    );
                }
            }

            // 3. Cranelift JIT (vec3)
            if let Ok(jit) = QuaternionJit::new() {
                let var_specs: Vec<VarSpec> = free_vars
                    .iter()
                    .map(|v| VarSpec::new(*v, var_map[*v].typ()))
                    .collect();

                if let Ok(compiled) = jit.compile_vec3(expr.ast(), &var_specs) {
                    let jit_val = compiled.call(&args);

                    if !vec3_approx_eq(&eval_vec3, &jit_val) {
                        panic!(
                            "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {:?}\nJIT: {:?}",
                            input.expr, var_map, eval_vec3, jit_val
                        );
                    }
                }
            }
        }
        QType::Quaternion => {
            let Value::Quaternion(eval_quat) = eval_val else { unreachable!() };

            // 2. Lua evaluation
            if let Ok(lua_val) = eval_lua(expr.ast(), &var_map) {
                let Value::Quaternion(lua_quat) = lua_val else {
                    panic!(
                        "LUA TYPE MISMATCH: expected Quaternion, got {:?}\nExpr: {}",
                        lua_val, input.expr
                    );
                };

                if !quat_approx_eq(&eval_quat, &lua_quat) {
                    panic!(
                        "PARITY MISMATCH: eval vs lua\nExpr: {}\nVars: {:?}\nEval: {:?}\nLua: {:?}",
                        input.expr, var_map, eval_quat, lua_quat
                    );
                }
            }

            // 3. Cranelift JIT (quaternion)
            if let Ok(jit) = QuaternionJit::new() {
                let var_specs: Vec<VarSpec> = free_vars
                    .iter()
                    .map(|v| VarSpec::new(*v, var_map[*v].typ()))
                    .collect();

                if let Ok(compiled) = jit.compile_quaternion(expr.ast(), &var_specs) {
                    let jit_val = compiled.call(&args);

                    if !quat_approx_eq(&eval_quat, &jit_val) {
                        panic!(
                            "PARITY MISMATCH: eval vs cranelift\nExpr: {}\nVars: {:?}\nEval: {:?}\nJIT: {:?}",
                            input.expr, var_map, eval_quat, jit_val
                        );
                    }
                }
            }
        }
    }
});
