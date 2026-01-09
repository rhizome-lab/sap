//! Lua code generation for scalar expressions.
//!
//! Compiles expression ASTs to Lua code and optionally executes via mlua.

use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Errors
// ============================================================================

/// Lua emission error.
#[derive(Debug, Clone, PartialEq)]
pub enum LuaError {
    /// Unknown function.
    UnknownFunction(String),
}

impl std::fmt::Display for LuaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LuaError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
        }
    }
}

impl std::error::Error for LuaError {}

// ============================================================================
// Lua expression output
// ============================================================================

/// Result of emitting Lua code.
#[derive(Debug, Clone)]
pub struct LuaExpr {
    /// The Lua expression string.
    pub code: String,
}

impl LuaExpr {
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

// ============================================================================
// Function emission
// ============================================================================

/// Emit Lua code for a function call.
fn emit_func(name: &str, args: &[String]) -> Option<String> {
    Some(match name {
        // Constants
        "pi" => "math.pi".to_string(),
        "e" => "math.exp(1)".to_string(),
        "tau" => "(2 * math.pi)".to_string(),

        // Trig - direct mapping
        "sin" => format!("math.sin({})", args.first()?),
        "cos" => format!("math.cos({})", args.first()?),
        "tan" => format!("math.tan({})", args.first()?),
        "asin" => format!("math.asin({})", args.first()?),
        "acos" => format!("math.acos({})", args.first()?),
        "atan" => format!("math.atan({})", args.first()?),
        "atan2" => format!("math.atan({}, {})", args.first()?, args.get(1)?),

        // Hyperbolic - computed (Lua doesn't have these)
        "sinh" => {
            let x = args.first()?;
            format!("((math.exp({x}) - math.exp(-{x})) / 2)")
        }
        "cosh" => {
            let x = args.first()?;
            format!("((math.exp({x}) + math.exp(-{x})) / 2)")
        }
        "tanh" => {
            let x = args.first()?;
            format!("((math.exp({x}) - math.exp(-{x})) / (math.exp({x}) + math.exp(-{x})))")
        }

        // Exp/log
        "exp" => format!("math.exp({})", args.first()?),
        "exp2" => format!("(2 ^ {})", args.first()?),
        "log" | "ln" => format!("math.log({})", args.first()?),
        "log2" => format!("math.log({}, 2)", args.first()?),
        "log10" => format!("math.log({}, 10)", args.first()?),
        "pow" => format!("({} ^ {})", args.first()?, args.get(1)?),
        "sqrt" => format!("math.sqrt({})", args.first()?),
        "inversesqrt" => format!("(1 / math.sqrt({}))", args.first()?),

        // Common math
        "abs" => format!("math.abs({})", args.first()?),
        "sign" => {
            let x = args.first()?;
            format!("(({x} > 0 and 1) or ({x} < 0 and -1) or 0)")
        }
        "floor" => format!("math.floor({})", args.first()?),
        "ceil" => format!("math.ceil({})", args.first()?),
        "round" => {
            let x = args.first()?;
            format!("(({x} >= 0) and math.floor({x} + 0.5) or math.ceil({x} - 0.5))")
        }
        "trunc" => {
            let x = args.first()?;
            format!("(({x} >= 0) and math.floor({x}) or math.ceil({x}))")
        }
        "fract" => {
            let x = args.first()?;
            format!("({x} - math.floor({x}))")
        }
        "min" => format!("math.min({}, {})", args.first()?, args.get(1)?),
        "max" => format!("math.max({}, {})", args.first()?, args.get(1)?),
        "clamp" => {
            let (x, lo, hi) = (args.first()?, args.get(1)?, args.get(2)?);
            format!("math.max({lo}, math.min({hi}, {x}))")
        }
        "saturate" => {
            let x = args.first()?;
            format!("math.max(0, math.min(1, {x}))")
        }

        // Interpolation
        "lerp" | "mix" => {
            let (a, b, t) = (args.first()?, args.get(1)?, args.get(2)?);
            format!("({a} + ({b} - {a}) * {t})")
        }
        "step" => {
            let (edge, x) = (args.first()?, args.get(1)?);
            format!("(({x} < {edge}) and 0 or 1)")
        }
        "smoothstep" => {
            let (e0, e1, x) = (args.first()?, args.get(1)?, args.get(2)?);
            format!(
                "(function() local t = math.max(0, math.min(1, ({x} - {e0}) / ({e1} - {e0}))); return t * t * (3 - 2 * t) end)()"
            )
        }
        "inverse_lerp" => {
            let (a, b, v) = (args.first()?, args.get(1)?, args.get(2)?);
            format!("(({v} - {a}) / ({b} - {a}))")
        }
        "remap" => {
            let (x, in_lo, in_hi, out_lo, out_hi) = (
                args.first()?,
                args.get(1)?,
                args.get(2)?,
                args.get(3)?,
                args.get(4)?,
            );
            format!("({out_lo} + ({out_hi} - {out_lo}) * (({x} - {in_lo}) / ({in_hi} - {in_lo})))")
        }

        _ => return None,
    })
}

// ============================================================================
// Code generation
// ============================================================================

/// Emit Lua code for an AST.
pub fn emit_lua(ast: &Ast) -> Result<LuaExpr, LuaError> {
    Ok(LuaExpr::new(emit(ast)?))
}

/// Generate a complete Lua function.
pub fn emit_lua_fn(name: &str, ast: &Ast, params: &[&str]) -> Result<String, LuaError> {
    let param_list = params.join(", ");
    let body = emit(ast)?;
    Ok(format!(
        "function {}({})\n    return {}\nend",
        name, param_list, body
    ))
}

fn emit(ast: &Ast) -> Result<String, LuaError> {
    match ast {
        Ast::Num(n) => Ok(format_float(*n)),
        Ast::Var(name) => Ok(name.clone()),
        Ast::BinOp(op, left, right) => {
            let l = emit_with_parens(left, Some(*op), true)?;
            let r = emit_with_parens(right, Some(*op), false)?;
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            };
            Ok(format!("{} {} {}", l, op_str, r))
        }
        Ast::UnaryOp(op, inner) => {
            let inner_str = emit_with_parens(inner, None, false)?;
            match op {
                UnaryOp::Neg => Ok(format!("-{}", inner_str)),
            }
        }
        Ast::Call(name, args) => {
            let args_str: Vec<String> = args.iter().map(|a| emit(a)).collect::<Result<_, _>>()?;

            emit_func(name, &args_str).ok_or_else(|| LuaError::UnknownFunction(name.clone()))
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
) -> Result<String, LuaError> {
    let inner = emit(ast)?;

    let needs_parens = match ast {
        Ast::BinOp(child_op, _, _) => {
            if let Some(parent) = parent_op {
                let parent_prec = precedence(parent);
                let child_prec = precedence(*child_op);
                if child_prec < parent_prec {
                    true
                } else if child_prec == parent_prec && !is_left {
                    matches!(parent, BinOp::Sub | BinOp::Div)
                } else {
                    false
                }
            } else {
                false
            }
        }
        _ => false,
    };

    if needs_parens {
        Ok(format!("({})", inner))
    } else {
        Ok(inner)
    }
}

fn precedence(op: BinOp) -> u8 {
    match op {
        BinOp::Add | BinOp::Sub => 1,
        BinOp::Mul | BinOp::Div => 2,
        BinOp::Pow => 3,
    }
}

fn format_float(n: f32) -> String {
    if n.fract() == 0.0 && n.abs() < 1e10 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

// ============================================================================
// Execution via mlua
// ============================================================================

/// Compiles and evaluates an expression with mlua.
pub fn eval_lua(ast: &Ast, vars: &HashMap<String, f32>) -> Result<f32, EvalError> {
    let lua = mlua::Lua::new();
    let globals = lua.globals();

    for (name, value) in vars {
        globals.set(name.as_str(), *value).map_err(EvalError::Lua)?;
    }

    let code = emit(ast).map_err(EvalError::Emit)?;
    lua.load(format!("return {}", code))
        .eval()
        .map_err(EvalError::Lua)
}

/// Error during Lua evaluation.
#[derive(Debug)]
pub enum EvalError {
    /// Emission error.
    Emit(LuaError),
    /// Lua runtime error.
    Lua(mlua::Error),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Emit(e) => write!(f, "emit error: {e}"),
            EvalError::Lua(e) => write!(f, "lua error: {e}"),
        }
    }
}

impl std::error::Error for EvalError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        emit_lua(expr.ast()).unwrap().code
    }

    fn eval(input: &str, vars: &[(&str, f32)]) -> f32 {
        let expr = Expr::parse(input).unwrap();
        let var_map: HashMap<String, f32> = vars.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        eval_lua(expr.ast(), &var_map).unwrap()
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval("atan2(1, 1)", &[]) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    }

    #[test]
    fn test_hyperbolic() {
        assert!(eval("sinh(0)", &[]).abs() < 0.001);
        assert!((eval("cosh(0)", &[]) - 1.0).abs() < 0.001);
        assert!(eval("tanh(0)", &[]).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[]) - 1.0).abs() < 0.001);
        assert!((eval("exp2(3)", &[]) - 8.0).abs() < 0.001);
        assert!(eval("ln(1)", &[]).abs() < 0.001);
        assert!((eval("log2(8)", &[]) - 3.0).abs() < 0.001);
        assert!((eval("log10(100)", &[]) - 2.0).abs() < 0.001);
        assert!((eval("pow(2, 3)", &[]) - 8.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[]) - 4.0).abs() < 0.001);
        assert!((eval("inversesqrt(4)", &[]) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(-5)", &[]), 5.0);
        assert_eq!(eval("sign(-3)", &[]), -1.0);
        assert_eq!(eval("sign(3)", &[]), 1.0);
        assert_eq!(eval("sign(0)", &[]), 0.0);
        assert_eq!(eval("floor(3.7)", &[]), 3.0);
        assert_eq!(eval("ceil(3.2)", &[]), 4.0);
        assert_eq!(eval("round(3.5)", &[]), 4.0);
        assert_eq!(eval("trunc(3.7)", &[]), 3.0);
        assert_eq!(eval("trunc(-3.7)", &[]), -3.0);
        assert!((eval("fract(3.7)", &[]) - 0.7).abs() < 0.001);
        assert_eq!(eval("min(3, 7)", &[]), 3.0);
        assert_eq!(eval("max(3, 7)", &[]), 7.0);
        assert_eq!(eval("clamp(5, 0, 3)", &[]), 3.0);
        assert_eq!(eval("saturate(1.5)", &[]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(eval("lerp(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("mix(0, 10, 0.5)", &[]), 5.0);
        assert_eq!(eval("step(0.5, 0.3)", &[]), 0.0);
        assert_eq!(eval("step(0.5, 0.7)", &[]), 1.0);
        assert!((eval("smoothstep(0, 1, 0.5)", &[]) - 0.5).abs() < 0.1);
        assert_eq!(eval("inverse_lerp(0, 10, 5)", &[]), 0.5);
    }

    #[test]
    fn test_remap() {
        assert_eq!(eval("remap(5, 0, 10, 0, 100)", &[]), 50.0);
    }

    #[test]
    fn test_code_generation() {
        assert_eq!(compile("sin(x)"), "math.sin(x)");
        assert!(compile("pi()").contains("math.pi"));
        assert!(compile("pow(x, 2)").contains("^"));
    }

    #[test]
    fn test_operators() {
        assert_eq!(compile("x + y"), "x + y");
        assert_eq!(compile("x ^ 2"), "x ^ 2.0");
        assert_eq!(compile("-x"), "-x");
    }
}
