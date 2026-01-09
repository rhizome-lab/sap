//! WGSL code generation for scalar expressions.
//!
//! Compiles expression ASTs to WGSL shader code.

use rhizome_sap_core::{Ast, BinOp, UnaryOp};

// ============================================================================
// Errors
// ============================================================================

/// WGSL emission error.
#[derive(Debug, Clone, PartialEq)]
pub enum WgslError {
    /// Unknown function.
    UnknownFunction(String),
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
        }
    }
}

impl std::error::Error for WgslError {}

// ============================================================================
// WGSL expression output
// ============================================================================

/// Result of emitting WGSL code.
#[derive(Debug, Clone)]
pub struct WgslExpr {
    /// The WGSL expression string.
    pub code: String,
}

impl WgslExpr {
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }
}

// ============================================================================
// Function name mapping
// ============================================================================

/// Returns the WGSL equivalent for a scalar function name.
fn wgsl_func_name(name: &str) -> Option<WgslFunc> {
    Some(match name {
        // Constants
        "pi" => WgslFunc::Const(std::f64::consts::PI),
        "e" => WgslFunc::Const(std::f64::consts::E),
        "tau" => WgslFunc::Const(std::f64::consts::TAU),

        // Trig - direct mapping
        "sin" => WgslFunc::Direct("sin"),
        "cos" => WgslFunc::Direct("cos"),
        "tan" => WgslFunc::Direct("tan"),
        "asin" => WgslFunc::Direct("asin"),
        "acos" => WgslFunc::Direct("acos"),
        "atan" => WgslFunc::Direct("atan"),
        "atan2" => WgslFunc::Direct("atan2"),
        "sinh" => WgslFunc::Direct("sinh"),
        "cosh" => WgslFunc::Direct("cosh"),
        "tanh" => WgslFunc::Direct("tanh"),

        // Exp/log
        "exp" => WgslFunc::Direct("exp"),
        "exp2" => WgslFunc::Direct("exp2"),
        "log2" => WgslFunc::Direct("log2"),
        "pow" => WgslFunc::Direct("pow"),
        "sqrt" => WgslFunc::Direct("sqrt"),
        "log" | "ln" => WgslFunc::Direct("log"),
        "log10" => WgslFunc::Log10,
        "inversesqrt" => WgslFunc::Direct("inverseSqrt"),

        // Common math
        "abs" => WgslFunc::Direct("abs"),
        "sign" => WgslFunc::Direct("sign"),
        "floor" => WgslFunc::Direct("floor"),
        "ceil" => WgslFunc::Direct("ceil"),
        "round" => WgslFunc::Direct("round"),
        "trunc" => WgslFunc::Direct("trunc"),
        "fract" => WgslFunc::Direct("fract"),
        "min" => WgslFunc::Direct("min"),
        "max" => WgslFunc::Direct("max"),
        "clamp" => WgslFunc::Direct("clamp"),
        "saturate" => WgslFunc::Direct("saturate"),

        // Interpolation
        "lerp" | "mix" => WgslFunc::Direct("mix"),
        "step" => WgslFunc::Direct("step"),
        "smoothstep" => WgslFunc::Direct("smoothstep"),
        "inverse_lerp" => WgslFunc::InverseLerp,
        "remap" => WgslFunc::Remap,

        _ => return None,
    })
}

enum WgslFunc {
    /// Constant value
    Const(f64),
    /// Direct function name mapping
    Direct(&'static str),
    /// log10(x) = log(x) / log(10)
    Log10,
    /// inverse_lerp(a, b, v) = (v - a) / (b - a)
    InverseLerp,
    /// remap(x, in_lo, in_hi, out_lo, out_hi)
    Remap,
}

// ============================================================================
// Code generation
// ============================================================================

/// Emit WGSL code for an AST.
pub fn emit_wgsl(ast: &Ast) -> Result<WgslExpr, WgslError> {
    Ok(WgslExpr::new(emit(ast)?))
}

/// Generate a complete WGSL function.
pub fn emit_wgsl_fn(name: &str, ast: &Ast, params: &[&str]) -> Result<String, WgslError> {
    let param_list: String = params
        .iter()
        .map(|p| format!("{}: f32", p))
        .collect::<Vec<_>>()
        .join(", ");

    let body = emit(ast)?;
    Ok(format!(
        "fn {}({}) -> f32 {{\n    return {};\n}}",
        name, param_list, body
    ))
}

fn emit(ast: &Ast) -> Result<String, WgslError> {
    match ast {
        Ast::Num(n) => Ok(format_float(*n)),
        Ast::Var(name) => Ok(name.clone()),
        Ast::BinOp(op, left, right) => {
            let l = emit_with_parens(left, Some(*op), true)?;
            let r = emit_with_parens(right, Some(*op), false)?;
            match op {
                BinOp::Add => Ok(format!("{} + {}", l, r)),
                BinOp::Sub => Ok(format!("{} - {}", l, r)),
                BinOp::Mul => Ok(format!("{} * {}", l, r)),
                BinOp::Div => Ok(format!("{} / {}", l, r)),
                BinOp::Pow => Ok(format!("pow({}, {})", emit(left)?, emit(right)?)),
            }
        }
        Ast::UnaryOp(op, inner) => {
            let inner_str = emit_with_parens(inner, None, false)?;
            match op {
                UnaryOp::Neg => Ok(format!("-{}", inner_str)),
            }
        }
        Ast::Call(name, args) => {
            let func =
                wgsl_func_name(name).ok_or_else(|| WgslError::UnknownFunction(name.clone()))?;

            let args_str: Vec<String> = args.iter().map(|a| emit(a)).collect::<Result<_, _>>()?;

            Ok(match func {
                WgslFunc::Const(v) => format!("{:.10}", v),
                WgslFunc::Direct(wgsl_name) => format!("{}({})", wgsl_name, args_str.join(", ")),
                WgslFunc::Log10 => {
                    format!(
                        "(log({}) / log(10.0))",
                        args_str.first().unwrap_or(&"0.0".to_string())
                    )
                }
                WgslFunc::InverseLerp => {
                    let a = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let b = args_str.get(1).map(|s| s.as_str()).unwrap_or("1.0");
                    let v = args_str.get(2).map(|s| s.as_str()).unwrap_or("0.0");
                    format!("(({v} - {a}) / ({b} - {a}))")
                }
                WgslFunc::Remap => {
                    let x = args_str.first().map(|s| s.as_str()).unwrap_or("0.0");
                    let in_lo = args_str.get(1).map(|s| s.as_str()).unwrap_or("0.0");
                    let in_hi = args_str.get(2).map(|s| s.as_str()).unwrap_or("1.0");
                    let out_lo = args_str.get(3).map(|s| s.as_str()).unwrap_or("0.0");
                    let out_hi = args_str.get(4).map(|s| s.as_str()).unwrap_or("1.0");
                    format!(
                        "({out_lo} + ({out_hi} - {out_lo}) * (({x} - {in_lo}) / ({in_hi} - {in_lo})))"
                    )
                }
            })
        }
    }
}

fn emit_with_parens(
    ast: &Ast,
    parent_op: Option<BinOp>,
    is_left: bool,
) -> Result<String, WgslError> {
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
    if n.fract() == 0.0 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn compile(input: &str) -> String {
        let expr = Expr::parse(input).unwrap();
        emit_wgsl(expr.ast()).unwrap().code
    }

    #[test]
    fn test_constants() {
        assert!(compile("pi()").contains("3.14159"));
        assert!(compile("e()").contains("2.71828"));
        assert!(compile("tau()").contains("6.28318"));
    }

    #[test]
    fn test_trig() {
        assert_eq!(compile("sin(x)"), "sin(x)");
        assert_eq!(compile("cos(x)"), "cos(x)");
        assert_eq!(compile("atan2(y, x)"), "atan2(y, x)");
    }

    #[test]
    fn test_exp_log() {
        assert_eq!(compile("exp(x)"), "exp(x)");
        assert_eq!(compile("ln(x)"), "log(x)");
        assert_eq!(compile("log10(x)"), "(log(x) / log(10.0))");
        assert_eq!(compile("pow(x, 2)"), "pow(x, 2.0)");
        assert_eq!(compile("sqrt(x)"), "sqrt(x)");
        assert_eq!(compile("inversesqrt(x)"), "inverseSqrt(x)");
    }

    #[test]
    fn test_common() {
        assert_eq!(compile("abs(x)"), "abs(x)");
        assert_eq!(compile("floor(x)"), "floor(x)");
        assert_eq!(compile("clamp(x, 0, 1)"), "clamp(x, 0.0, 1.0)");
        assert_eq!(compile("saturate(x)"), "saturate(x)");
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(compile("lerp(a, b, t)"), "mix(a, b, t)");
        assert_eq!(compile("mix(a, b, t)"), "mix(a, b, t)");
        assert_eq!(compile("step(e, x)"), "step(e, x)");
        assert_eq!(compile("smoothstep(0, 1, x)"), "smoothstep(0.0, 1.0, x)");
    }

    #[test]
    fn test_inverse_lerp() {
        assert_eq!(
            compile("inverse_lerp(0, 10, x)"),
            "((x - 0.0) / (10.0 - 0.0))"
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(compile("x + y"), "x + y");
        assert_eq!(compile("x * y + z"), "x * y + z");
        assert_eq!(compile("(x + y) * z"), "(x + y) * z");
        assert_eq!(compile("-x"), "-x");
        assert_eq!(compile("x ^ 2"), "pow(x, 2.0)");
    }

    #[test]
    fn test_wgsl_fn() {
        let expr = Expr::parse("x + y").unwrap();
        let code = emit_wgsl_fn("add", expr.ast(), &["x", "y"]).unwrap();
        assert!(code.contains("fn add(x: f32, y: f32) -> f32"));
        assert!(code.contains("return x + y"));
    }
}
