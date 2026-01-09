//! WGSL code generation for complex expressions.
//!
//! Complex numbers are represented as vec2<f32> where x=real, y=imag.

use crate::Type;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during WGSL code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum WgslError {
    UnknownVariable(String),
    UnknownFunction(String),
    TypeMismatch {
        op: &'static str,
        left: Type,
        right: Type,
    },
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            WgslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            WgslError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
        }
    }
}

impl std::error::Error for WgslError {}

/// Convert a Type to its WGSL representation.
pub fn type_to_wgsl(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Complex => "vec2<f32>",
    }
}

/// Result of WGSL emission: code string and its type.
pub struct WgslExpr {
    pub code: String,
    pub typ: Type,
}

/// Emit WGSL code for an AST with type propagation.
pub fn emit_wgsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<WgslExpr, WgslError> {
    match ast {
        Ast::Num(n) => Ok(WgslExpr {
            code: format!("{n:.10}"),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| WgslError::UnknownVariable(name.clone()))?;
            Ok(WgslExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_wgsl(left, var_types)?;
            let right_expr = emit_wgsl(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_wgsl(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<WgslExpr> = args
                .iter()
                .map(|a| emit_wgsl(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }
    }
}

fn emit_binop(op: BinOp, left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match op {
        BinOp::Add => emit_add(left, right),
        BinOp::Sub => emit_sub(left, right),
        BinOp::Mul => emit_mul(left, right),
        BinOp::Div => emit_div(left, right),
        BinOp::Pow => emit_pow(left, right),
    }
}

fn emit_add(left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(WgslExpr {
            code: format!("({} + {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(WgslError::TypeMismatch {
            op: "+",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_sub(left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) | (Type::Complex, Type::Complex) => Ok(WgslExpr {
            code: format!("({} - {})", left.code, right.code),
            typ: left.typ,
        }),
        _ => Err(WgslError::TypeMismatch {
            op: "-",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_mul(left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Scalar * Complex
        (Type::Scalar, Type::Complex) => Ok(WgslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Complex,
        }),
        // Complex * Complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        (Type::Complex, Type::Complex) => Ok(WgslExpr {
            code: format!(
                "vec2<f32>({l}.x * {r}.x - {l}.y * {r}.y, {l}.x * {r}.y + {l}.y * {r}.x)",
                l = left.code,
                r = right.code
            ),
            typ: Type::Complex,
        }),
    }
}

fn emit_div(left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Complex / Scalar
        (Type::Complex, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Complex,
        }),
        // Complex / Complex: multiply by conjugate
        // (a+bi)/(c+di) = (a+bi)(c-di) / (c²+d²)
        (Type::Complex, Type::Complex) => Ok(WgslExpr {
            code: format!(
                "(vec2<f32>({l}.x * {r}.x + {l}.y * {r}.y, {l}.y * {r}.x - {l}.x * {r}.y) / ({r}.x * {r}.x + {r}.y * {r}.y))",
                l = left.code,
                r = right.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(WgslError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(base: WgslExpr, exp: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (base.typ, exp.typ) {
        (Type::Scalar, Type::Scalar) => Ok(WgslExpr {
            code: format!("pow({}, {})", base.code, exp.code),
            typ: Type::Scalar,
        }),
        // Complex^Scalar using polar form
        (Type::Complex, Type::Scalar) => Ok(WgslExpr {
            code: format!(
                "(func() -> vec2<f32> {{ let _r = pow(length({z}), {n}); let _t = atan2({z}.y, {z}.x) * {n}; return vec2<f32>(_r * cos(_t), _r * sin(_t)); }})()",
                z = base.code,
                n = exp.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(WgslError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: WgslExpr) -> Result<WgslExpr, WgslError> {
    match op {
        UnaryOp::Neg => Ok(WgslExpr {
            code: format!("(-{})", inner.code),
            typ: inner.typ,
        }),
    }
}

fn emit_function_call(name: &str, args: Vec<WgslExpr>) -> Result<WgslExpr, WgslError> {
    match name {
        "re" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.x", args[0].code),
                typ: Type::Scalar,
            })
        }

        "im" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("{}.y", args[0].code),
                typ: Type::Scalar,
            })
        }

        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("vec2<f32>({}.x, -{}.y)", args[0].code, args[0].code),
                typ: Type::Complex,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(WgslExpr {
                    code: format!("abs({})", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(WgslExpr {
                    code: format!("length({})", args[0].code),
                    typ: Type::Scalar,
                }),
            }
        }

        "arg" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("atan2({}.y, {}.x)", args[0].code, args[0].code),
                typ: Type::Scalar,
            })
        }

        "norm" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("dot({z}, {z})", z = args[0].code),
                typ: Type::Scalar,
            })
        }

        "exp" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(WgslExpr {
                    code: format!("exp({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // exp(a+bi) = e^a * (cos(b) + i*sin(b))
                Type::Complex => Ok(WgslExpr {
                    code: format!(
                        "(exp({z}.x) * vec2<f32>(cos({z}.y), sin({z}.y)))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "log" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(WgslExpr {
                    code: format!("log({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // log(z) = log|z| + i*arg(z)
                Type::Complex => Ok(WgslExpr {
                    code: format!(
                        "vec2<f32>(log(length({z})), atan2({z}.y, {z}.x))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(WgslExpr {
                    code: format!("sqrt({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
                Type::Complex => Ok(WgslExpr {
                    code: format!(
                        "(sqrt(length({z})) * vec2<f32>(cos(atan2({z}.y, {z}.x) * 0.5), sin(atan2({z}.y, {z}.x) * 0.5)))",
                        z = args[0].code
                    ),
                    typ: Type::Complex,
                }),
            }
        }

        "polar" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!(
                    "({r} * vec2<f32>(cos({t}), sin({t})))",
                    r = args[0].code,
                    t = args[1].code
                ),
                typ: Type::Complex,
            })
        }

        _ => Err(WgslError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<WgslExpr, WgslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_wgsl(expr.ast(), &types)
    }

    #[test]
    fn test_complex_add() {
        let result = emit("a + b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("+"));
    }

    #[test]
    fn test_complex_mul() {
        let result = emit("a * b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        // Should contain the complex multiplication formula
        assert!(result.code.contains(".x") && result.code.contains(".y"));
    }

    #[test]
    fn test_re() {
        let result = emit("re(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains(".x"));
    }

    #[test]
    fn test_abs() {
        let result = emit("abs(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("length"));
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
    }

    #[test]
    fn test_exp() {
        let result = emit("exp(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("cos") && result.code.contains("sin"));
    }
}
