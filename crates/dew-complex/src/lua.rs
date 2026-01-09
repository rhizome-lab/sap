//! Lua code generation for complex expressions.
//!
//! Complex numbers are represented as tables {real, imag}.

use crate::Type;
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

/// Error during Lua code generation.
#[derive(Debug, Clone, PartialEq)]
pub enum LuaError {
    UnknownVariable(String),
    UnknownFunction(String),
    TypeMismatch {
        op: &'static str,
        left: Type,
        right: Type,
    },
}

impl std::fmt::Display for LuaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LuaError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            LuaError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            LuaError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
        }
    }
}

impl std::error::Error for LuaError {}

/// Result of Lua emission: code string and its type.
pub struct LuaExpr {
    pub code: String,
    pub typ: Type,
}

/// Emit Lua code for an AST with type propagation.
pub fn emit_lua(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<LuaExpr, LuaError> {
    match ast {
        Ast::Num(n) => Ok(LuaExpr {
            code: format_float(*n),
            typ: Type::Scalar,
        }),

        Ast::Var(name) => {
            let typ = var_types
                .get(name)
                .copied()
                .ok_or_else(|| LuaError::UnknownVariable(name.clone()))?;
            Ok(LuaExpr {
                code: name.clone(),
                typ,
            })
        }

        Ast::BinOp(op, left, right) => {
            let left_expr = emit_lua(left, var_types)?;
            let right_expr = emit_lua(right, var_types)?;
            emit_binop(*op, left_expr, right_expr)
        }

        Ast::UnaryOp(op, inner) => {
            let inner_expr = emit_lua(inner, var_types)?;
            emit_unaryop(*op, inner_expr)
        }

        Ast::Call(name, args) => {
            let arg_exprs: Vec<LuaExpr> = args
                .iter()
                .map(|a| emit_lua(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }
    }
}

fn format_float(n: f32) -> String {
    if n.fract() == 0.0 && n.abs() < 1e10 {
        format!("{:.1}", n)
    } else {
        format!("{}", n)
    }
}

fn emit_binop(op: BinOp, left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match op {
        BinOp::Add => emit_add(left, right),
        BinOp::Sub => emit_sub(left, right),
        BinOp::Mul => emit_mul(left, right),
        BinOp::Div => emit_div(left, right),
        BinOp::Pow => emit_pow(left, right),
    }
}

fn emit_add(left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} + {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Complex, Type::Complex) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(LuaError::TypeMismatch {
            op: "+",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_sub(left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} - {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Complex, Type::Complex) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Complex,
        }),
        _ => Err(LuaError::TypeMismatch {
            op: "-",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_mul(left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Scalar * Complex
        (Type::Scalar, Type::Complex) => Ok(LuaExpr {
            code: format!(
                "{{{} * {}[1], {} * {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Complex,
        }),
        (Type::Complex, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] * {}, {}[2] * {}}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Complex,
        }),
        // Complex * Complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        (Type::Complex, Type::Complex) => Ok(LuaExpr {
            code: format!(
                "{{{}[1]*{}[1] - {}[2]*{}[2], {}[1]*{}[2] + {}[2]*{}[1]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Complex,
        }),
    }
}

fn emit_div(left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        // Complex / Scalar
        (Type::Complex, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] / {}, {}[2] / {}}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Complex,
        }),
        // Complex / Complex
        (Type::Complex, Type::Complex) => {
            let l = &left.code;
            let r = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "(function() local _d = {r}[1]*{r}[1] + {r}[2]*{r}[2]; return {{({l}[1]*{r}[1] + {l}[2]*{r}[2])/_d, ({l}[2]*{r}[1] - {l}[1]*{r}[2])/_d}} end)()"
                ),
                typ: Type::Complex,
            })
        }
        _ => Err(LuaError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(base: LuaExpr, exp: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (base.typ, exp.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} ^ {})", base.code, exp.code),
            typ: Type::Scalar,
        }),
        // Complex^Scalar using polar form
        (Type::Complex, Type::Scalar) => {
            let z = &base.code;
            let n = &exp.code;
            Ok(LuaExpr {
                code: format!(
                    "(function() local _r = math.sqrt({z}[1]*{z}[1] + {z}[2]*{z}[2])^{n}; local _t = math.atan2({z}[2], {z}[1])*{n}; return {{_r*math.cos(_t), _r*math.sin(_t)}} end)()"
                ),
                typ: Type::Complex,
            })
        }
        _ => Err(LuaError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: LuaExpr) -> Result<LuaExpr, LuaError> {
    match op {
        UnaryOp::Neg => match inner.typ {
            Type::Scalar => Ok(LuaExpr {
                code: format!("(-{})", inner.code),
                typ: Type::Scalar,
            }),
            Type::Complex => Ok(LuaExpr {
                code: format!("{{-{}[1], -{}[2]}}", inner.code, inner.code),
                typ: Type::Complex,
            }),
        },
    }
}

fn emit_function_call(name: &str, args: Vec<LuaExpr>) -> Result<LuaExpr, LuaError> {
    match name {
        "re" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!("{}[1]", args[0].code),
                typ: Type::Scalar,
            })
        }

        "im" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!("{}[2]", args[0].code),
                typ: Type::Scalar,
            })
        }

        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!("{{{}[1], -{}[2]}}", args[0].code, args[0].code),
                typ: Type::Complex,
            })
        }

        "abs" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(LuaExpr {
                    code: format!("math.abs({})", args[0].code),
                    typ: Type::Scalar,
                }),
                Type::Complex => Ok(LuaExpr {
                    code: format!(
                        "math.sqrt({}[1]*{}[1] + {}[2]*{}[2])",
                        args[0].code, args[0].code, args[0].code, args[0].code
                    ),
                    typ: Type::Scalar,
                }),
            }
        }

        "arg" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!("math.atan2({}[2], {}[1])", args[0].code, args[0].code),
                typ: Type::Scalar,
            })
        }

        "norm" => {
            if args.len() != 1 || args[0].typ != Type::Complex {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!(
                    "({}[1]*{}[1] + {}[2]*{}[2])",
                    args[0].code, args[0].code, args[0].code, args[0].code
                ),
                typ: Type::Scalar,
            })
        }

        "exp" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(LuaExpr {
                    code: format!("math.exp({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // exp(a+bi) = e^a * (cos(b) + i*sin(b))
                Type::Complex => {
                    let z = &args[0].code;
                    Ok(LuaExpr {
                        code: format!(
                            "(function() local _e = math.exp({z}[1]); return {{_e*math.cos({z}[2]), _e*math.sin({z}[2])}} end)()"
                        ),
                        typ: Type::Complex,
                    })
                }
            }
        }

        "log" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(LuaExpr {
                    code: format!("math.log({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // log(z) = log|z| + i*arg(z)
                Type::Complex => {
                    let z = &args[0].code;
                    Ok(LuaExpr {
                        code: format!(
                            "{{math.log(math.sqrt({z}[1]*{z}[1] + {z}[2]*{z}[2])), math.atan2({z}[2], {z}[1])}}"
                        ),
                        typ: Type::Complex,
                    })
                }
            }
        }

        "sqrt" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            match args[0].typ {
                Type::Scalar => Ok(LuaExpr {
                    code: format!("math.sqrt({})", args[0].code),
                    typ: Type::Scalar,
                }),
                // sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
                Type::Complex => {
                    let z = &args[0].code;
                    Ok(LuaExpr {
                        code: format!(
                            "(function() local _r = math.sqrt(math.sqrt({z}[1]*{z}[1] + {z}[2]*{z}[2])); local _t = math.atan2({z}[2], {z}[1])*0.5; return {{_r*math.cos(_t), _r*math.sin(_t)}} end)()"
                        ),
                        typ: Type::Complex,
                    })
                }
            }
        }

        "polar" => {
            if args.len() != 2 || args[0].typ != Type::Scalar || args[1].typ != Type::Scalar {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            Ok(LuaExpr {
                code: format!(
                    "{{{}*math.cos({}), {}*math.sin({})}}",
                    args[0].code, args[1].code, args[0].code, args[1].code
                ),
                typ: Type::Complex,
            })
        }

        _ => Err(LuaError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<LuaExpr, LuaError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_lua(expr.ast(), &types)
    }

    #[test]
    fn test_complex_add() {
        let result = emit("a + b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
        assert!(result.code.contains("[1]") && result.code.contains("[2]"));
    }

    #[test]
    fn test_complex_mul() {
        let result = emit("a * b", &[("a", Type::Complex), ("b", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
    }

    #[test]
    fn test_re() {
        let result = emit("re(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("[1]"));
    }

    #[test]
    fn test_abs() {
        let result = emit("abs(z)", &[("z", Type::Complex)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("math.sqrt"));
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
        assert!(result.code.contains("math.cos") && result.code.contains("math.sin"));
    }

    #[test]
    fn test_polar() {
        let result = emit("polar(r, t)", &[("r", Type::Scalar), ("t", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Complex);
    }
}
