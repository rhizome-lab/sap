//! Lua code generation for complex expressions.
//!
//! Complex numbers are represented as tables {real, imag}.

use crate::Type;
use rhizome_dew_cond::lua as cond;
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
    /// Conditionals require scalar types.
    UnsupportedTypeForConditional(Type),
}

impl std::fmt::Display for LuaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LuaError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            LuaError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            LuaError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            LuaError::UnsupportedTypeForConditional(t) => {
                write!(f, "conditionals require scalar type, got {t}")
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

        Ast::Compare(op, left, right) => {
            let left_expr = emit_lua(left, var_types)?;
            let right_expr = emit_lua(right, var_types)?;
            // Comparisons only supported for scalars
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(Type::Complex));
            }
            let bool_expr = cond::emit_compare(*op, &left_expr.code, &right_expr.code);
            Ok(LuaExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::And(left, right) => {
            let left_expr = emit_lua(left, var_types)?;
            let right_expr = emit_lua(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(Type::Complex));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_and(&l_bool, &r_bool);
            Ok(LuaExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::Or(left, right) => {
            let left_expr = emit_lua(left, var_types)?;
            let right_expr = emit_lua(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(Type::Complex));
            }
            let l_bool = cond::scalar_to_bool(&left_expr.code);
            let r_bool = cond::scalar_to_bool(&right_expr.code);
            let bool_expr = cond::emit_or(&l_bool, &r_bool);
            Ok(LuaExpr {
                code: cond::bool_to_scalar(&bool_expr),
                typ: Type::Scalar,
            })
        }

        Ast::If(cond_ast, then_ast, else_ast) => {
            let cond_expr = emit_lua(cond_ast, var_types)?;
            let then_expr = emit_lua(then_ast, var_types)?;
            let else_expr = emit_lua(else_ast, var_types)?;
            if cond_expr.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(cond_expr.typ));
            }
            // then and else can be any matching types
            if then_expr.typ != else_expr.typ {
                return Err(LuaError::TypeMismatch {
                    op: "if/else",
                    left: then_expr.typ,
                    right: else_expr.typ,
                });
            }
            let cond_bool = cond::scalar_to_bool(&cond_expr.code);
            Ok(LuaExpr {
                code: cond::emit_if(&cond_bool, &then_expr.code, &else_expr.code),
                typ: then_expr.typ,
            })
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
        UnaryOp::Not => {
            if inner.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(inner.typ));
            }
            let bool_expr = cond::scalar_to_bool(&inner.code);
            Ok(LuaExpr {
                code: cond::bool_to_scalar(&cond::emit_not(&bool_expr)),
                typ: Type::Scalar,
            })
        }
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

// ============================================================================
// Execution via mlua (requires "lua" feature)
// ============================================================================

#[cfg(feature = "lua")]
use crate::Value;

#[cfg(feature = "lua")]
/// Error during Lua evaluation.
#[derive(Debug)]
pub enum EvalError {
    /// Emission error.
    Emit(LuaError),
    /// Lua runtime error.
    Lua(mlua::Error),
    /// Type conversion error.
    TypeConversion(String),
}

#[cfg(feature = "lua")]
impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Emit(e) => write!(f, "emit error: {e}"),
            EvalError::Lua(e) => write!(f, "lua error: {e}"),
            EvalError::TypeConversion(e) => write!(f, "type conversion error: {e}"),
        }
    }
}

#[cfg(feature = "lua")]
impl std::error::Error for EvalError {}

#[cfg(feature = "lua")]
/// Compiles and evaluates an expression with mlua.
pub fn eval_lua<T: num_traits::Float + mlua::IntoLua + mlua::FromLua>(
    ast: &rhizome_dew_core::Ast,
    vars: &HashMap<String, Value<T>>,
) -> Result<Value<T>, EvalError> {
    use mlua::Value as LuaValue;

    let lua = mlua::Lua::new();
    let globals = lua.globals();

    // Build var_types map and set variables in Lua
    let mut var_types = HashMap::new();
    for (name, value) in vars {
        var_types.insert(name.clone(), value.typ());
        set_lua_var(&lua, &globals, name, value).map_err(EvalError::Lua)?;
    }

    // Emit Lua code
    let expr = emit_lua(ast, &var_types).map_err(EvalError::Emit)?;
    let result_type = expr.typ;

    // Execute
    let lua_result: LuaValue = lua
        .load(format!("return {}", expr.code))
        .eval()
        .map_err(EvalError::Lua)?;

    // Convert back to Value<T>
    lua_to_value(&lua_result, result_type)
}

#[cfg(feature = "lua")]
fn set_lua_var<T: num_traits::Float + mlua::IntoLua>(
    lua: &mlua::Lua,
    globals: &mlua::Table,
    name: &str,
    value: &Value<T>,
) -> Result<(), mlua::Error> {
    match value {
        Value::Scalar(s) => {
            globals.set(name, (*s).into_lua(lua)?)?;
        }
        Value::Complex(c) => {
            let table = lua.create_table()?;
            table.set(1, c[0].into_lua(lua)?)?;
            table.set(2, c[1].into_lua(lua)?)?;
            globals.set(name, table)?;
        }
    }
    Ok(())
}

#[cfg(feature = "lua")]
fn lua_to_value<T: num_traits::Float + mlua::FromLua>(
    lua_val: &mlua::Value,
    typ: Type,
) -> Result<Value<T>, EvalError> {
    use mlua::Value as LuaValue;

    match typ {
        Type::Scalar => {
            if let LuaValue::Number(n) = lua_val {
                Ok(Value::Scalar(T::from(*n).ok_or_else(|| {
                    EvalError::TypeConversion("failed to convert number".into())
                })?))
            } else if let LuaValue::Integer(n) = lua_val {
                Ok(Value::Scalar(T::from(*n).ok_or_else(|| {
                    EvalError::TypeConversion("failed to convert integer".into())
                })?))
            } else {
                Err(EvalError::TypeConversion(format!(
                    "expected number, got {:?}",
                    lua_val
                )))
            }
        }
        Type::Complex => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Complex".into()))?;
            let re: f64 = table.get(1).map_err(EvalError::Lua)?;
            let im: f64 = table.get(2).map_err(EvalError::Lua)?;
            Ok(Value::Complex([
                T::from(re).ok_or_else(|| EvalError::TypeConversion("complex[1]".into()))?,
                T::from(im).ok_or_else(|| EvalError::TypeConversion("complex[2]".into()))?,
            ]))
        }
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
