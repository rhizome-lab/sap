//! Lua code generation for linalg expressions.
//!
//! Emits Lua code with vectors as tables and matrices as flat tables (column-major).
//!
//! Vector: `{x, y}` for vec2, `{x, y, z}` for vec3
//! Matrix: `{c0r0, c0r1, c1r0, c1r1}` for mat2 (column-major)

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
    UnsupportedType(Type),
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
            LuaError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
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

        Ast::Call(name, args) => {
            let arg_exprs: Vec<LuaExpr> = args
                .iter()
                .map(|a| emit_lua(a, var_types))
                .collect::<Result<_, _>>()?;
            emit_function_call(name, arg_exprs)
        }

        Ast::Compare(op, left, right) => {
            let left_expr = emit_lua(left, var_types)?;
            let right_expr = emit_lua(right, var_types)?;
            if left_expr.typ != Type::Scalar || right_expr.typ != Type::Scalar {
                return Err(LuaError::UnsupportedTypeForConditional(left_expr.typ));
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
                return Err(LuaError::UnsupportedTypeForConditional(left_expr.typ));
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
                return Err(LuaError::UnsupportedTypeForConditional(left_expr.typ));
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
    let result_type = infer_binop_type(op, left.typ, right.typ)?;

    match op {
        BinOp::Add => emit_add(left, right, result_type),
        BinOp::Sub => emit_sub(left, right, result_type),
        BinOp::Mul => emit_mul(left, right, result_type),
        BinOp::Div => emit_div(left, right, result_type),
        BinOp::Pow => emit_pow(left, right),
    }
}

fn emit_add(left: LuaExpr, right: LuaExpr, result_type: Type) -> Result<LuaExpr, LuaError> {
    match result_type {
        Type::Scalar => Ok(LuaExpr {
            code: format!("({} + {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        Type::Vec2 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        Type::Vec3 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2], {}[3] + {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        #[cfg(feature = "4d")]
        Type::Vec4 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2], {}[3] + {}[3], {}[4] + {}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec4,
        }),
        Type::Mat2 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2], {}[3] + {}[3], {}[4] + {}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Mat2,
        }),
        #[cfg(feature = "3d")]
        Type::Mat3 => {
            let parts: Vec<String> = (1..=9)
                .map(|i| format!("{}[{}] + {}[{}]", left.code, i, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat3,
            })
        }
        #[cfg(feature = "4d")]
        Type::Mat4 => {
            let parts: Vec<String> = (1..=16)
                .map(|i| format!("{}[{}] + {}[{}]", left.code, i, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat4,
            })
        }
    }
}

fn emit_sub(left: LuaExpr, right: LuaExpr, result_type: Type) -> Result<LuaExpr, LuaError> {
    match result_type {
        Type::Scalar => Ok(LuaExpr {
            code: format!("({} - {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        Type::Vec2 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        Type::Vec3 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2], {}[3] - {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        #[cfg(feature = "4d")]
        Type::Vec4 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2], {}[3] - {}[3], {}[4] - {}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec4,
        }),
        Type::Mat2 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2], {}[3] - {}[3], {}[4] - {}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Mat2,
        }),
        #[cfg(feature = "3d")]
        Type::Mat3 => {
            let parts: Vec<String> = (1..=9)
                .map(|i| format!("{}[{}] - {}[{}]", left.code, i, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat3,
            })
        }
        #[cfg(feature = "4d")]
        Type::Mat4 => {
            let parts: Vec<String> = (1..=16)
                .map(|i| format!("{}[{}] - {}[{}]", left.code, i, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat4,
            })
        }
    }
}

fn emit_mul(left: LuaExpr, right: LuaExpr, result_type: Type) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        // Scalar * Scalar
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Scalar,
        }),

        // Vec * Scalar
        (Type::Vec2, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] * {}, {}[2] * {}}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] * {}, {}[2] * {}, {}[3] * {}}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] * {}, {}[2] * {}, {}[3] * {}, {}[4] * {}}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec4,
        }),

        // Scalar * Vec
        (Type::Scalar, Type::Vec2) => Ok(LuaExpr {
            code: format!(
                "{{{} * {}[1], {} * {}[2]}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        (Type::Scalar, Type::Vec3) => Ok(LuaExpr {
            code: format!(
                "{{{} * {}[1], {} * {}[2], {} * {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        #[cfg(feature = "4d")]
        (Type::Scalar, Type::Vec4) => Ok(LuaExpr {
            code: format!(
                "{{{} * {}[1], {} * {}[2], {} * {}[3], {} * {}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec4,
        }),

        // Mat * Scalar
        (Type::Mat2, Type::Scalar) => {
            let parts: Vec<String> = (1..=4)
                .map(|i| format!("{}[{}] * {}", left.code, i, right.code))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat2,
            })
        }
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Scalar) => {
            let parts: Vec<String> = (1..=9)
                .map(|i| format!("{}[{}] * {}", left.code, i, right.code))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat3,
            })
        }
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Scalar) => {
            let parts: Vec<String> = (1..=16)
                .map(|i| format!("{}[{}] * {}", left.code, i, right.code))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat4,
            })
        }

        // Scalar * Mat
        (Type::Scalar, Type::Mat2) => {
            let parts: Vec<String> = (1..=4)
                .map(|i| format!("{} * {}[{}]", left.code, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat2,
            })
        }
        #[cfg(feature = "3d")]
        (Type::Scalar, Type::Mat3) => {
            let parts: Vec<String> = (1..=9)
                .map(|i| format!("{} * {}[{}]", left.code, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat3,
            })
        }
        #[cfg(feature = "4d")]
        (Type::Scalar, Type::Mat4) => {
            let parts: Vec<String> = (1..=16)
                .map(|i| format!("{} * {}[{}]", left.code, right.code, i))
                .collect();
            Ok(LuaExpr {
                code: format!("{{{}}}", parts.join(", ")),
                typ: Type::Mat4,
            })
        }

        // Mat * Vec (column-major: m[col*dim + row + 1])
        // mat2: m[1],m[2] = col0, m[3],m[4] = col1
        (Type::Mat2, Type::Vec2) => Ok(LuaExpr {
            code: format!(
                "{{{}[1]*{}[1] + {}[3]*{}[2], {}[2]*{}[1] + {}[4]*{}[2]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Vec3) => {
            // mat3 column-major: col0=[1,2,3], col1=[4,5,6], col2=[7,8,9]
            Ok(LuaExpr {
                code: format!(
                    "{{{}[1]*{}[1] + {}[4]*{}[2] + {}[7]*{}[3], {}[2]*{}[1] + {}[5]*{}[2] + {}[8]*{}[3], {}[3]*{}[1] + {}[6]*{}[2] + {}[9]*{}[3]}}",
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code,
                    left.code,
                    right.code
                ),
                typ: Type::Vec3,
            })
        }
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Vec4) => {
            // mat4 column-major: col0=[1-4], col1=[5-8], col2=[9-12], col3=[13-16]
            let m = &left.code;
            let v = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "{{{m}[1]*{v}[1] + {m}[5]*{v}[2] + {m}[9]*{v}[3] + {m}[13]*{v}[4], \
                      {m}[2]*{v}[1] + {m}[6]*{v}[2] + {m}[10]*{v}[3] + {m}[14]*{v}[4], \
                      {m}[3]*{v}[1] + {m}[7]*{v}[2] + {m}[11]*{v}[3] + {m}[15]*{v}[4], \
                      {m}[4]*{v}[1] + {m}[8]*{v}[2] + {m}[12]*{v}[3] + {m}[16]*{v}[4]}}"
                ),
                typ: Type::Vec4,
            })
        }

        // Vec * Mat (row vector convention)
        (Type::Vec2, Type::Mat2) => Ok(LuaExpr {
            code: format!(
                "{{{}[1]*{}[1] + {}[2]*{}[2], {}[1]*{}[3] + {}[2]*{}[4]}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Mat3) => {
            let v = &left.code;
            let m = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "{{{v}[1]*{m}[1] + {v}[2]*{m}[2] + {v}[3]*{m}[3], \
                      {v}[1]*{m}[4] + {v}[2]*{m}[5] + {v}[3]*{m}[6], \
                      {v}[1]*{m}[7] + {v}[2]*{m}[8] + {v}[3]*{m}[9]}}"
                ),
                typ: Type::Vec3,
            })
        }
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Mat4) => {
            let v = &left.code;
            let m = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "{{{v}[1]*{m}[1] + {v}[2]*{m}[2] + {v}[3]*{m}[3] + {v}[4]*{m}[4], \
                      {v}[1]*{m}[5] + {v}[2]*{m}[6] + {v}[3]*{m}[7] + {v}[4]*{m}[8], \
                      {v}[1]*{m}[9] + {v}[2]*{m}[10] + {v}[3]*{m}[11] + {v}[4]*{m}[12], \
                      {v}[1]*{m}[13] + {v}[2]*{m}[14] + {v}[3]*{m}[15] + {v}[4]*{m}[16]}}"
                ),
                typ: Type::Vec4,
            })
        }

        // Mat * Mat
        (Type::Mat2, Type::Mat2) => {
            let a = &left.code;
            let b = &right.code;
            // Column-major: result[col][row] = sum(a[k][row] * b[col][k])
            Ok(LuaExpr {
                code: format!(
                    "{{{a}[1]*{b}[1] + {a}[3]*{b}[2], {a}[2]*{b}[1] + {a}[4]*{b}[2], \
                      {a}[1]*{b}[3] + {a}[3]*{b}[4], {a}[2]*{b}[3] + {a}[4]*{b}[4]}}"
                ),
                typ: Type::Mat2,
            })
        }
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Mat3) => {
            // This gets complex - using function call
            let a = &left.code;
            let b = &right.code;
            Ok(LuaExpr {
                code: format!("__linalg_mat3_mul({a}, {b})"),
                typ: Type::Mat3,
            })
        }
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Mat4) => {
            let a = &left.code;
            let b = &right.code;
            Ok(LuaExpr {
                code: format!("__linalg_mat4_mul({a}, {b})"),
                typ: Type::Mat4,
            })
        }

        _ => Ok(LuaExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: result_type,
        }),
    }
}

fn emit_div(left: LuaExpr, right: LuaExpr, result_type: Type) -> Result<LuaExpr, LuaError> {
    match result_type {
        Type::Scalar => Ok(LuaExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        Type::Vec2 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] / {}, {}[2] / {}}}",
                left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec2,
        }),
        #[cfg(feature = "3d")]
        Type::Vec3 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] / {}, {}[2] / {}, {}[3] / {}}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        #[cfg(feature = "4d")]
        Type::Vec4 => Ok(LuaExpr {
            code: format!(
                "{{{}[1] / {}, {}[2] / {}, {}[3] / {}, {}[4] / {}}}",
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code,
                left.code,
                right.code
            ),
            typ: Type::Vec4,
        }),
        _ => Err(LuaError::TypeMismatch {
            op: "/",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_pow(base: LuaExpr, exp: LuaExpr) -> Result<LuaExpr, LuaError> {
    if base.typ != Type::Scalar || exp.typ != Type::Scalar {
        return Err(LuaError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        });
    }
    Ok(LuaExpr {
        code: format!("({} ^ {})", base.code, exp.code),
        typ: Type::Scalar,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, LuaError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            if left == right {
                Ok(left)
            } else {
                Err(LuaError::TypeMismatch {
                    op: if op == BinOp::Add { "+" } else { "-" },
                    left,
                    right,
                })
            }
        }
        BinOp::Mul => infer_mul_type(left, right),
        BinOp::Div => match (left, right) {
            (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),
            (Type::Vec2, Type::Scalar) => Ok(Type::Vec2),
            #[cfg(feature = "3d")]
            (Type::Vec3, Type::Scalar) => Ok(Type::Vec3),
            #[cfg(feature = "4d")]
            (Type::Vec4, Type::Scalar) => Ok(Type::Vec4),
            _ => Err(LuaError::TypeMismatch {
                op: "/",
                left,
                right,
            }),
        },
        BinOp::Pow => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                Err(LuaError::TypeMismatch {
                    op: "^",
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, LuaError> {
    match (left, right) {
        (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),

        (Type::Vec2, Type::Scalar) | (Type::Scalar, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Scalar) | (Type::Scalar, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Scalar) | (Type::Scalar, Type::Vec4) => Ok(Type::Vec4),

        (Type::Mat2, Type::Scalar) | (Type::Scalar, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Scalar) | (Type::Scalar, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Scalar) | (Type::Scalar, Type::Mat4) => Ok(Type::Mat4),

        (Type::Mat2, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Vec4) => Ok(Type::Vec4),

        (Type::Vec2, Type::Mat2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Mat3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Mat4) => Ok(Type::Vec4),

        (Type::Mat2, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Mat4) => Ok(Type::Mat4),

        _ => Err(LuaError::TypeMismatch {
            op: "*",
            left,
            right,
        }),
    }
}

fn emit_unaryop(op: UnaryOp, inner: LuaExpr) -> Result<LuaExpr, LuaError> {
    match op {
        UnaryOp::Neg => {
            let code = match inner.typ {
                Type::Scalar => format!("(-{})", inner.code),
                Type::Vec2 => format!("{{-{}[1], -{}[2]}}", inner.code, inner.code),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "{{-{}[1], -{}[2], -{}[3]}}",
                    inner.code, inner.code, inner.code
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "{{-{}[1], -{}[2], -{}[3], -{}[4]}}",
                    inner.code, inner.code, inner.code, inner.code
                ),
                Type::Mat2 => {
                    let parts: Vec<String> =
                        (1..=4).map(|i| format!("-{}[{}]", inner.code, i)).collect();
                    format!("{{{}}}", parts.join(", "))
                }
                #[cfg(feature = "3d")]
                Type::Mat3 => {
                    let parts: Vec<String> =
                        (1..=9).map(|i| format!("-{}[{}]", inner.code, i)).collect();
                    format!("{{{}}}", parts.join(", "))
                }
                #[cfg(feature = "4d")]
                Type::Mat4 => {
                    let parts: Vec<String> = (1..=16)
                        .map(|i| format!("-{}[{}]", inner.code, i))
                        .collect();
                    format!("{{{}}}", parts.join(", "))
                }
            };
            Ok(LuaExpr {
                code,
                typ: inner.typ,
            })
        }
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
        "dot" => {
            if args.len() != 2 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let code = match args[0].typ {
                Type::Vec2 => format!(
                    "({}[1]*{}[1] + {}[2]*{}[2])",
                    args[0].code, args[1].code, args[0].code, args[1].code
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "({}[1]*{}[1] + {}[2]*{}[2] + {}[3]*{}[3])",
                    args[0].code,
                    args[1].code,
                    args[0].code,
                    args[1].code,
                    args[0].code,
                    args[1].code
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "({}[1]*{}[1] + {}[2]*{}[2] + {}[3]*{}[3] + {}[4]*{}[4])",
                    args[0].code,
                    args[1].code,
                    args[0].code,
                    args[1].code,
                    args[0].code,
                    args[1].code,
                    args[0].code,
                    args[1].code
                ),
                _ => return Err(LuaError::UnsupportedType(args[0].typ)),
            };
            Ok(LuaExpr {
                code,
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            Ok(LuaExpr {
                code: format!(
                    "{{{a}[2]*{b}[3] - {a}[3]*{b}[2], {a}[3]*{b}[1] - {a}[1]*{b}[3], {a}[1]*{b}[2] - {a}[2]*{b}[1]}}"
                ),
                typ: Type::Vec3,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let code = match args[0].typ {
                Type::Vec2 => format!(
                    "math.sqrt({}[1]*{}[1] + {}[2]*{}[2])",
                    args[0].code, args[0].code, args[0].code, args[0].code
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "math.sqrt({}[1]*{}[1] + {}[2]*{}[2] + {}[3]*{}[3])",
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "math.sqrt({}[1]*{}[1] + {}[2]*{}[2] + {}[3]*{}[3] + {}[4]*{}[4])",
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code,
                    args[0].code
                ),
                _ => return Err(LuaError::UnsupportedType(args[0].typ)),
            };
            Ok(LuaExpr {
                code,
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let typ = args[0].typ;
            let code = match typ {
                Type::Vec2 => format!(
                    "(function() local _len = math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2]); return {{{v}[1]/_len, {v}[2]/_len}} end)()"
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "(function() local _len = math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3]); return {{{v}[1]/_len, {v}[2]/_len, {v}[3]/_len}} end)()"
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "(function() local _len = math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3] + {v}[4]*{v}[4]); return {{{v}[1]/_len, {v}[2]/_len, {v}[3]/_len, {v}[4]/_len}} end)()"
                ),
                _ => return Err(LuaError::UnsupportedType(typ)),
            };
            Ok(LuaExpr { code, typ })
        }

        "distance" => {
            if args.len() != 2 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let code = match args[0].typ {
                Type::Vec2 => format!(
                    "math.sqrt(({a}[1]-{b}[1])*({a}[1]-{b}[1]) + ({a}[2]-{b}[2])*({a}[2]-{b}[2]))"
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "math.sqrt(({a}[1]-{b}[1])*({a}[1]-{b}[1]) + ({a}[2]-{b}[2])*({a}[2]-{b}[2]) + ({a}[3]-{b}[3])*({a}[3]-{b}[3]))"
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "math.sqrt(({a}[1]-{b}[1])*({a}[1]-{b}[1]) + ({a}[2]-{b}[2])*({a}[2]-{b}[2]) + ({a}[3]-{b}[3])*({a}[3]-{b}[3]) + ({a}[4]-{b}[4])*({a}[4]-{b}[4]))"
                ),
                _ => return Err(LuaError::UnsupportedType(args[0].typ)),
            };
            Ok(LuaExpr {
                code,
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let i = &args[0].code;
            let n = &args[1].code;
            let typ = args[0].typ;
            let code = match typ {
                Type::Vec2 => format!(
                    "(function() local _d = 2*({i}[1]*{n}[1] + {i}[2]*{n}[2]); return {{{i}[1] - _d*{n}[1], {i}[2] - _d*{n}[2]}} end)()"
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "(function() local _d = 2*({i}[1]*{n}[1] + {i}[2]*{n}[2] + {i}[3]*{n}[3]); return {{{i}[1] - _d*{n}[1], {i}[2] - _d*{n}[2], {i}[3] - _d*{n}[3]}} end)()"
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "(function() local _d = 2*({i}[1]*{n}[1] + {i}[2]*{n}[2] + {i}[3]*{n}[3] + {i}[4]*{n}[4]); return {{{i}[1] - _d*{n}[1], {i}[2] - _d*{n}[2], {i}[3] - _d*{n}[3], {i}[4] - _d*{n}[4]}} end)()"
                ),
                _ => return Err(LuaError::UnsupportedType(typ)),
            };
            Ok(LuaExpr { code, typ })
        }

        "hadamard" => {
            if args.len() != 2 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let typ = args[0].typ;
            let code = match typ {
                Type::Vec2 => format!("{{{a}[1]*{b}[1], {a}[2]*{b}[2]}}"),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!("{{{a}[1]*{b}[1], {a}[2]*{b}[2], {a}[3]*{b}[3]}}"),
                #[cfg(feature = "4d")]
                Type::Vec4 => {
                    format!("{{{a}[1]*{b}[1], {a}[2]*{b}[2], {a}[3]*{b}[3], {a}[4]*{b}[4]}}")
                }
                _ => return Err(LuaError::UnsupportedType(typ)),
            };
            Ok(LuaExpr { code, typ })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let t = &args[2].code;
            let typ = args[0].typ;
            let code = match typ {
                Type::Scalar => format!("({a} + ({b} - {a}) * {t})"),
                Type::Vec2 => format!(
                    "{{{a}[1] + ({b}[1] - {a}[1]) * {t}, {a}[2] + ({b}[2] - {a}[2]) * {t}}}"
                ),
                #[cfg(feature = "3d")]
                Type::Vec3 => format!(
                    "{{{a}[1] + ({b}[1] - {a}[1]) * {t}, {a}[2] + ({b}[2] - {a}[2]) * {t}, {a}[3] + ({b}[3] - {a}[3]) * {t}}}"
                ),
                #[cfg(feature = "4d")]
                Type::Vec4 => format!(
                    "{{{a}[1] + ({b}[1] - {a}[1]) * {t}, {a}[2] + ({b}[2] - {a}[2]) * {t}, {a}[3] + ({b}[3] - {a}[3]) * {t}, {a}[4] + ({b}[4] - {a}[4]) * {t}}}"
                ),
                _ => return Err(LuaError::UnsupportedType(typ)),
            };
            Ok(LuaExpr { code, typ })
        }

        _ => Err(LuaError::UnknownFunction(name.to_string())),
    }
}

/// Returns Lua helper functions needed for matrix operations.
pub fn lua_helpers() -> &'static str {
    r#"
function __linalg_mat3_mul(a, b)
    return {
        a[1]*b[1] + a[4]*b[2] + a[7]*b[3], a[2]*b[1] + a[5]*b[2] + a[8]*b[3], a[3]*b[1] + a[6]*b[2] + a[9]*b[3],
        a[1]*b[4] + a[4]*b[5] + a[7]*b[6], a[2]*b[4] + a[5]*b[5] + a[8]*b[6], a[3]*b[4] + a[6]*b[5] + a[9]*b[6],
        a[1]*b[7] + a[4]*b[8] + a[7]*b[9], a[2]*b[7] + a[5]*b[8] + a[8]*b[9], a[3]*b[7] + a[6]*b[8] + a[9]*b[9]
    }
end

function __linalg_mat4_mul(a, b)
    return {
        a[1]*b[1] + a[5]*b[2] + a[9]*b[3] + a[13]*b[4], a[2]*b[1] + a[6]*b[2] + a[10]*b[3] + a[14]*b[4],
        a[3]*b[1] + a[7]*b[2] + a[11]*b[3] + a[15]*b[4], a[4]*b[1] + a[8]*b[2] + a[12]*b[3] + a[16]*b[4],
        a[1]*b[5] + a[5]*b[6] + a[9]*b[7] + a[13]*b[8], a[2]*b[5] + a[6]*b[6] + a[10]*b[7] + a[14]*b[8],
        a[3]*b[5] + a[7]*b[6] + a[11]*b[7] + a[15]*b[8], a[4]*b[5] + a[8]*b[6] + a[12]*b[7] + a[16]*b[8],
        a[1]*b[9] + a[5]*b[10] + a[9]*b[11] + a[13]*b[12], a[2]*b[9] + a[6]*b[10] + a[10]*b[11] + a[14]*b[12],
        a[3]*b[9] + a[7]*b[10] + a[11]*b[11] + a[15]*b[12], a[4]*b[9] + a[8]*b[10] + a[12]*b[11] + a[16]*b[12],
        a[1]*b[13] + a[5]*b[14] + a[9]*b[15] + a[13]*b[16], a[2]*b[13] + a[6]*b[14] + a[10]*b[15] + a[14]*b[16],
        a[3]*b[13] + a[7]*b[14] + a[11]*b[15] + a[15]*b[16], a[4]*b[13] + a[8]*b[14] + a[12]*b[15] + a[16]*b[16]
    }
end
"#
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

    // Load helper functions
    lua.load(lua_helpers()).exec().map_err(EvalError::Lua)?;

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
        Value::Vec2(v) => {
            let table = lua.create_table()?;
            table.set(1, v[0].into_lua(lua)?)?;
            table.set(2, v[1].into_lua(lua)?)?;
            globals.set(name, table)?;
        }
        #[cfg(feature = "3d")]
        Value::Vec3(v) => {
            let table = lua.create_table()?;
            table.set(1, v[0].into_lua(lua)?)?;
            table.set(2, v[1].into_lua(lua)?)?;
            table.set(3, v[2].into_lua(lua)?)?;
            globals.set(name, table)?;
        }
        #[cfg(feature = "4d")]
        Value::Vec4(v) => {
            let table = lua.create_table()?;
            for (i, val) in v.iter().enumerate() {
                table.set(i + 1, (*val).into_lua(lua)?)?;
            }
            globals.set(name, table)?;
        }
        Value::Mat2(m) => {
            let table = lua.create_table()?;
            for (i, val) in m.iter().enumerate() {
                table.set(i + 1, (*val).into_lua(lua)?)?;
            }
            globals.set(name, table)?;
        }
        #[cfg(feature = "3d")]
        Value::Mat3(m) => {
            let table = lua.create_table()?;
            for (i, val) in m.iter().enumerate() {
                table.set(i + 1, (*val).into_lua(lua)?)?;
            }
            globals.set(name, table)?;
        }
        #[cfg(feature = "4d")]
        Value::Mat4(m) => {
            let table = lua.create_table()?;
            for (i, val) in m.iter().enumerate() {
                table.set(i + 1, (*val).into_lua(lua)?)?;
            }
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
        Type::Vec2 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Vec2".into()))?;
            let v1: f64 = table.get(1).map_err(EvalError::Lua)?;
            let v2: f64 = table.get(2).map_err(EvalError::Lua)?;
            Ok(Value::Vec2([
                T::from(v1).ok_or_else(|| EvalError::TypeConversion("vec2[1]".into()))?,
                T::from(v2).ok_or_else(|| EvalError::TypeConversion("vec2[2]".into()))?,
            ]))
        }
        #[cfg(feature = "3d")]
        Type::Vec3 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Vec3".into()))?;
            let v1: f64 = table.get(1).map_err(EvalError::Lua)?;
            let v2: f64 = table.get(2).map_err(EvalError::Lua)?;
            let v3: f64 = table.get(3).map_err(EvalError::Lua)?;
            Ok(Value::Vec3([
                T::from(v1).ok_or_else(|| EvalError::TypeConversion("vec3[1]".into()))?,
                T::from(v2).ok_or_else(|| EvalError::TypeConversion("vec3[2]".into()))?,
                T::from(v3).ok_or_else(|| EvalError::TypeConversion("vec3[3]".into()))?,
            ]))
        }
        #[cfg(feature = "4d")]
        Type::Vec4 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Vec4".into()))?;
            let mut arr = [T::zero(); 4];
            for i in 0..4 {
                let v: f64 = table.get(i + 1).map_err(EvalError::Lua)?;
                arr[i] = T::from(v)
                    .ok_or_else(|| EvalError::TypeConversion(format!("vec4[{}]", i + 1)))?;
            }
            Ok(Value::Vec4(arr))
        }
        Type::Mat2 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Mat2".into()))?;
            let mut arr = [T::zero(); 4];
            for i in 0..4 {
                let v: f64 = table.get(i + 1).map_err(EvalError::Lua)?;
                arr[i] = T::from(v)
                    .ok_or_else(|| EvalError::TypeConversion(format!("mat2[{}]", i + 1)))?;
            }
            Ok(Value::Mat2(arr))
        }
        #[cfg(feature = "3d")]
        Type::Mat3 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Mat3".into()))?;
            let mut arr = [T::zero(); 9];
            for i in 0..9 {
                let v: f64 = table.get(i + 1).map_err(EvalError::Lua)?;
                arr[i] = T::from(v)
                    .ok_or_else(|| EvalError::TypeConversion(format!("mat3[{}]", i + 1)))?;
            }
            Ok(Value::Mat3(arr))
        }
        #[cfg(feature = "4d")]
        Type::Mat4 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::TypeConversion("expected table for Mat4".into()))?;
            let mut arr = [T::zero(); 16];
            for i in 0..16 {
                let v: f64 = table.get(i + 1).map_err(EvalError::Lua)?;
                arr[i] = T::from(v)
                    .ok_or_else(|| EvalError::TypeConversion(format!("mat4[{}]", i + 1)))?;
            }
            Ok(Value::Mat4(arr))
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
    fn test_scalar_add() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Scalar)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("+"));
    }

    #[test]
    fn test_vec2_add() {
        let result = emit("a + b", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
        assert!(result.code.contains("[1]"));
    }

    #[test]
    fn test_mat_vec_mul() {
        let result = emit("m * v", &[("m", Type::Mat2), ("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_vec_mat_mul() {
        let result = emit("v * m", &[("v", Type::Vec2), ("m", Type::Mat2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_dot() {
        let result = emit("dot(a, b)", &[("a", Type::Vec2), ("b", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
        assert!(result.code.contains("math.sqrt"));
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
    }

    #[test]
    fn test_lerp() {
        let result = emit(
            "lerp(a, b, t)",
            &[("a", Type::Vec2), ("b", Type::Vec2), ("t", Type::Scalar)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Vec2);
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(LuaError::TypeMismatch { .. })));
    }
}
