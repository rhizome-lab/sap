//! Lua code generation and evaluation for quaternion expressions.
//!
//! Quaternions are represented as tables {x, y, z, w}.
//! Vectors are tables {x, y, z}.

use crate::{Type, Value};
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
        (Type::Vec3, Type::Vec3) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] + {}[1], {}[2] + {}[2], {}[3] + {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Quaternion) => Ok(LuaExpr {
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
            typ: Type::Quaternion,
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
        (Type::Vec3, Type::Vec3) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] - {}[1], {}[2] - {}[2], {}[3] - {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Quaternion) => Ok(LuaExpr {
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
            typ: Type::Quaternion,
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

        // Scalar * Vec3
        (Type::Scalar, Type::Vec3) => Ok(LuaExpr {
            code: format!(
                "{{{} * {}[1], {} * {}[2], {} * {}[3]}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        (Type::Vec3, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] * {}, {}[2] * {}, {}[3] * {}}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),

        // Scalar * Quaternion
        (Type::Scalar, Type::Quaternion) => Ok(LuaExpr {
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
            typ: Type::Quaternion,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(LuaExpr {
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
            typ: Type::Quaternion,
        }),

        // Quaternion * Quaternion (Hamilton product)
        // q1 = {x1, y1, z1, w1}, q2 = {x2, y2, z2, w2}
        (Type::Quaternion, Type::Quaternion) => {
            let q1 = &left.code;
            let q2 = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "{{\
                        {q1}[4]*{q2}[1] + {q1}[1]*{q2}[4] + {q1}[2]*{q2}[3] - {q1}[3]*{q2}[2], \
                        {q1}[4]*{q2}[2] - {q1}[1]*{q2}[3] + {q1}[2]*{q2}[4] + {q1}[3]*{q2}[1], \
                        {q1}[4]*{q2}[3] + {q1}[1]*{q2}[2] - {q1}[2]*{q2}[1] + {q1}[3]*{q2}[4], \
                        {q1}[4]*{q2}[4] - {q1}[1]*{q2}[1] - {q1}[2]*{q2}[2] - {q1}[3]*{q2}[3]}}"
                ),
                typ: Type::Quaternion,
            })
        }

        // Quaternion * Vec3 (rotate vector)
        (Type::Quaternion, Type::Vec3) => {
            let q = &left.code;
            let v = &right.code;
            Ok(LuaExpr {
                code: format!(
                    "(function() \
                        local _tx = 2.0 * ({q}[2]*{v}[3] - {q}[3]*{v}[2]); \
                        local _ty = 2.0 * ({q}[3]*{v}[1] - {q}[1]*{v}[3]); \
                        local _tz = 2.0 * ({q}[1]*{v}[2] - {q}[2]*{v}[1]); \
                        return {{ \
                            {v}[1] + {q}[4]*_tx + ({q}[2]*_tz - {q}[3]*_ty), \
                            {v}[2] + {q}[4]*_ty + ({q}[3]*_tx - {q}[1]*_tz), \
                            {v}[3] + {q}[4]*_tz + ({q}[1]*_ty - {q}[2]*_tx) \
                        }} \
                    end)()"
                ),
                typ: Type::Vec3,
            })
        }

        _ => Err(LuaError::TypeMismatch {
            op: "*",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_div(left: LuaExpr, right: LuaExpr) -> Result<LuaExpr, LuaError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(LuaExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Vec3, Type::Scalar) => Ok(LuaExpr {
            code: format!(
                "{{{}[1] / {}, {}[2] / {}, {}[3] / {}}}",
                left.code, right.code, left.code, right.code, left.code, right.code
            ),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(LuaExpr {
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
            typ: Type::Quaternion,
        }),
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
            Type::Vec3 => Ok(LuaExpr {
                code: format!(
                    "{{-{}[1], -{}[2], -{}[3]}}",
                    inner.code, inner.code, inner.code
                ),
                typ: Type::Vec3,
            }),
            Type::Quaternion => Ok(LuaExpr {
                code: format!(
                    "{{-{}[1], -{}[2], -{}[3], -{}[4]}}",
                    inner.code, inner.code, inner.code, inner.code
                ),
                typ: Type::Quaternion,
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
        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let q = &args[0].code;
            Ok(LuaExpr {
                code: format!("{{-{q}[1], -{q}[2], -{q}[3], {q}[4]}}"),
                typ: Type::Quaternion,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            match args[0].typ {
                Type::Vec3 => Ok(LuaExpr {
                    code: format!("math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3])"),
                    typ: Type::Scalar,
                }),
                Type::Quaternion => Ok(LuaExpr {
                    code: format!(
                        "math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3] + {v}[4]*{v}[4])"
                    ),
                    typ: Type::Scalar,
                }),
                _ => Err(LuaError::UnknownFunction(name.to_string())),
            }
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            match args[0].typ {
                Type::Vec3 => Ok(LuaExpr {
                    code: format!(
                        "(function() local _len = math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3]); return {{{v}[1]/_len, {v}[2]/_len, {v}[3]/_len}} end)()"
                    ),
                    typ: Type::Vec3,
                }),
                Type::Quaternion => Ok(LuaExpr {
                    code: format!(
                        "(function() local _len = math.sqrt({v}[1]*{v}[1] + {v}[2]*{v}[2] + {v}[3]*{v}[3] + {v}[4]*{v}[4]); return {{{v}[1]/_len, {v}[2]/_len, {v}[3]/_len, {v}[4]/_len}} end)()"
                    ),
                    typ: Type::Quaternion,
                }),
                _ => Err(LuaError::UnknownFunction(name.to_string())),
            }
        }

        "inverse" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let q = &args[0].code;
            Ok(LuaExpr {
                code: format!(
                    "(function() local _norm = {q}[1]*{q}[1] + {q}[2]*{q}[2] + {q}[3]*{q}[3] + {q}[4]*{q}[4]; return {{-{q}[1]/_norm, -{q}[2]/_norm, -{q}[3]/_norm, {q}[4]/_norm}} end)()"
                ),
                typ: Type::Quaternion,
            })
        }

        "dot" => {
            if args.len() != 2 || args[0].typ != args[1].typ {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            match args[0].typ {
                Type::Vec3 => Ok(LuaExpr {
                    code: format!("({a}[1]*{b}[1] + {a}[2]*{b}[2] + {a}[3]*{b}[3])"),
                    typ: Type::Scalar,
                }),
                Type::Quaternion => Ok(LuaExpr {
                    code: format!(
                        "({a}[1]*{b}[1] + {a}[2]*{b}[2] + {a}[3]*{b}[3] + {a}[4]*{b}[4])"
                    ),
                    typ: Type::Scalar,
                }),
                _ => Err(LuaError::UnknownFunction(name.to_string())),
            }
        }

        "lerp" => {
            if args.len() != 3 || args[2].typ != Type::Scalar {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let a = &args[0].code;
            let b = &args[1].code;
            let t = &args[2].code;
            match args[0].typ {
                Type::Scalar => Ok(LuaExpr {
                    code: format!("({a} + ({b} - {a}) * {t})"),
                    typ: Type::Scalar,
                }),
                Type::Vec3 => Ok(LuaExpr {
                    code: format!(
                        "{{{a}[1] + ({b}[1] - {a}[1])*{t}, {a}[2] + ({b}[2] - {a}[2])*{t}, {a}[3] + ({b}[3] - {a}[3])*{t}}}"
                    ),
                    typ: Type::Vec3,
                }),
                Type::Quaternion => Ok(LuaExpr {
                    code: format!(
                        "{{{a}[1] + ({b}[1] - {a}[1])*{t}, {a}[2] + ({b}[2] - {a}[2])*{t}, {a}[3] + ({b}[3] - {a}[3])*{t}, {a}[4] + ({b}[4] - {a}[4])*{t}}}"
                    ),
                    typ: Type::Quaternion,
                }),
            }
        }

        "slerp" => {
            if args.len() != 3
                || args[0].typ != Type::Quaternion
                || args[1].typ != Type::Quaternion
                || args[2].typ != Type::Scalar
            {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let q1 = &args[0].code;
            let q2 = &args[1].code;
            let t = &args[2].code;
            Ok(LuaExpr {
                code: format!(
                    "(function() \
                        local _d = {q1}[1]*{q2}[1] + {q1}[2]*{q2}[2] + {q1}[3]*{q2}[3] + {q1}[4]*{q2}[4]; \
                        local _q2 = {q2}; \
                        if _d < 0 then _d = -_d; _q2 = {{-{q2}[1], -{q2}[2], -{q2}[3], -{q2}[4]}} end; \
                        if _d > 0.9995 then \
                            local _len = math.sqrt( \
                                ({q1}[1] + (_q2[1] - {q1}[1])*{t})^2 + ({q1}[2] + (_q2[2] - {q1}[2])*{t})^2 + \
                                ({q1}[3] + (_q2[3] - {q1}[3])*{t})^2 + ({q1}[4] + (_q2[4] - {q1}[4])*{t})^2); \
                            return {{ \
                                ({q1}[1] + (_q2[1] - {q1}[1])*{t})/_len, \
                                ({q1}[2] + (_q2[2] - {q1}[2])*{t})/_len, \
                                ({q1}[3] + (_q2[3] - {q1}[3])*{t})/_len, \
                                ({q1}[4] + (_q2[4] - {q1}[4])*{t})/_len \
                            }} \
                        end; \
                        local _theta = math.acos(_d); \
                        local _s = math.sin(_theta); \
                        local _s1 = math.sin((1 - {t}) * _theta) / _s; \
                        local _s2 = math.sin({t} * _theta) / _s; \
                        return {{ \
                            {q1}[1]*_s1 + _q2[1]*_s2, \
                            {q1}[2]*_s1 + _q2[2]*_s2, \
                            {q1}[3]*_s1 + _q2[3]*_s2, \
                            {q1}[4]*_s1 + _q2[4]*_s2 \
                        }} \
                    end)()"
                ),
                typ: Type::Quaternion,
            })
        }

        "axis_angle" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Scalar {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let axis = &args[0].code;
            let angle = &args[1].code;
            Ok(LuaExpr {
                code: format!(
                    "(function() \
                        local _len = math.sqrt({axis}[1]*{axis}[1] + {axis}[2]*{axis}[2] + {axis}[3]*{axis}[3]); \
                        local _half = {angle} * 0.5; \
                        local _s = math.sin(_half) / _len; \
                        return {{{axis}[1]*_s, {axis}[2]*_s, {axis}[3]*_s, math.cos(_half)}} \
                    end)()"
                ),
                typ: Type::Quaternion,
            })
        }

        "rotate" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Quaternion {
                return Err(LuaError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let q = &args[1].code;
            Ok(LuaExpr {
                code: format!(
                    "(function() \
                        local _tx = 2.0 * ({q}[2]*{v}[3] - {q}[3]*{v}[2]); \
                        local _ty = 2.0 * ({q}[3]*{v}[1] - {q}[1]*{v}[3]); \
                        local _tz = 2.0 * ({q}[1]*{v}[2] - {q}[2]*{v}[1]); \
                        return {{ \
                            {v}[1] + {q}[4]*_tx + ({q}[2]*_tz - {q}[3]*_ty), \
                            {v}[2] + {q}[4]*_ty + ({q}[3]*_tx - {q}[1]*_tz), \
                            {v}[3] + {q}[4]*_tz + ({q}[1]*_ty - {q}[2]*_tx) \
                        }} \
                    end)()"
                ),
                typ: Type::Vec3,
            })
        }

        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Vec3 {
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

        _ => Err(LuaError::UnknownFunction(name.to_string())),
    }
}

// ============================================================================
// Lua Evaluation
// ============================================================================

/// Error during Lua evaluation.
#[derive(Debug)]
pub enum EvalError {
    /// Code generation failed.
    CodeGen(LuaError),
    /// Lua runtime error.
    Runtime(String),
    /// Result type conversion failed.
    ResultConversion(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::CodeGen(e) => write!(f, "code generation error: {e}"),
            EvalError::Runtime(e) => write!(f, "lua runtime error: {e}"),
            EvalError::ResultConversion(e) => write!(f, "result conversion error: {e}"),
        }
    }
}

impl std::error::Error for EvalError {}

impl From<LuaError> for EvalError {
    fn from(e: LuaError) -> Self {
        EvalError::CodeGen(e)
    }
}

/// Evaluate an expression using Lua.
///
/// This creates a Lua VM, sets up variables, emits the expression as Lua code,
/// and executes it, returning the result as a Value.
#[cfg(feature = "lua")]
pub fn eval_lua<T: num_traits::Float + mlua::IntoLua + mlua::FromLua>(
    ast: &Ast,
    vars: &HashMap<String, Value<T>>,
) -> Result<Value<T>, EvalError> {
    use mlua::Lua;

    // Build var_types map for emit_lua
    let var_types: HashMap<String, Type> = vars.iter().map(|(k, v)| (k.clone(), v.typ())).collect();

    // Generate Lua code
    let lua_expr = emit_lua(ast, &var_types)?;

    // Create Lua VM
    let lua = Lua::new();

    // Set up variables
    for (name, value) in vars {
        set_lua_var(&lua, name, value).map_err(|e| EvalError::Runtime(e.to_string()))?;
    }

    // Execute and convert result
    let code = format!("return {}", lua_expr.code);
    let result: mlua::Value = lua
        .load(&code)
        .eval()
        .map_err(|e| EvalError::Runtime(e.to_string()))?;

    lua_to_value(&result, lua_expr.typ)
}

/// Set a Lua variable from a Value.
#[cfg(feature = "lua")]
fn set_lua_var<T: num_traits::Float + mlua::IntoLua>(
    lua: &mlua::Lua,
    name: &str,
    value: &Value<T>,
) -> mlua::Result<()> {
    let globals = lua.globals();
    match value {
        Value::Scalar(s) => {
            globals.set(name, s.clone())?;
        }
        Value::Vec3(v) => {
            let table = lua.create_table()?;
            table.set(1, v[0].clone())?;
            table.set(2, v[1].clone())?;
            table.set(3, v[2].clone())?;
            globals.set(name, table)?;
        }
        Value::Quaternion(q) => {
            let table = lua.create_table()?;
            table.set(1, q[0].clone())?;
            table.set(2, q[1].clone())?;
            table.set(3, q[2].clone())?;
            table.set(4, q[3].clone())?;
            globals.set(name, table)?;
        }
    }
    Ok(())
}

/// Convert a Lua value back to a Value.
#[cfg(feature = "lua")]
fn lua_to_value<T: num_traits::Float + mlua::FromLua>(
    lua_val: &mlua::Value,
    expected_type: Type,
) -> Result<Value<T>, EvalError> {
    match expected_type {
        Type::Scalar => {
            let n = T::from_lua(lua_val.clone(), &mlua::Lua::new())
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            Ok(Value::Scalar(n))
        }
        Type::Vec3 => {
            let table = lua_val
                .as_table()
                .ok_or_else(|| EvalError::ResultConversion("expected table for Vec3".into()))?;
            let x: T = table
                .get(1)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            let y: T = table
                .get(2)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            let z: T = table
                .get(3)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            Ok(Value::Vec3([x, y, z]))
        }
        Type::Quaternion => {
            let table = lua_val.as_table().ok_or_else(|| {
                EvalError::ResultConversion("expected table for Quaternion".into())
            })?;
            let x: T = table
                .get(1)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            let y: T = table
                .get(2)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            let z: T = table
                .get(3)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            let w: T = table
                .get(4)
                .map_err(|e| EvalError::ResultConversion(e.to_string()))?;
            Ok(Value::Quaternion([x, y, z, w]))
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
    fn test_quaternion_add() {
        let result = emit("a + b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
    }

    #[test]
    fn test_quaternion_mul() {
        let result = emit("a * b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
    }

    #[test]
    fn test_quaternion_rotate_vec() {
        let result = emit("q * v", &[("q", Type::Quaternion), ("v", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
    }

    #[test]
    fn test_normalize() {
        let result = emit("normalize(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
    }

    #[test]
    fn test_conj() {
        let result = emit("conj(q)", &[("q", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
    }

    #[test]
    fn test_dot() {
        let result = emit(
            "dot(a, b)",
            &[("a", Type::Quaternion), ("b", Type::Quaternion)],
        )
        .unwrap();
        assert_eq!(result.typ, Type::Scalar);
    }

    #[test]
    fn test_cross() {
        let result = emit("cross(a, b)", &[("a", Type::Vec3), ("b", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
    }
}
