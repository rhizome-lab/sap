//! Lua code generation for quaternion expressions.
//!
//! Quaternions are represented as tables {x, y, z, w}.
//! Vectors are tables {x, y, z}.

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
