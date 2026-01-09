//! WGSL code generation for quaternion expressions.
//!
//! Quaternions are represented as vec4<f32> where xyz=imaginary, w=real (xyzw order).
//! Vectors are vec3<f32>.

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
        Type::Vec3 => "vec3<f32>",
        Type::Quaternion => "vec4<f32>",
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
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(WgslExpr {
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
        (Type::Scalar, Type::Scalar)
        | (Type::Vec3, Type::Vec3)
        | (Type::Quaternion, Type::Quaternion) => Ok(WgslExpr {
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

        // Scalar * Vec3 / Vec3 * Scalar
        (Type::Scalar, Type::Vec3) | (Type::Vec3, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Vec3,
        }),

        // Scalar * Quaternion / Quaternion * Scalar
        (Type::Scalar, Type::Quaternion) | (Type::Quaternion, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} * {})", left.code, right.code),
            typ: Type::Quaternion,
        }),

        // Quaternion * Quaternion (Hamilton product)
        // q1 = [x1, y1, z1, w1], q2 = [x2, y2, z2, w2]
        (Type::Quaternion, Type::Quaternion) => {
            let q1 = &left.code;
            let q2 = &right.code;
            Ok(WgslExpr {
                code: format!(
                    "vec4<f32>(\
                        {q1}.w * {q2}.x + {q1}.x * {q2}.w + {q1}.y * {q2}.z - {q1}.z * {q2}.y, \
                        {q1}.w * {q2}.y - {q1}.x * {q2}.z + {q1}.y * {q2}.w + {q1}.z * {q2}.x, \
                        {q1}.w * {q2}.z + {q1}.x * {q2}.y - {q1}.y * {q2}.x + {q1}.z * {q2}.w, \
                        {q1}.w * {q2}.w - {q1}.x * {q2}.x - {q1}.y * {q2}.y - {q1}.z * {q2}.z)"
                ),
                typ: Type::Quaternion,
            })
        }

        // Quaternion * Vec3 (rotate vector)
        // Using optimized formula: v' = v + 2w(q×v) + 2(q×(q×v))
        (Type::Quaternion, Type::Vec3) => {
            let q = &left.code;
            let v = &right.code;
            Ok(WgslExpr {
                code: format!(
                    "(func() -> vec3<f32> {{ \
                        let _t = 2.0 * cross({q}.xyz, {v}); \
                        return {v} + {q}.w * _t + cross({q}.xyz, _t); \
                    }})()"
                ),
                typ: Type::Vec3,
            })
        }

        _ => Err(WgslError::TypeMismatch {
            op: "*",
            left: left.typ,
            right: right.typ,
        }),
    }
}

fn emit_div(left: WgslExpr, right: WgslExpr) -> Result<WgslExpr, WgslError> {
    match (left.typ, right.typ) {
        (Type::Scalar, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Scalar,
        }),
        (Type::Vec3, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Vec3,
        }),
        (Type::Quaternion, Type::Scalar) => Ok(WgslExpr {
            code: format!("({} / {})", left.code, right.code),
            typ: Type::Quaternion,
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
        "conj" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("vec4<f32>(-{q}.xyz, {q}.w)", q = args[0].code),
                typ: Type::Quaternion,
            })
        }

        "length" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("length({})", args[0].code),
                typ: Type::Scalar,
            })
        }

        "normalize" => {
            if args.len() != 1 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("normalize({})", args[0].code),
                typ: args[0].typ,
            })
        }

        "inverse" => {
            if args.len() != 1 || args[0].typ != Type::Quaternion {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            // inverse(q) = conj(q) / |q|²
            let q = &args[0].code;
            Ok(WgslExpr {
                code: format!("(vec4<f32>(-{q}.xyz, {q}.w) / dot({q}, {q}))"),
                typ: Type::Quaternion,
            })
        }

        "dot" => {
            if args.len() != 2 || args[0].typ != args[1].typ {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("dot({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "lerp" => {
            if args.len() != 3 || args[2].typ != Type::Scalar {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("mix({}, {}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        "slerp" => {
            if args.len() != 3
                || args[0].typ != Type::Quaternion
                || args[1].typ != Type::Quaternion
                || args[2].typ != Type::Scalar
            {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let q1 = &args[0].code;
            let q2 = &args[1].code;
            let t = &args[2].code;
            // Simplified slerp - for production, should handle edge cases
            Ok(WgslExpr {
                code: format!(
                    "(func() -> vec4<f32> {{ \
                        var _d = dot({q1}, {q2}); \
                        var _q2 = {q2}; \
                        if (_d < 0.0) {{ _d = -_d; _q2 = -{q2}; }} \
                        if (_d > 0.9995) {{ return normalize(mix({q1}, _q2, {t})); }} \
                        let _theta = acos(_d); \
                        let _s = sin(_theta); \
                        return ({q1} * sin((1.0 - {t}) * _theta) + _q2 * sin({t} * _theta)) / _s; \
                    }})()"
                ),
                typ: Type::Quaternion,
            })
        }

        "axis_angle" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Scalar {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let axis = &args[0].code;
            let angle = &args[1].code;
            Ok(WgslExpr {
                code: format!(
                    "(func() -> vec4<f32> {{ \
                        let _half = {angle} * 0.5; \
                        return vec4<f32>(normalize({axis}) * sin(_half), cos(_half)); \
                    }})()"
                ),
                typ: Type::Quaternion,
            })
        }

        "rotate" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Quaternion {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            let v = &args[0].code;
            let q = &args[1].code;
            // Using optimized formula
            Ok(WgslExpr {
                code: format!(
                    "(func() -> vec3<f32> {{ \
                        let _t = 2.0 * cross({q}.xyz, {v}); \
                        return {v} + {q}.w * _t + cross({q}.xyz, _t); \
                    }})()"
                ),
                typ: Type::Vec3,
            })
        }

        "cross" => {
            if args.len() != 2 || args[0].typ != Type::Vec3 || args[1].typ != Type::Vec3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("cross({}, {})", args[0].code, args[1].code),
                typ: Type::Vec3,
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
    fn test_quaternion_add() {
        let result = emit("a + b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
    }

    #[test]
    fn test_quaternion_mul() {
        let result = emit("a * b", &[("a", Type::Quaternion), ("b", Type::Quaternion)]).unwrap();
        assert_eq!(result.typ, Type::Quaternion);
        // Should contain Hamilton product
        assert!(result.code.contains(".w") && result.code.contains(".x"));
    }

    #[test]
    fn test_quaternion_rotate_vec() {
        let result = emit("q * v", &[("q", Type::Quaternion), ("v", Type::Vec3)]).unwrap();
        assert_eq!(result.typ, Type::Vec3);
        assert!(result.code.contains("cross"));
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
}
