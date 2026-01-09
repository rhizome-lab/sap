//! WGSL code generation for linalg expressions.
//!
//! Emits WGSL code with proper type handling for vectors and matrices.

use crate::Type;
use rhizome_sap_core::{Ast, BinOp, UnaryOp};
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
    UnsupportedType(Type),
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WgslError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            WgslError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            WgslError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            WgslError::UnsupportedType(t) => write!(f, "unsupported type: {t}"),
        }
    }
}

impl std::error::Error for WgslError {}

/// Convert a Type to its WGSL representation.
pub fn type_to_wgsl(t: Type) -> &'static str {
    match t {
        Type::Scalar => "f32",
        Type::Vec2 => "vec2<f32>",
        #[cfg(feature = "3d")]
        Type::Vec3 => "vec3<f32>",
        #[cfg(feature = "4d")]
        Type::Vec4 => "vec4<f32>",
        Type::Mat2 => "mat2x2<f32>",
        #[cfg(feature = "3d")]
        Type::Mat3 => "mat3x3<f32>",
        #[cfg(feature = "4d")]
        Type::Mat4 => "mat4x4<f32>",
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
    let op_str = match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => return emit_pow(left, right),
    };

    let result_type = infer_binop_type(op, left.typ, right.typ)?;

    Ok(WgslExpr {
        code: format!("({} {} {})", left.code, op_str, right.code),
        typ: result_type,
    })
}

fn emit_pow(base: WgslExpr, exp: WgslExpr) -> Result<WgslExpr, WgslError> {
    // WGSL pow() only works on scalars
    if base.typ != Type::Scalar || exp.typ != Type::Scalar {
        return Err(WgslError::TypeMismatch {
            op: "^",
            left: base.typ,
            right: exp.typ,
        });
    }
    Ok(WgslExpr {
        code: format!("pow({}, {})", base.code, exp.code),
        typ: Type::Scalar,
    })
}

fn infer_binop_type(op: BinOp, left: Type, right: Type) -> Result<Type, WgslError> {
    match op {
        BinOp::Add | BinOp::Sub => {
            // Same types only
            if left == right {
                Ok(left)
            } else {
                Err(WgslError::TypeMismatch {
                    op: if op == BinOp::Add { "+" } else { "-" },
                    left,
                    right,
                })
            }
        }
        BinOp::Mul => infer_mul_type(left, right),
        BinOp::Div => {
            // vec / scalar or scalar / scalar
            match (left, right) {
                (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),
                (Type::Vec2, Type::Scalar) => Ok(Type::Vec2),
                #[cfg(feature = "3d")]
                (Type::Vec3, Type::Scalar) => Ok(Type::Vec3),
                #[cfg(feature = "4d")]
                (Type::Vec4, Type::Scalar) => Ok(Type::Vec4),
                _ => Err(WgslError::TypeMismatch {
                    op: "/",
                    left,
                    right,
                }),
            }
        }
        BinOp::Pow => {
            if left == Type::Scalar && right == Type::Scalar {
                Ok(Type::Scalar)
            } else {
                Err(WgslError::TypeMismatch {
                    op: "^",
                    left,
                    right,
                })
            }
        }
    }
}

fn infer_mul_type(left: Type, right: Type) -> Result<Type, WgslError> {
    match (left, right) {
        // Scalar * Scalar
        (Type::Scalar, Type::Scalar) => Ok(Type::Scalar),

        // Vec * Scalar or Scalar * Vec
        (Type::Vec2, Type::Scalar) | (Type::Scalar, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Scalar) | (Type::Scalar, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Scalar) | (Type::Scalar, Type::Vec4) => Ok(Type::Vec4),

        // Mat * Scalar or Scalar * Mat
        (Type::Mat2, Type::Scalar) | (Type::Scalar, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Scalar) | (Type::Scalar, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Scalar) | (Type::Scalar, Type::Mat4) => Ok(Type::Mat4),

        // Mat * Vec
        (Type::Mat2, Type::Vec2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Vec3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Vec4) => Ok(Type::Vec4),

        // Vec * Mat
        (Type::Vec2, Type::Mat2) => Ok(Type::Vec2),
        #[cfg(feature = "3d")]
        (Type::Vec3, Type::Mat3) => Ok(Type::Vec3),
        #[cfg(feature = "4d")]
        (Type::Vec4, Type::Mat4) => Ok(Type::Vec4),

        // Mat * Mat
        (Type::Mat2, Type::Mat2) => Ok(Type::Mat2),
        #[cfg(feature = "3d")]
        (Type::Mat3, Type::Mat3) => Ok(Type::Mat3),
        #[cfg(feature = "4d")]
        (Type::Mat4, Type::Mat4) => Ok(Type::Mat4),

        _ => Err(WgslError::TypeMismatch {
            op: "*",
            left,
            right,
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
        // Vector functions that map directly to WGSL builtins
        "dot" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("dot({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        #[cfg(feature = "3d")]
        "cross" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("cross({}, {})", args[0].code, args[1].code),
                typ: Type::Vec3,
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

        "distance" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("distance({}, {})", args[0].code, args[1].code),
                typ: Type::Scalar,
            })
        }

        "reflect" => {
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("reflect({}, {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "hadamard" => {
            // WGSL doesn't have hadamard, use element-wise multiply
            if args.len() != 2 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("({} * {})", args[0].code, args[1].code),
                typ: args[0].typ,
            })
        }

        "lerp" | "mix" => {
            if args.len() != 3 {
                return Err(WgslError::UnknownFunction(name.to_string()));
            }
            Ok(WgslExpr {
                code: format!("mix({}, {}, {})", args[0].code, args[1].code, args[2].code),
                typ: args[0].typ,
            })
        }

        _ => Err(WgslError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn emit(expr: &str, var_types: &[(&str, Type)]) -> Result<WgslExpr, WgslError> {
        let expr = Expr::parse(expr).unwrap();
        let types: HashMap<String, Type> =
            var_types.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        emit_wgsl(expr.ast(), &types)
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
        assert!(result.code.contains("dot"));
    }

    #[test]
    fn test_length() {
        let result = emit("length(v)", &[("v", Type::Vec2)]).unwrap();
        assert_eq!(result.typ, Type::Scalar);
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
        assert!(result.code.contains("mix")); // WGSL uses mix
    }

    #[test]
    fn test_type_mismatch() {
        let result = emit("a + b", &[("a", Type::Scalar), ("b", Type::Vec2)]);
        assert!(matches!(result, Err(WgslError::TypeMismatch { .. })));
    }
}
