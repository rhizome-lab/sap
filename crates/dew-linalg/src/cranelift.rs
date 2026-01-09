//! Cranelift JIT compilation for linalg expressions.
//!
//! Compiles typed expressions to native code via Cranelift.
//!
//! # Vector Representation
//!
//! Vectors are passed as consecutive f32 parameters:
//! - Vec2: 2 f32 values
//! - Vec3: 3 f32 values
//! - Vec4: 4 f32 values
//!
//! This module currently supports expressions that evaluate to scalars
//! (using dot, length, distance, etc. on vectors).

use crate::Type;
use cranelift_codegen::ir::{AbiParam, FuncRef, InstBuilder, Value as CraneliftValue, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use rhizome_dew_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Errors
// ============================================================================

/// Error during Cranelift compilation.
#[derive(Debug, Clone)]
pub enum CraneliftError {
    UnknownVariable(String),
    UnknownFunction(String),
    TypeMismatch {
        op: &'static str,
        left: Type,
        right: Type,
    },
    UnsupportedReturnType(Type),
    JitError(String),
}

impl std::fmt::Display for CraneliftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CraneliftError::UnknownVariable(name) => write!(f, "unknown variable: '{name}'"),
            CraneliftError::UnknownFunction(name) => write!(f, "unknown function: '{name}'"),
            CraneliftError::TypeMismatch { op, left, right } => {
                write!(f, "type mismatch for {op}: {left} vs {right}")
            }
            CraneliftError::UnsupportedReturnType(t) => {
                write!(f, "unsupported return type: {t} (only scalar supported)")
            }
            CraneliftError::JitError(msg) => write!(f, "JIT error: {msg}"),
        }
    }
}

impl std::error::Error for CraneliftError {}

// ============================================================================
// Math function wrappers
// ============================================================================

extern "C" fn math_sqrt(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn math_pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "linalg_sqrt",
            ptr: math_sqrt as *const u8,
        },
        MathSymbol {
            name: "linalg_pow",
            ptr: math_pow as *const u8,
        },
    ]
}

// ============================================================================
// Typed values during compilation
// ============================================================================

/// A typed value during compilation.
/// Scalars are single CraneliftValue, vectors are multiple CraneliftValues.
#[derive(Clone)]
pub enum TypedValue {
    Scalar(CraneliftValue),
    Vec2([CraneliftValue; 2]),
    #[cfg(feature = "3d")]
    Vec3([CraneliftValue; 3]),
    #[cfg(feature = "4d")]
    Vec4([CraneliftValue; 4]),
}

impl TypedValue {
    fn typ(&self) -> Type {
        match self {
            TypedValue::Scalar(_) => Type::Scalar,
            TypedValue::Vec2(_) => Type::Vec2,
            #[cfg(feature = "3d")]
            TypedValue::Vec3(_) => Type::Vec3,
            #[cfg(feature = "4d")]
            TypedValue::Vec4(_) => Type::Vec4,
        }
    }

    fn as_scalar(&self) -> Option<CraneliftValue> {
        match self {
            TypedValue::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

// ============================================================================
// Variable specification
// ============================================================================

/// Specification of a variable with its type.
#[derive(Debug, Clone)]
pub struct VarSpec {
    pub name: String,
    pub typ: Type,
}

impl VarSpec {
    pub fn new(name: impl Into<String>, typ: Type) -> Self {
        Self {
            name: name.into(),
            typ,
        }
    }

    /// Number of f32 parameters this variable needs.
    pub fn param_count(&self) -> usize {
        match self.typ {
            Type::Scalar => 1,
            Type::Vec2 => 2,
            #[cfg(feature = "3d")]
            Type::Vec3 => 3,
            #[cfg(feature = "4d")]
            Type::Vec4 => 4,
            Type::Mat2 => 4,
            #[cfg(feature = "3d")]
            Type::Mat3 => 9,
            #[cfg(feature = "4d")]
            Type::Mat4 => 16,
        }
    }
}

// ============================================================================
// Compiled Function
// ============================================================================

/// A compiled linalg function.
pub struct CompiledLinalgFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledLinalgFn {}
unsafe impl Sync for CompiledLinalgFn {}

impl CompiledLinalgFn {
    /// Calls the compiled function.
    /// All vector components are flattened into the args array.
    pub fn call(&self, args: &[f32]) -> f32 {
        assert_eq!(args.len(), self.param_count, "wrong number of arguments");

        unsafe {
            match self.param_count {
                0 => {
                    let f: extern "C" fn() -> f32 = std::mem::transmute(self.func_ptr);
                    f()
                }
                1 => {
                    let f: extern "C" fn(f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0])
                }
                2 => {
                    let f: extern "C" fn(f32, f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0], args[1])
                }
                3 => {
                    let f: extern "C" fn(f32, f32, f32) -> f32 = std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2])
                }
                4 => {
                    let f: extern "C" fn(f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3])
                }
                5 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4])
                }
                6 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(args[0], args[1], args[2], args[3], args[4], args[5])
                }
                7 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(
                        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                    )
                }
                8 => {
                    let f: extern "C" fn(f32, f32, f32, f32, f32, f32, f32, f32) -> f32 =
                        std::mem::transmute(self.func_ptr);
                    f(
                        args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                    )
                }
                _ => panic!("too many parameters (max 8 for linalg)"),
            }
        }
    }
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for linalg expressions.
pub struct LinalgJit {
    builder: JITBuilder,
}

impl LinalgJit {
    /// Creates a new JIT compiler.
    pub fn new() -> Result<Self, CraneliftError> {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        for sym in math_symbols() {
            builder.symbol(sym.name, sym.ptr);
        }

        Ok(Self { builder })
    }

    /// Compiles an expression that returns a scalar.
    pub fn compile_scalar(
        self,
        ast: &Ast,
        vars: &[VarSpec],
    ) -> Result<CompiledLinalgFn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sqrt_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let pow_sig = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("linalg_sqrt", Linkage::Import, &sqrt_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("linalg_pow", Linkage::Import, &pow_sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature: all params as f32, returns f32
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("linalg_expr", Linkage::Export, &sig)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        ctx.func.signature = sig;

        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions
            let sqrt_ref = module.declare_func_in_func(sqrt_id, builder.func);
            let pow_ref = module.declare_func_in_func(pow_id, builder.func);

            // Map variables to typed values
            let block_params = builder.block_params(entry_block).to_vec();
            let mut var_map: HashMap<String, TypedValue> = HashMap::new();
            let mut param_idx = 0;

            for var in vars {
                let typed_val = match var.typ {
                    Type::Scalar => {
                        let v = TypedValue::Scalar(block_params[param_idx]);
                        param_idx += 1;
                        v
                    }
                    Type::Vec2 => {
                        let v = TypedValue::Vec2([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
                        v
                    }
                    #[cfg(feature = "3d")]
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    #[cfg(feature = "4d")]
                    Type::Vec4 => {
                        let v = TypedValue::Vec4([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                    _ => return Err(CraneliftError::UnsupportedReturnType(var.typ)),
                };
                var_map.insert(var.name.clone(), typed_val);
            }

            let math_funcs = MathFuncs {
                sqrt: sqrt_ref,
                pow: pow_ref,
            };
            let result = compile_ast(ast, &mut builder, &var_map, &math_funcs)?;

            let scalar = result
                .as_scalar()
                .ok_or(CraneliftError::UnsupportedReturnType(result.typ()))?;
            builder.ins().return_(&[scalar]);
            builder.finalize();
        }

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledLinalgFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }
}

struct MathFuncs {
    sqrt: FuncRef,
    pow: FuncRef,
}

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, TypedValue>,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match ast {
        Ast::Num(n) => Ok(TypedValue::Scalar(builder.ins().f32const(*n))),

        Ast::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| CraneliftError::UnknownVariable(name.clone())),

        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            compile_binop(*op, l, r, builder, math)
        }

        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, math)?;
            compile_unaryop(*op, v, builder)
        }

        Ast::Call(name, args) => {
            let arg_vals: Vec<TypedValue> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, math))
                .collect::<Result<_, _>>()?;
            compile_call(name, arg_vals, builder, math)
        }
    }
}

fn compile_binop(
    op: BinOp,
    left: TypedValue,
    right: TypedValue,
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match (op, &left, &right) {
        // Scalar operations
        (BinOp::Add, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fadd(*l, *r)))
        }
        (BinOp::Sub, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fsub(*l, *r)))
        }
        (BinOp::Mul, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fmul(*l, *r)))
        }
        (BinOp::Div, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            Ok(TypedValue::Scalar(builder.ins().fdiv(*l, *r)))
        }
        (BinOp::Pow, TypedValue::Scalar(l), TypedValue::Scalar(r)) => {
            let call = builder.ins().call(math.pow, &[*l, *r]);
            Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
        }

        // Vec2 + Vec2
        (BinOp::Add, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
        ])),
        (BinOp::Sub, TypedValue::Vec2(l), TypedValue::Vec2(r)) => Ok(TypedValue::Vec2([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
        ])),

        // Vec2 * Scalar
        (BinOp::Mul, TypedValue::Vec2(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec2(v)) => Ok(TypedValue::Vec2([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
        ])),

        // Vec2 / Scalar
        (BinOp::Div, TypedValue::Vec2(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec2([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
        ])),

        #[cfg(feature = "3d")]
        (BinOp::Add, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Sub, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec3(v)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
        ])),
        #[cfg(feature = "3d")]
        (BinOp::Div, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
            builder.ins().fdiv(v[2], *s),
        ])),

        #[cfg(feature = "4d")]
        (BinOp::Add, TypedValue::Vec4(l), TypedValue::Vec4(r)) => Ok(TypedValue::Vec4([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
            builder.ins().fadd(l[3], r[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Sub, TypedValue::Vec4(l), TypedValue::Vec4(r)) => Ok(TypedValue::Vec4([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
            builder.ins().fsub(l[3], r[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Vec4(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec4([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
            builder.ins().fmul(v[3], *s),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec4(v)) => Ok(TypedValue::Vec4([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
            builder.ins().fmul(*s, v[3]),
        ])),
        #[cfg(feature = "4d")]
        (BinOp::Div, TypedValue::Vec4(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec4([
            builder.ins().fdiv(v[0], *s),
            builder.ins().fdiv(v[1], *s),
            builder.ins().fdiv(v[2], *s),
            builder.ins().fdiv(v[3], *s),
        ])),

        _ => Err(CraneliftError::TypeMismatch {
            op: match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            },
            left: left.typ(),
            right: right.typ(),
        }),
    }
}

fn compile_unaryop(
    op: UnaryOp,
    val: TypedValue,
    builder: &mut FunctionBuilder,
) -> Result<TypedValue, CraneliftError> {
    match op {
        UnaryOp::Neg => match val {
            TypedValue::Scalar(v) => Ok(TypedValue::Scalar(builder.ins().fneg(v))),
            TypedValue::Vec2(v) => Ok(TypedValue::Vec2([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
            ])),
            #[cfg(feature = "3d")]
            TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
            ])),
            #[cfg(feature = "4d")]
            TypedValue::Vec4(v) => Ok(TypedValue::Vec4([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
                builder.ins().fneg(v[3]),
            ])),
        },
    }
}

fn compile_call(
    name: &str,
    args: Vec<TypedValue>,
    builder: &mut FunctionBuilder,
    math: &MathFuncs,
) -> Result<TypedValue, CraneliftError> {
    match name {
        "dot" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    Ok(TypedValue::Scalar(builder.ins().fadd(x, y)))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let xy = builder.ins().fadd(x, y);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, z)))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let w = builder.ins().fmul(a[3], b[3]);
                    let xy = builder.ins().fadd(x, y);
                    let zw = builder.ins().fadd(z, w);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, zw)))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "length" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec2(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let sum = builder.ins().fadd(x2, y2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "3d")]
                TypedValue::Vec3(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let xy = builder.ins().fadd(x2, y2);
                    let sum = builder.ins().fadd(xy, z2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "4d")]
                TypedValue::Vec4(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let w2 = builder.ins().fmul(v[3], v[3]);
                    let xy = builder.ins().fadd(x2, y2);
                    let zw = builder.ins().fadd(z2, w2);
                    let sum = builder.ins().fadd(xy, zw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "distance" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec2(a), TypedValue::Vec2(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let sum = builder.ins().fadd(dx2, dy2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "3d")]
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dz = builder.ins().fsub(a[2], b[2]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let dz2 = builder.ins().fmul(dz, dz);
                    let dxy = builder.ins().fadd(dx2, dy2);
                    let sum = builder.ins().fadd(dxy, dz2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                #[cfg(feature = "4d")]
                (TypedValue::Vec4(a), TypedValue::Vec4(b)) => {
                    let dx = builder.ins().fsub(a[0], b[0]);
                    let dy = builder.ins().fsub(a[1], b[1]);
                    let dz = builder.ins().fsub(a[2], b[2]);
                    let dw = builder.ins().fsub(a[3], b[3]);
                    let dx2 = builder.ins().fmul(dx, dx);
                    let dy2 = builder.ins().fmul(dy, dy);
                    let dz2 = builder.ins().fmul(dz, dz);
                    let dw2 = builder.ins().fmul(dw, dw);
                    let dxy = builder.ins().fadd(dx2, dy2);
                    let dzw = builder.ins().fadd(dz2, dw2);
                    let sum = builder.ins().fadd(dxy, dzw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        _ => Err(CraneliftError::UnknownFunction(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_dew_core::Expr;

    #[test]
    fn test_scalar_add() {
        let expr = Expr::parse("a + b").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Scalar),
                    VarSpec::new("b", Type::Scalar),
                ],
            )
            .unwrap();
        assert_eq!(func.call(&[3.0, 4.0]), 7.0);
    }

    #[test]
    fn test_dot_vec2() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        // dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11
        assert_eq!(func.call(&[1.0, 2.0, 3.0, 4.0]), 11.0);
    }

    #[test]
    fn test_length_vec2() {
        let expr = Expr::parse("length(v)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
            .unwrap();
        // length([3, 4]) = 5
        assert_eq!(func.call(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_distance_vec2() {
        let expr = Expr::parse("distance(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        // distance([0, 0], [3, 4]) = 5
        assert_eq!(func.call(&[0.0, 0.0, 3.0, 4.0]), 5.0);
    }

    #[cfg(feature = "3d")]
    #[test]
    fn test_dot_vec3() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec3), VarSpec::new("b", Type::Vec3)],
            )
            .unwrap();
        // dot([1, 2, 3], [4, 5, 6]) = 1*4 + 2*5 + 3*6 = 32
        assert_eq!(func.call(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_complex_expression() {
        // length(a - b) should equal distance(a, b)
        let expr = Expr::parse("length(a - b)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[VarSpec::new("a", Type::Vec2), VarSpec::new("b", Type::Vec2)],
            )
            .unwrap();
        assert_eq!(func.call(&[0.0, 0.0, 3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_vec_scalar_mul() {
        let expr = Expr::parse("length(v * 2)").unwrap();
        let jit = LinalgJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec2)])
            .unwrap();
        // length([3, 4] * 2) = length([6, 8]) = 10
        assert_eq!(func.call(&[3.0, 4.0]), 10.0);
    }
}
