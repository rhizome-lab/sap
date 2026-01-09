//! Cranelift JIT compilation for quaternion expressions.
//!
//! Compiles typed expressions to native code via Cranelift.
//!
//! # Representation
//!
//! - Scalar: single f32
//! - Vec3: three f32 values (x, y, z)
//! - Quaternion: four f32 values (x, y, z, w)

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
                write!(f, "unsupported return type: {t}")
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
extern "C" fn math_acos(x: f32) -> f32 {
    x.acos()
}
extern "C" fn math_sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cos(x: f32) -> f32 {
    x.cos()
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "quat_sqrt",
            ptr: math_sqrt as *const u8,
        },
        MathSymbol {
            name: "quat_pow",
            ptr: math_pow as *const u8,
        },
        MathSymbol {
            name: "quat_acos",
            ptr: math_acos as *const u8,
        },
        MathSymbol {
            name: "quat_sin",
            ptr: math_sin as *const u8,
        },
        MathSymbol {
            name: "quat_cos",
            ptr: math_cos as *const u8,
        },
    ]
}

// ============================================================================
// Typed values during compilation
// ============================================================================

/// A typed value during compilation.
#[derive(Clone)]
pub enum TypedValue {
    Scalar(CraneliftValue),
    Vec3([CraneliftValue; 3]),
    Quaternion([CraneliftValue; 4]),
}

impl TypedValue {
    fn typ(&self) -> Type {
        match self {
            TypedValue::Scalar(_) => Type::Scalar,
            TypedValue::Vec3(_) => Type::Vec3,
            TypedValue::Quaternion(_) => Type::Quaternion,
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
            Type::Vec3 => 3,
            Type::Quaternion => 4,
        }
    }
}

// ============================================================================
// Compiled Function
// ============================================================================

/// A compiled quaternion function that returns a scalar.
pub struct CompiledQuaternionFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledQuaternionFn {}
unsafe impl Sync for CompiledQuaternionFn {}

impl CompiledQuaternionFn {
    /// Calls the compiled function.
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
                _ => panic!("too many parameters (max 8 for quaternion)"),
            }
        }
    }
}

// ============================================================================
// Math functions struct
// ============================================================================

#[allow(dead_code)]
struct MathFuncs {
    sqrt: FuncRef,
    pow: FuncRef,
    acos: FuncRef,
    sin: FuncRef,
    cos: FuncRef,
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for quaternion expressions.
pub struct QuaternionJit {
    builder: JITBuilder,
}

impl QuaternionJit {
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
    ) -> Result<CompiledQuaternionFn, CraneliftError> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let sig_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };
        let sig_f32_f32_f32 = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::F32));
            sig.params.push(AbiParam::new(types::F32));
            sig.returns.push(AbiParam::new(types::F32));
            sig
        };

        let sqrt_id = module
            .declare_function("quat_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("quat_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let acos_id = module
            .declare_function("quat_acos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("quat_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("quat_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("quat_expr", Linkage::Export, &sig)
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
            let math_funcs = MathFuncs {
                sqrt: module.declare_func_in_func(sqrt_id, builder.func),
                pow: module.declare_func_in_func(pow_id, builder.func),
                acos: module.declare_func_in_func(acos_id, builder.func),
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
            };

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
                    Type::Vec3 => {
                        let v = TypedValue::Vec3([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                        ]);
                        param_idx += 3;
                        v
                    }
                    Type::Quaternion => {
                        let v = TypedValue::Quaternion([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                            block_params[param_idx + 2],
                            block_params[param_idx + 3],
                        ]);
                        param_idx += 4;
                        v
                    }
                };
                var_map.insert(var.name.clone(), typed_val);
            }

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

        Ok(CompiledQuaternionFn {
            _module: module,
            func_ptr,
            param_count: total_params,
        })
    }
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

        // Vec3 + Vec3
        (BinOp::Add, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
            builder.ins().fadd(l[2], r[2]),
        ])),
        (BinOp::Sub, TypedValue::Vec3(l), TypedValue::Vec3(r)) => Ok(TypedValue::Vec3([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
            builder.ins().fsub(l[2], r[2]),
        ])),

        // Vec3 * Scalar
        (BinOp::Mul, TypedValue::Vec3(v), TypedValue::Scalar(s)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(v[0], *s),
            builder.ins().fmul(v[1], *s),
            builder.ins().fmul(v[2], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Vec3(v)) => Ok(TypedValue::Vec3([
            builder.ins().fmul(*s, v[0]),
            builder.ins().fmul(*s, v[1]),
            builder.ins().fmul(*s, v[2]),
        ])),

        // Quaternion + Quaternion
        (BinOp::Add, TypedValue::Quaternion(l), TypedValue::Quaternion(r)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fadd(l[0], r[0]),
                builder.ins().fadd(l[1], r[1]),
                builder.ins().fadd(l[2], r[2]),
                builder.ins().fadd(l[3], r[3]),
            ]))
        }
        (BinOp::Sub, TypedValue::Quaternion(l), TypedValue::Quaternion(r)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fsub(l[0], r[0]),
                builder.ins().fsub(l[1], r[1]),
                builder.ins().fsub(l[2], r[2]),
                builder.ins().fsub(l[3], r[3]),
            ]))
        }

        // Quaternion * Scalar
        (BinOp::Mul, TypedValue::Quaternion(q), TypedValue::Scalar(s)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fmul(q[0], *s),
                builder.ins().fmul(q[1], *s),
                builder.ins().fmul(q[2], *s),
                builder.ins().fmul(q[3], *s),
            ]))
        }
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Quaternion(q)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fmul(*s, q[0]),
                builder.ins().fmul(*s, q[1]),
                builder.ins().fmul(*s, q[2]),
                builder.ins().fmul(*s, q[3]),
            ]))
        }

        // Quaternion * Quaternion (Hamilton product)
        (BinOp::Mul, TypedValue::Quaternion(a), TypedValue::Quaternion(b)) => {
            let (x1, y1, z1, w1) = (a[0], a[1], a[2], a[3]);
            let (x2, y2, z2, w2) = (b[0], b[1], b[2], b[3]);

            // x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            let wx2 = builder.ins().fmul(w1, x2);
            let xw2 = builder.ins().fmul(x1, w2);
            let yz2 = builder.ins().fmul(y1, z2);
            let zy2 = builder.ins().fmul(z1, y2);
            let x_part1 = builder.ins().fadd(wx2, xw2);
            let x_part2 = builder.ins().fsub(yz2, zy2);
            let new_x = builder.ins().fadd(x_part1, x_part2);

            // y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            let wy2 = builder.ins().fmul(w1, y2);
            let xz2 = builder.ins().fmul(x1, z2);
            let yw2 = builder.ins().fmul(y1, w2);
            let zx2 = builder.ins().fmul(z1, x2);
            let y_part1 = builder.ins().fsub(wy2, xz2);
            let y_part2 = builder.ins().fadd(yw2, zx2);
            let new_y = builder.ins().fadd(y_part1, y_part2);

            // z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            let wz2 = builder.ins().fmul(w1, z2);
            let xy2 = builder.ins().fmul(x1, y2);
            let yx2 = builder.ins().fmul(y1, x2);
            let zw2 = builder.ins().fmul(z1, w2);
            let z_part1 = builder.ins().fadd(wz2, xy2);
            let z_part2 = builder.ins().fsub(zw2, yx2);
            let new_z = builder.ins().fadd(z_part1, z_part2);

            // w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            let ww2 = builder.ins().fmul(w1, w2);
            let xx2 = builder.ins().fmul(x1, x2);
            let yy2 = builder.ins().fmul(y1, y2);
            let zz2 = builder.ins().fmul(z1, z2);
            let w_part1 = builder.ins().fsub(ww2, xx2);
            let w_part2 = builder.ins().fadd(yy2, zz2);
            let new_w = builder.ins().fsub(w_part1, w_part2);

            Ok(TypedValue::Quaternion([new_x, new_y, new_z, new_w]))
        }

        // Quaternion / Scalar
        (BinOp::Div, TypedValue::Quaternion(q), TypedValue::Scalar(s)) => {
            Ok(TypedValue::Quaternion([
                builder.ins().fdiv(q[0], *s),
                builder.ins().fdiv(q[1], *s),
                builder.ins().fdiv(q[2], *s),
                builder.ins().fdiv(q[3], *s),
            ]))
        }

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
            TypedValue::Vec3(v) => Ok(TypedValue::Vec3([
                builder.ins().fneg(v[0]),
                builder.ins().fneg(v[1]),
                builder.ins().fneg(v[2]),
            ])),
            TypedValue::Quaternion(q) => Ok(TypedValue::Quaternion([
                builder.ins().fneg(q[0]),
                builder.ins().fneg(q[1]),
                builder.ins().fneg(q[2]),
                builder.ins().fneg(q[3]),
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
        "length" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Vec3(v) => {
                    let x2 = builder.ins().fmul(v[0], v[0]);
                    let y2 = builder.ins().fmul(v[1], v[1]);
                    let z2 = builder.ins().fmul(v[2], v[2]);
                    let xy = builder.ins().fadd(x2, y2);
                    let sum = builder.ins().fadd(xy, z2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Quaternion(q) => {
                    let x2 = builder.ins().fmul(q[0], q[0]);
                    let y2 = builder.ins().fmul(q[1], q[1]);
                    let z2 = builder.ins().fmul(q[2], q[2]);
                    let w2 = builder.ins().fmul(q[3], q[3]);
                    let xy = builder.ins().fadd(x2, y2);
                    let zw = builder.ins().fadd(z2, w2);
                    let sum = builder.ins().fadd(xy, zw);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "dot" => {
            if args.len() != 2 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match (&args[0], &args[1]) {
                (TypedValue::Vec3(a), TypedValue::Vec3(b)) => {
                    let x = builder.ins().fmul(a[0], b[0]);
                    let y = builder.ins().fmul(a[1], b[1]);
                    let z = builder.ins().fmul(a[2], b[2]);
                    let xy = builder.ins().fadd(x, y);
                    Ok(TypedValue::Scalar(builder.ins().fadd(xy, z)))
                }
                (TypedValue::Quaternion(a), TypedValue::Quaternion(b)) => {
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

        "conj" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Quaternion(q) => Ok(TypedValue::Quaternion([
                    builder.ins().fneg(q[0]),
                    builder.ins().fneg(q[1]),
                    builder.ins().fneg(q[2]),
                    q[3],
                ])),
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

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn test_scalar_add() {
        let expr = Expr::parse("a + b").unwrap();
        let jit = QuaternionJit::new().unwrap();
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
    fn test_quaternion_length() {
        let expr = Expr::parse("length(q)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("q", Type::Quaternion)])
            .unwrap();
        // length([0, 0, 0, 1]) = 1
        assert!(approx_eq(func.call(&[0.0, 0.0, 0.0, 1.0]), 1.0));
        // length([1, 2, 2, 0]) = 3
        assert!(approx_eq(func.call(&[1.0, 2.0, 2.0, 0.0]), 3.0));
    }

    #[test]
    fn test_vec3_length() {
        let expr = Expr::parse("length(v)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("v", Type::Vec3)])
            .unwrap();
        // length([3, 4, 0]) = 5
        assert!(approx_eq(func.call(&[3.0, 4.0, 0.0]), 5.0));
    }

    #[test]
    fn test_quaternion_dot() {
        let expr = Expr::parse("dot(a, b)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Quaternion),
                    VarSpec::new("b", Type::Quaternion),
                ],
            )
            .unwrap();
        // dot([1,0,0,0], [1,0,0,0]) = 1
        assert!(approx_eq(
            func.call(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            1.0
        ));
    }

    #[test]
    fn test_quaternion_mul_identity() {
        // q * identity, then take length (should preserve length)
        let expr = Expr::parse("length(q * identity)").unwrap();
        let jit = QuaternionJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("q", Type::Quaternion),
                    VarSpec::new("identity", Type::Quaternion),
                ],
            )
            .unwrap();
        // q=[1,2,2,0], identity=[0,0,0,1], |q|=3
        assert!(approx_eq(
            func.call(&[1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            3.0
        ));
    }
}
