//! Cranelift JIT compilation for complex expressions.
//!
//! Compiles typed expressions to native code via Cranelift.
//!
//! # Representation
//!
//! - Scalar: single f32
//! - Complex: two f32 values (real, imag)

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
extern "C" fn math_sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cos(x: f32) -> f32 {
    x.cos()
}
extern "C" fn math_exp(x: f32) -> f32 {
    x.exp()
}
extern "C" fn math_log(x: f32) -> f32 {
    x.ln()
}
extern "C" fn math_atan2(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "complex_sqrt",
            ptr: math_sqrt as *const u8,
        },
        MathSymbol {
            name: "complex_pow",
            ptr: math_pow as *const u8,
        },
        MathSymbol {
            name: "complex_sin",
            ptr: math_sin as *const u8,
        },
        MathSymbol {
            name: "complex_cos",
            ptr: math_cos as *const u8,
        },
        MathSymbol {
            name: "complex_exp",
            ptr: math_exp as *const u8,
        },
        MathSymbol {
            name: "complex_log",
            ptr: math_log as *const u8,
        },
        MathSymbol {
            name: "complex_atan2",
            ptr: math_atan2 as *const u8,
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
    Complex([CraneliftValue; 2]),
}

impl TypedValue {
    fn typ(&self) -> Type {
        match self {
            TypedValue::Scalar(_) => Type::Scalar,
            TypedValue::Complex(_) => Type::Complex,
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
            Type::Complex => 2,
        }
    }
}

// ============================================================================
// Compiled Function
// ============================================================================

/// A compiled complex function that returns a scalar.
pub struct CompiledComplexFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

unsafe impl Send for CompiledComplexFn {}
unsafe impl Sync for CompiledComplexFn {}

impl CompiledComplexFn {
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
                _ => panic!("too many parameters (max 6 for complex)"),
            }
        }
    }
}

// ============================================================================
// Math functions struct
// ============================================================================

struct MathFuncs {
    sqrt: FuncRef,
    pow: FuncRef,
    sin: FuncRef,
    cos: FuncRef,
    exp: FuncRef,
    log: FuncRef,
    atan2: FuncRef,
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for complex expressions.
pub struct ComplexJit {
    builder: JITBuilder,
}

impl ComplexJit {
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
    ) -> Result<CompiledComplexFn, CraneliftError> {
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
            .declare_function("complex_sqrt", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let pow_id = module
            .declare_function("complex_pow", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let sin_id = module
            .declare_function("complex_sin", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let cos_id = module
            .declare_function("complex_cos", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let exp_id = module
            .declare_function("complex_exp", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let log_id = module
            .declare_function("complex_log", Linkage::Import, &sig_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;
        let atan2_id = module
            .declare_function("complex_atan2", Linkage::Import, &sig_f32_f32_f32)
            .map_err(|e| CraneliftError::JitError(e.to_string()))?;

        // Build function signature
        let total_params: usize = vars.iter().map(|v| v.param_count()).sum();
        let mut sig = module.make_signature();
        for _ in 0..total_params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("complex_expr", Linkage::Export, &sig)
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
                sin: module.declare_func_in_func(sin_id, builder.func),
                cos: module.declare_func_in_func(cos_id, builder.func),
                exp: module.declare_func_in_func(exp_id, builder.func),
                log: module.declare_func_in_func(log_id, builder.func),
                atan2: module.declare_func_in_func(atan2_id, builder.func),
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
                    Type::Complex => {
                        let v = TypedValue::Complex([
                            block_params[param_idx],
                            block_params[param_idx + 1],
                        ]);
                        param_idx += 2;
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

        Ok(CompiledComplexFn {
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

        // Complex + Complex
        (BinOp::Add, TypedValue::Complex(l), TypedValue::Complex(r)) => Ok(TypedValue::Complex([
            builder.ins().fadd(l[0], r[0]),
            builder.ins().fadd(l[1], r[1]),
        ])),
        (BinOp::Sub, TypedValue::Complex(l), TypedValue::Complex(r)) => Ok(TypedValue::Complex([
            builder.ins().fsub(l[0], r[0]),
            builder.ins().fsub(l[1], r[1]),
        ])),

        // Complex * Scalar
        (BinOp::Mul, TypedValue::Complex(c), TypedValue::Scalar(s)) => Ok(TypedValue::Complex([
            builder.ins().fmul(c[0], *s),
            builder.ins().fmul(c[1], *s),
        ])),
        (BinOp::Mul, TypedValue::Scalar(s), TypedValue::Complex(c)) => Ok(TypedValue::Complex([
            builder.ins().fmul(*s, c[0]),
            builder.ins().fmul(*s, c[1]),
        ])),

        // Complex * Complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        (BinOp::Mul, TypedValue::Complex(l), TypedValue::Complex(r)) => {
            let ac = builder.ins().fmul(l[0], r[0]);
            let bd = builder.ins().fmul(l[1], r[1]);
            let ad = builder.ins().fmul(l[0], r[1]);
            let bc = builder.ins().fmul(l[1], r[0]);
            Ok(TypedValue::Complex([
                builder.ins().fsub(ac, bd),
                builder.ins().fadd(ad, bc),
            ]))
        }

        // Complex / Scalar
        (BinOp::Div, TypedValue::Complex(c), TypedValue::Scalar(s)) => Ok(TypedValue::Complex([
            builder.ins().fdiv(c[0], *s),
            builder.ins().fdiv(c[1], *s),
        ])),

        // Complex / Complex
        (BinOp::Div, TypedValue::Complex(l), TypedValue::Complex(r)) => {
            // (a+bi)/(c+di) = (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²)i
            let c2 = builder.ins().fmul(r[0], r[0]);
            let d2 = builder.ins().fmul(r[1], r[1]);
            let denom = builder.ins().fadd(c2, d2);

            let ac = builder.ins().fmul(l[0], r[0]);
            let bd = builder.ins().fmul(l[1], r[1]);
            let bc = builder.ins().fmul(l[1], r[0]);
            let ad = builder.ins().fmul(l[0], r[1]);

            let real_num = builder.ins().fadd(ac, bd);
            let imag_num = builder.ins().fsub(bc, ad);

            Ok(TypedValue::Complex([
                builder.ins().fdiv(real_num, denom),
                builder.ins().fdiv(imag_num, denom),
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
            TypedValue::Complex(c) => Ok(TypedValue::Complex([
                builder.ins().fneg(c[0]),
                builder.ins().fneg(c[1]),
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
        "re" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Scalar(c[0])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "im" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Scalar(c[1])),
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "abs" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Scalar(v) => {
                    // For scalar, just compute absolute value via sqrt(x*x)
                    let sq = builder.ins().fmul(*v, *v);
                    let call = builder.ins().call(math.sqrt, &[sq]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                TypedValue::Complex(c) => {
                    // |z| = sqrt(a² + b²)
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    let sum = builder.ins().fadd(a2, b2);
                    let call = builder.ins().call(math.sqrt, &[sum]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
            }
        }

        "arg" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => {
                    let call = builder.ins().call(math.atan2, &[c[1], c[0]]);
                    Ok(TypedValue::Scalar(builder.inst_results(call)[0]))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "norm" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => {
                    // norm(z) = a² + b²
                    let a2 = builder.ins().fmul(c[0], c[0]);
                    let b2 = builder.ins().fmul(c[1], c[1]);
                    Ok(TypedValue::Scalar(builder.ins().fadd(a2, b2)))
                }
                _ => Err(CraneliftError::UnknownFunction(name.to_string())),
            }
        }

        "conj" => {
            if args.len() != 1 {
                return Err(CraneliftError::UnknownFunction(name.to_string()));
            }
            match &args[0] {
                TypedValue::Complex(c) => Ok(TypedValue::Complex([c[0], builder.ins().fneg(c[1])])),
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
        let jit = ComplexJit::new().unwrap();
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
    fn test_complex_abs() {
        let expr = Expr::parse("abs(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        // abs(3+4i) = 5
        assert_eq!(func.call(&[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_complex_re_im() {
        let expr_re = Expr::parse("re(z)").unwrap();
        let expr_im = Expr::parse("im(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func_re = jit
            .compile_scalar(expr_re.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        let jit = ComplexJit::new().unwrap();
        let func_im = jit
            .compile_scalar(expr_im.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();

        assert_eq!(func_re.call(&[3.0, 4.0]), 3.0);
        assert_eq!(func_im.call(&[3.0, 4.0]), 4.0);
    }

    #[test]
    fn test_complex_arg() {
        let expr = Expr::parse("arg(z)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(expr.ast(), &[VarSpec::new("z", Type::Complex)])
            .unwrap();
        // arg(1+1i) = pi/4
        let result = func.call(&[1.0, 1.0]);
        assert!(approx_eq(result, std::f32::consts::FRAC_PI_4));
    }

    #[test]
    fn test_complex_mul() {
        // abs((1+2i) * (3+4i)) = abs(-5+10i) = sqrt(125)
        let expr = Expr::parse("abs(a * b)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let result = func.call(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx_eq(result, (125.0_f32).sqrt()));
    }

    #[test]
    fn test_complex_div() {
        // (4+2i) / (2+0i) = (2+1i), abs = sqrt(5)
        let expr = Expr::parse("abs(a / b)").unwrap();
        let jit = ComplexJit::new().unwrap();
        let func = jit
            .compile_scalar(
                expr.ast(),
                &[
                    VarSpec::new("a", Type::Complex),
                    VarSpec::new("b", Type::Complex),
                ],
            )
            .unwrap();
        let result = func.call(&[4.0, 2.0, 2.0, 0.0]);
        assert!(approx_eq(result, (5.0_f32).sqrt()));
    }
}
