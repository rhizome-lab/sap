//! Cranelift JIT compilation for scalar expressions.
//!
//! Compiles expression ASTs to native code via Cranelift.

use cranelift_codegen::ir::{AbiParam, types};
use cranelift_codegen::ir::{InstBuilder, Value as CraneliftValue};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use std::collections::HashMap;

// ============================================================================
// Math function wrappers (extern "C" for Cranelift to call)
// ============================================================================

extern "C" fn math_sin(x: f32) -> f32 {
    x.sin()
}
extern "C" fn math_cos(x: f32) -> f32 {
    x.cos()
}
extern "C" fn math_tan(x: f32) -> f32 {
    x.tan()
}
extern "C" fn math_asin(x: f32) -> f32 {
    x.asin()
}
extern "C" fn math_acos(x: f32) -> f32 {
    x.acos()
}
extern "C" fn math_atan(x: f32) -> f32 {
    x.atan()
}
extern "C" fn math_atan2(y: f32, x: f32) -> f32 {
    y.atan2(x)
}
extern "C" fn math_sinh(x: f32) -> f32 {
    x.sinh()
}
extern "C" fn math_cosh(x: f32) -> f32 {
    x.cosh()
}
extern "C" fn math_tanh(x: f32) -> f32 {
    x.tanh()
}
extern "C" fn math_exp(x: f32) -> f32 {
    x.exp()
}
extern "C" fn math_exp2(x: f32) -> f32 {
    x.exp2()
}
extern "C" fn math_ln(x: f32) -> f32 {
    x.ln()
}
extern "C" fn math_log2(x: f32) -> f32 {
    x.log2()
}
extern "C" fn math_log10(x: f32) -> f32 {
    x.log10()
}
extern "C" fn math_pow(base: f32, exp: f32) -> f32 {
    base.powf(exp)
}
extern "C" fn math_sqrt(x: f32) -> f32 {
    x.sqrt()
}
extern "C" fn math_inversesqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

struct MathSymbol {
    name: &'static str,
    ptr: *const u8,
    arity: usize,
}

fn math_symbols() -> Vec<MathSymbol> {
    vec![
        MathSymbol {
            name: "sap_sin",
            ptr: math_sin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_cos",
            ptr: math_cos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_tan",
            ptr: math_tan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_asin",
            ptr: math_asin as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_acos",
            ptr: math_acos as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_atan",
            ptr: math_atan as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_atan2",
            ptr: math_atan2 as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "sap_sinh",
            ptr: math_sinh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_cosh",
            ptr: math_cosh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_tanh",
            ptr: math_tanh as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_exp",
            ptr: math_exp as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_exp2",
            ptr: math_exp2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_ln",
            ptr: math_ln as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_log2",
            ptr: math_log2 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_log10",
            ptr: math_log10 as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_pow",
            ptr: math_pow as *const u8,
            arity: 2,
        },
        MathSymbol {
            name: "sap_sqrt",
            ptr: math_sqrt as *const u8,
            arity: 1,
        },
        MathSymbol {
            name: "sap_inversesqrt",
            ptr: math_inversesqrt as *const u8,
            arity: 1,
        },
    ]
}

// ============================================================================
// Compiled function
// ============================================================================

/// A compiled function that can be called.
pub struct CompiledFn {
    _module: JITModule,
    func_ptr: *const u8,
    param_count: usize,
}

// SAFETY: The compiled code is self-contained
unsafe impl Send for CompiledFn {}
unsafe impl Sync for CompiledFn {}

impl CompiledFn {
    /// Calls the compiled function with the given arguments.
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
                _ => panic!("too many parameters (max 6)"),
            }
        }
    }
}

// ============================================================================
// JIT Compiler
// ============================================================================

/// JIT compiler for scalar expressions.
pub struct ScalarJit {
    builder: JITBuilder,
}

impl ScalarJit {
    /// Creates a new JIT compiler.
    pub fn new() -> Result<Self, String> {
        let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
            .map_err(|e| e.to_string())?;

        // Register math symbols
        for sym in math_symbols() {
            builder.symbol(sym.name, sym.ptr);
        }

        Ok(Self { builder })
    }

    /// Compiles an expression to a callable function.
    pub fn compile(self, ast: &Ast, params: &[&str]) -> Result<CompiledFn, String> {
        let mut module = JITModule::new(self.builder);
        let mut ctx = module.make_context();

        // Declare math functions
        let math_ids = declare_math_funcs(&mut module)?;

        // Build function signature
        let mut sig = module.make_signature();
        for _ in params {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function("expr", Linkage::Export, &sig)
            .map_err(|e| e.to_string())?;

        ctx.func.signature = sig;

        // Build function body
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Import math functions
            let math_refs = import_math_funcs(&mut builder, &mut module, &math_ids);

            // Map params
            let mut var_map: HashMap<String, CraneliftValue> = HashMap::new();
            for (i, name) in params.iter().enumerate() {
                let val = builder.block_params(entry_block)[i];
                var_map.insert(name.to_string(), val);
            }

            // Compile
            let result = compile_ast(ast, &mut builder, &var_map, &math_refs)?;
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile to machine code
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let func_ptr = module.get_finalized_function(func_id);

        Ok(CompiledFn {
            _module: module,
            func_ptr,
            param_count: params.len(),
        })
    }
}

// ============================================================================
// Math function registration
// ============================================================================

struct DeclaredMathFuncs {
    func_ids: HashMap<String, (FuncId, usize)>,
}

fn declare_math_funcs(module: &mut JITModule) -> Result<DeclaredMathFuncs, String> {
    let mut func_ids = HashMap::new();

    for sym in math_symbols() {
        let mut sig = module.make_signature();
        for _ in 0..sym.arity {
            sig.params.push(AbiParam::new(types::F32));
        }
        sig.returns.push(AbiParam::new(types::F32));

        let func_id = module
            .declare_function(sym.name, Linkage::Import, &sig)
            .map_err(|e| e.to_string())?;

        func_ids.insert(sym.name.to_string(), (func_id, sym.arity));
    }

    Ok(DeclaredMathFuncs { func_ids })
}

struct MathRefs {
    funcs: HashMap<String, cranelift_codegen::ir::FuncRef>,
}

fn import_math_funcs(
    builder: &mut FunctionBuilder,
    module: &mut JITModule,
    declared: &DeclaredMathFuncs,
) -> MathRefs {
    let mut funcs = HashMap::new();

    for (name, (func_id, _)) in &declared.func_ids {
        let func_ref = module.declare_func_in_func(*func_id, builder.func);
        funcs.insert(name.clone(), func_ref);
    }

    MathRefs { funcs }
}

// ============================================================================
// AST compilation
// ============================================================================

fn compile_ast(
    ast: &Ast,
    builder: &mut FunctionBuilder,
    vars: &HashMap<String, CraneliftValue>,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    match ast {
        Ast::Num(n) => Ok(builder.ins().f32const(*n)),

        Ast::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown variable: {}", name)),

        Ast::BinOp(op, left, right) => {
            let l = compile_ast(left, builder, vars, math)?;
            let r = compile_ast(right, builder, vars, math)?;
            Ok(match op {
                BinOp::Add => builder.ins().fadd(l, r),
                BinOp::Sub => builder.ins().fsub(l, r),
                BinOp::Mul => builder.ins().fmul(l, r),
                BinOp::Div => builder.ins().fdiv(l, r),
                BinOp::Pow => {
                    let func_ref = math.funcs.get("sap_pow").ok_or("pow not available")?;
                    let call = builder.ins().call(*func_ref, &[l, r]);
                    builder.inst_results(call)[0]
                }
            })
        }

        Ast::UnaryOp(op, inner) => {
            let v = compile_ast(inner, builder, vars, math)?;
            Ok(match op {
                UnaryOp::Neg => builder.ins().fneg(v),
            })
        }

        Ast::Call(name, args) => {
            let arg_vals: Vec<CraneliftValue> = args
                .iter()
                .map(|a| compile_ast(a, builder, vars, math))
                .collect::<Result<_, _>>()?;

            compile_function(name, &arg_vals, builder, math)
        }
    }
}

fn compile_function(
    name: &str,
    args: &[CraneliftValue],
    builder: &mut FunctionBuilder,
    math: &MathRefs,
) -> Result<CraneliftValue, String> {
    use cranelift_codegen::ir::condcodes::FloatCC;

    Ok(match name {
        // Constants
        "pi" => builder.ins().f32const(std::f32::consts::PI),
        "e" => builder.ins().f32const(std::f32::consts::E),
        "tau" => builder.ins().f32const(std::f32::consts::TAU),

        // Transcendental functions via Rust callbacks
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "exp"
        | "exp2" | "ln" | "log" | "log2" | "log10" | "sqrt" | "inversesqrt" => {
            let sym_name = if name == "log" || name == "ln" {
                "sap_ln".to_string()
            } else {
                format!("sap_{}", name)
            };
            let func_ref = math
                .funcs
                .get(&sym_name)
                .ok_or_else(|| format!("{} not available", name))?;
            let call = builder.ins().call(*func_ref, &[args[0]]);
            builder.inst_results(call)[0]
        }
        "atan2" => {
            let func_ref = math.funcs.get("sap_atan2").ok_or("atan2 not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }
        "pow" => {
            let func_ref = math.funcs.get("sap_pow").ok_or("pow not available")?;
            let call = builder.ins().call(*func_ref, &[args[0], args[1]]);
            builder.inst_results(call)[0]
        }

        // Native IR functions
        "abs" => builder.ins().fabs(args[0]),
        "sign" => {
            let x = args[0];
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let neg_one = builder.ins().f32const(-1.0);
            let gt_zero = builder.ins().fcmp(FloatCC::GreaterThan, x, zero);
            let lt_zero = builder.ins().fcmp(FloatCC::LessThan, x, zero);
            let neg_or_zero = builder.ins().select(lt_zero, neg_one, zero);
            builder.ins().select(gt_zero, one, neg_or_zero)
        }
        "floor" => builder.ins().floor(args[0]),
        "ceil" => builder.ins().ceil(args[0]),
        "round" => builder.ins().nearest(args[0]),
        "trunc" => builder.ins().trunc(args[0]),
        "fract" => {
            let x = args[0];
            let floor_x = builder.ins().floor(x);
            builder.ins().fsub(x, floor_x)
        }
        "min" => builder.ins().fmin(args[0], args[1]),
        "max" => builder.ins().fmax(args[0], args[1]),
        "clamp" => {
            let (x, lo, hi) = (args[0], args[1], args[2]);
            let min_val = builder.ins().fmin(hi, x);
            builder.ins().fmax(lo, min_val)
        }
        "saturate" => {
            let x = args[0];
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let min_val = builder.ins().fmin(one, x);
            builder.ins().fmax(zero, min_val)
        }

        // Interpolation
        "lerp" | "mix" => {
            let (a, b, t) = (args[0], args[1], args[2]);
            let diff = builder.ins().fsub(b, a);
            let scaled = builder.ins().fmul(diff, t);
            builder.ins().fadd(a, scaled)
        }
        "step" => {
            let (edge, x) = (args[0], args[1]);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let cmp = builder.ins().fcmp(FloatCC::LessThan, x, edge);
            builder.ins().select(cmp, zero, one)
        }
        "smoothstep" => {
            let (edge0, edge1, x) = (args[0], args[1], args[2]);
            let zero = builder.ins().f32const(0.0);
            let one = builder.ins().f32const(1.0);
            let two = builder.ins().f32const(2.0);
            let three = builder.ins().f32const(3.0);
            let numer = builder.ins().fsub(x, edge0);
            let denom = builder.ins().fsub(edge1, edge0);
            let t_raw = builder.ins().fdiv(numer, denom);
            let t_min = builder.ins().fmin(one, t_raw);
            let t = builder.ins().fmax(zero, t_min);
            let t2 = builder.ins().fmul(t, t);
            let two_t = builder.ins().fmul(two, t);
            let three_minus = builder.ins().fsub(three, two_t);
            builder.ins().fmul(t2, three_minus)
        }
        "inverse_lerp" => {
            let (a, b, v) = (args[0], args[1], args[2]);
            let numer = builder.ins().fsub(v, a);
            let denom = builder.ins().fsub(b, a);
            builder.ins().fdiv(numer, denom)
        }
        "remap" => {
            let (x, in_lo, in_hi, out_lo, out_hi) = (args[0], args[1], args[2], args[3], args[4]);
            let numer = builder.ins().fsub(x, in_lo);
            let denom = builder.ins().fsub(in_hi, in_lo);
            let t = builder.ins().fdiv(numer, denom);
            let out_range = builder.ins().fsub(out_hi, out_lo);
            let scaled = builder.ins().fmul(out_range, t);
            builder.ins().fadd(out_lo, scaled)
        }

        _ => return Err(format!("unknown function: {}", name)),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhizome_sap_core::Expr;

    fn eval(input: &str, params: &[&str], args: &[f32]) -> f32 {
        let expr = Expr::parse(input).unwrap();
        let jit = ScalarJit::new().unwrap();
        let func = jit.compile(expr.ast(), params).unwrap();
        func.call(args)
    }

    #[test]
    fn test_constants() {
        assert!((eval("pi()", &[], &[]) - std::f32::consts::PI).abs() < 0.001);
        assert!((eval("e()", &[], &[]) - std::f32::consts::E).abs() < 0.001);
        assert!((eval("tau()", &[], &[]) - std::f32::consts::TAU).abs() < 0.001);
    }

    #[test]
    fn test_operators() {
        assert_eq!(eval("x + y", &["x", "y"], &[3.0, 4.0]), 7.0);
        assert_eq!(eval("x * y", &["x", "y"], &[3.0, 4.0]), 12.0);
        assert_eq!(eval("-x", &["x"], &[5.0]), -5.0);
        assert_eq!(eval("x ^ 2", &["x"], &[3.0]), 9.0);
    }

    #[test]
    fn test_trig() {
        assert!(eval("sin(0)", &[], &[]).abs() < 0.001);
        assert!((eval("cos(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!((eval("atan2(1, 1)", &[], &[]) - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    }

    #[test]
    fn test_exp_log() {
        assert!((eval("exp(0)", &[], &[]) - 1.0).abs() < 0.001);
        assert!((eval("exp2(3)", &[], &[]) - 8.0).abs() < 0.001);
        assert!(eval("ln(1)", &[], &[]).abs() < 0.001);
        assert!((eval("log2(8)", &[], &[]) - 3.0).abs() < 0.001);
        assert!((eval("sqrt(16)", &[], &[]) - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_common() {
        assert_eq!(eval("abs(x)", &["x"], &[-5.0]), 5.0);
        assert_eq!(eval("floor(x)", &["x"], &[3.7]), 3.0);
        assert_eq!(eval("ceil(x)", &["x"], &[3.2]), 4.0);
        assert_eq!(eval("min(x, y)", &["x", "y"], &[3.0, 7.0]), 3.0);
        assert_eq!(eval("max(x, y)", &["x", "y"], &[3.0, 7.0]), 7.0);
        assert_eq!(
            eval("clamp(x, lo, hi)", &["x", "lo", "hi"], &[5.0, 0.0, 3.0]),
            3.0
        );
        assert_eq!(eval("saturate(x)", &["x"], &[1.5]), 1.0);
    }

    #[test]
    fn test_interpolation() {
        assert_eq!(
            eval("lerp(a, b, t)", &["a", "b", "t"], &[0.0, 10.0, 0.5]),
            5.0
        );
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.3]), 0.0);
        assert_eq!(eval("step(edge, x)", &["edge", "x"], &[0.5, 0.7]), 1.0);
        assert_eq!(
            eval("inverse_lerp(a, b, v)", &["a", "b", "v"], &[0.0, 10.0, 5.0]),
            0.5
        );
    }

    #[test]
    fn test_remap() {
        let result = eval(
            "remap(x, in_lo, in_hi, out_lo, out_hi)",
            &["x", "in_lo", "in_hi", "out_lo", "out_hi"],
            &[5.0, 0.0, 10.0, 0.0, 100.0],
        );
        assert_eq!(result, 50.0);
    }
}
