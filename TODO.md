# TODO

## Backlog

### Port from resin-expr

- [x] Port `resin-expr` core expression types and AST
- [x] Port `resin-expr-std` standard library functions (excluding noise)

### Core

- [x] Define expression AST (functions, numeric values)

Note: No type system in core. Core = syntax only, domains = semantics.
Each domain crate handles its own types during eval/emit.

### Backends (now self-contained in domain crates)

- [x] sap-scalar: WGSL, Lua, Cranelift backends (self-contained)
- [x] sap-linalg: WGSL, Lua, Cranelift backends (self-contained)

Note: Old standalone backend crates (sap-wgsl, sap-lua, sap-cranelift) removed.
Each domain crate now has self-contained backends behind feature flags.

### Standard Library (sap-scalar)

- [x] Generic over `T: Float` (works with f32, f64)
- [x] Own `ScalarFn<T>` trait and `FunctionRegistry<T>`
- [x] Own `eval<T>()` function
- [x] All standard functions: trig, exp/log, common math, interpolation
- [x] WGSL backend (feature = "wgsl")
- [x] Lua backend with mlua execution (feature = "lua")
- [x] Cranelift JIT backend (feature = "cranelift")

### Infrastructure

- [x] Set up CI tests for all backends
- [x] Add integration tests (parity tests across backends)
- [x] Exhaustive test matrix for all functions and operations across all backends
- [ ] Documentation examples

## In Progress

### Linear Algebra (sap-linalg)

Design doc: `docs/linalg-design.md`

Architecture decision: **core = syntax only, domains = semantics**.

#### Design Decisions (resolved)

- [x] Type checking: no separate pass, types propagate during eval/emit
- [x] Function dispatch: runtime dispatch on `Value` variants
- [x] AST changes: none needed, core stays untyped
- [x] Literals: f32, type comes from context

#### Implementation Plan
- [x] Core scaffold: Value<T>, Type, eval, ops
- [x] Vec2, Mat2 types and operations
- [x] Vec3, Mat3 types and operations (feature = "3d", default)
- [x] Vec4, Mat4 (feature = "4d")
- [x] Operator dispatch (`*` for scale, matmul, etc.)
- [x] Generic over numeric type (f32, f64 via num-traits)
- [x] vec * mat (row vector convention) in addition to mat * vec
- [x] Common linalg functions: dot, cross, length, normalize, distance, reflect, hadamard, lerp, mix
- [x] Backend implementations (WGSL, Lua, Cranelift)

#### Future Extensions (separate crates probably)

- Complex numbers (2D rotations)
- Quaternions (3D rotations)
- Dual numbers (autodiff)
- Rotors/spinors (geometric algebra)

#### Nice to Have (maybe)

- Expression normalization/simplification (constant folding, algebraic simplification)
  - Would live in domain crates, not core
  - Could be useful for optimization before JIT compilation

## Backlog - Architecture

### Crate Composability

Problem: What if a user wants to use multiple domain crates together? E.g., linalg + rotors in the same expression.

Current state:
- sap-scalar: `T: Float` scalars
- sap-linalg: `Value<T>` enum (Scalar, Vec2, Vec3, Vec4, Mat2, Mat3, Mat4)
- Future crates might add: Complex, Quaternion, Rotor, etc.

Each has its own `FunctionRegistry<T>` and `eval()` function.

Options to investigate:

1. **Dyn traits**: Common value trait, runtime dispatch
2. **Enum composition**: Generate combined value enum with auto-derived From/Into
   ```rust
   // Macro-generated
   enum CombinedValue<T> {
       Scalar(T),
       Vec2([T; 2]),
       Rotor(rotor::Rotor<T>),
       // ...
   }
   ```
3. **Generic over value type**: Each crate is generic over the value abstraction, users compose by providing their own combined type
4. **Extension trait pattern**: Shared base Value in sap-core, domain crates add methods via traits
   - **Problem**: Value must have ALL variants in sap-core upfront. Can't add new variants from external crates.
   - Only viable if sap-core is a monolith knowing all domains. Defeats modularity. **Not recommended.**

Trade-offs:

| Aspect | Option 2 (enum composition) | Option 3 (generic) |
|--------|---------------------------|-------------------|
| Standalone use | Easy: `LinalgValue<f32>` | Awkward: need concrete type |
| Trait bounds | None | Everywhere: `V: LinalgValue<T>` |
| Conversion cost | Small (enum moves cheap) | Zero |
| Compile time | Lower | Higher (monomorphization) |
| Implementation | From/Into macros | Trait impls for each combo |
| Flexibility | Closed (fixed variants) | Open (any type works) |

**Decision**: Option 3 (generic over value type) for open extension.

#### Implementation Status

Phase 1 (done):
- [x] Define `LinalgValue<T>` trait with construction/extraction methods
- [x] Implement trait for `Value<T>` (default concrete type)
- [x] Export trait from crate

Phase 2 (future, when needed):
- [ ] Make `LinalgFn<T>` generic: `LinalgFn<T, V: LinalgValue<T>>`
- [ ] Make `FunctionRegistry<T>` generic: `FunctionRegistry<T, V>`
- [ ] Make `eval()` generic: `eval<T, V: LinalgValue<T>>()`
- [ ] Update ops.rs to use trait extraction/construction
- [ ] Update funcs.rs to use trait

Current state: The trait is defined and users CAN implement it for their combined types.
But to actually USE combined types with eval/functions, Phase 2 is needed.

Each domain crate:
- Exposes a `DomainValue<T>` trait with construction/extraction methods
- Provides a default concrete type for standalone use (`Value<T>`)
- (Future) Has `eval<T, V: DomainValue<T>>()` generic over value type

Users composing multiple domains:
- Define their own combined enum
- Implement all domain traits for it
- (Future) Both crates work directly with it, zero conversion

Note: Phase 2 is low priority until we have 2+ domain crates that people want to compose.

### External Backend Support

How to create `sap-linalg-glsl` without modifying sap-linalg.

#### Pattern: Backends Don't Need Value, Only Type

Code generation backends (emit) work differently from evaluation:
- **Eval**: needs `Value<T>` to hold actual data
- **Emit**: needs `Type` for type inference, generates string output

Internal backends (wgsl.rs, lua.rs, cranelift.rs) show the pattern:
```rust
// Only needs Type, not Value
pub fn emit_wgsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<WgslExpr, WgslError>
```

#### Creating an External Backend

Example: `sap-linalg-glsl` crate

```toml
[dependencies]
rhizome-sap-core = "..."   # For Ast
rhizome-sap-linalg = "..." # For Type enum
```

```rust
use rhizome_sap_core::{Ast, BinOp, UnaryOp};
use rhizome_sap_linalg::Type;

pub fn emit_glsl(ast: &Ast, var_types: &HashMap<String, Type>) -> Result<GlslExpr, GlslError> {
    match ast {
        Ast::Num(n) => /* ... */,
        Ast::Var(name) => /* lookup type from var_types */,
        Ast::BinOp(op, left, right) => {
            let left_expr = emit_glsl(left, var_types)?;
            let right_expr = emit_glsl(right, var_types)?;
            // Infer result type based on operand types
            // Generate GLSL-specific code
        }
        Ast::Call(name, args) => {
            // Map function names to GLSL equivalents
            // e.g., "lerp" -> "mix", "hadamard" -> element-wise multiply
        }
    }
}
```

#### What Domain Crates Must Expose

- [x] Public `Type` enum with public variants
- [x] Public `Value<T>` enum (for users who want both eval and emit in one project)
- [ ] Shared type inference rules (currently duplicated in each backend)
  - Consider: `pub fn infer_binop_type(op, left, right) -> Result<Type, TypeError>`
  - Would reduce duplication and ensure consistency

#### Testing External Backends

Pattern: parity tests against eval results
```rust
// Parse, emit to your backend, execute, compare with eval()
let expr = Expr::parse("dot(a, b)").unwrap();
let glsl_code = emit_glsl(expr.ast(), &var_types)?;
let glsl_result = execute_glsl(&glsl_code, &values)?;
let eval_result = eval(expr.ast(), &values, &registry)?;
assert_close(glsl_result, eval_result);
```
