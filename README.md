# Sap

Expression language for procedural generation.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Overview

Sap is a domain-specific expression language designed for procedural content generation. It provides a composable way to define generation rules that can be compiled to multiple backends.

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-sap-core` | Core types and AST |
| `rhizome-sap-scalar` | Standard scalar math functions (sin, cos, etc.) |
| `rhizome-sap-linalg` | Linear algebra types and operations (Vec2, Mat3, etc.) |

Each domain crate (scalar, linalg) includes self-contained backends:
- `wgsl` feature: WGSL shader code generation
- `lua` feature: Lua code generation + mlua execution
- `cranelift` feature: Cranelift JIT native compilation

## Architecture

```
sap-core           # Syntax only: AST, parsing
    |
    +-- sap-scalar     # Scalar domain: f32/f64 math functions
    |                  # Backends: wgsl, lua, cranelift
    |
    +-- sap-linalg     # Linalg domain: Vec2, Vec3, Mat2, Mat3, etc.
                       # Backends: wgsl, lua, cranelift
```

Crates are independent - use one or both. Each has:
- Generic over numeric type `T: Float`
- Own `FunctionRegistry<T>` and `eval<T>()`
- Self-contained backend modules

## License

MIT
