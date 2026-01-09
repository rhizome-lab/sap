# CLAUDE.md

Behavioral rules for Claude Code in this repository.

## Architecture

**Dew** is a minimal expression language. Small, ephemeral, perfectly formed—like a droplet condensed from logic. Just dew it.

Functions + numeric values, compiled to multiple backends (WGSL, Cranelift, Lua).

**Crate structure:**
- `dew-core` - Core AST, types, and expression representation
- `dew-scalar` - Standard scalar math functions (sin, cos, etc.)
- `dew-linalg` - Linear algebra types and operations
- `dew-wgsl` - WGSL code generation backend
- `dew-cranelift` - Cranelift JIT compilation backend
- `dew-lua` - Lua code generation backend

## Core Rule

**Note things down immediately:**
- Bugs/issues -> fix or add to TODO.md
- Design decisions -> docs/ or code comments
- Future work -> TODO.md
- Key insights -> this file

**Triggers:** User corrects you, 2+ failed attempts, "aha" moment, framework quirk discovered -> document before proceeding.

**Don't say these (edit first):** "Fair point", "Should have", "That should go in X" -> edit the file BEFORE responding.

**Do the work properly.** When asked to analyze X, actually read X - don't synthesize from conversation. The cost of doing it right < redoing it.

**If citing CLAUDE.md after failing:** The file failed its purpose. Adjust it to actually prevent the failure.

## Negative Constraints

Do not:
- Announce actions ("I will now...") - just do them
- Leave work uncommitted
- Create special cases - design to avoid them
- Create legacy APIs - one API, update all callers
- Add to the monolith - split by domain into sub-crates
- Do half measures - migrate ALL callers when adding abstraction
- Ask permission when philosophy is clear - just do it
- Return tuples - use structs with named fields
- Mark as done prematurely - note what remains
- Fear "over-modularization" - 100 lines is fine for a module
- Consider time constraints - we're NOT short on time; optimize for correctness

## Design Principles

**Unify, don't multiply.** One interface for multiple cases > separate interfaces. Plugin systems > hardcoded switches.

**Simplicity over cleverness.** Functions > traits until you need the trait. Use ecosystem tooling over hand-rolling.

**Explicit over implicit.** Log when skipping. Show what's at stake before refusing.

**When stuck (2+ attempts):** Step back. Am I solving the right problem?
