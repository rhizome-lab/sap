# Dew Playground

Interactive web playground for the Dew expression language.

## Development

```bash
cd playground
bun install  # or npm install
bun dev      # or npm run dev
```

Open http://localhost:3000

## Architecture

- **SolidJS** - Reactive UI framework
- **Vite** - Build tooling
- **BEM** - CSS methodology
- **WASM** - Rust→WebAssembly for parsing/codegen

## Status

### Done
- [x] Project setup (Vite + SolidJS)
- [x] Basic layout
- [x] Expression editor
- [x] Collapsible AST viewer
- [x] Tab switching (AST/WGSL/Lua)
- [x] Glassmorphic styling

### TODO
- [ ] WASM bindings for dew-core
- [ ] Real parsing (currently mock)
- [ ] WGSL codegen output
- [ ] Lua codegen output
- [ ] Evaluation runner
- [ ] Feature toggles (cond/func)
- [ ] Variable input UI
- [ ] Domain crate selection

## Building WASM

(TODO: Document wasm-pack setup)

```bash
cd crates/dew-wasm
wasm-pack build --target web
```
