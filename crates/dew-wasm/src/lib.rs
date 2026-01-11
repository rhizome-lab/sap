//! WebAssembly bindings for Dew expression language.
//!
//! Provides parsing and code generation for use in web browsers.

use rhizome_dew_core::Expr;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// AST node representation for JavaScript.
#[derive(Serialize)]
pub struct JsAstNode {
    #[serde(rename = "type")]
    node_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    children: Option<Vec<JsAstNode>>,
}

/// Parse result for JavaScript.
#[derive(Serialize)]
pub struct JsParseResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    ast: Option<JsAstNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Code generation result for JavaScript.
#[derive(Serialize)]
pub struct JsCodeResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Parse a dew expression and return the AST as a JavaScript object.
#[wasm_bindgen]
pub fn parse(input: &str) -> JsValue {
    let result = match Expr::parse(input) {
        Ok(expr) => JsParseResult {
            ok: true,
            ast: Some(ast_to_js(expr.ast())),
            error: None,
        },
        Err(e) => JsParseResult {
            ok: false,
            ast: None,
            error: Some(e.to_string()),
        },
    };

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

/// Convert AST to JavaScript-friendly representation.
fn ast_to_js(ast: &rhizome_dew_core::Ast) -> JsAstNode {
    use rhizome_dew_core::Ast;

    match ast {
        Ast::Num(n) => JsAstNode {
            node_type: "Num".to_string(),
            value: Some(n.to_string()),
            children: None,
        },
        Ast::Var(name) => JsAstNode {
            node_type: "Var".to_string(),
            value: Some(name.clone()),
            children: None,
        },
        Ast::BinOp(op, left, right) => JsAstNode {
            node_type: "BinOp".to_string(),
            value: Some(format!("{:?}", op)),
            children: Some(vec![ast_to_js(left), ast_to_js(right)]),
        },
        Ast::UnaryOp(op, inner) => JsAstNode {
            node_type: "UnaryOp".to_string(),
            value: Some(format!("{:?}", op)),
            children: Some(vec![ast_to_js(inner)]),
        },
        Ast::Call(name, args) => JsAstNode {
            node_type: "Call".to_string(),
            value: Some(name.clone()),
            children: Some(args.iter().map(ast_to_js).collect()),
        },
        Ast::Compare(op, left, right) => JsAstNode {
            node_type: "Compare".to_string(),
            value: Some(format!("{:?}", op)),
            children: Some(vec![ast_to_js(left), ast_to_js(right)]),
        },
        Ast::And(left, right) => JsAstNode {
            node_type: "And".to_string(),
            value: None,
            children: Some(vec![ast_to_js(left), ast_to_js(right)]),
        },
        Ast::Or(left, right) => JsAstNode {
            node_type: "Or".to_string(),
            value: None,
            children: Some(vec![ast_to_js(left), ast_to_js(right)]),
        },
        Ast::If(cond, then_branch, else_branch) => JsAstNode {
            node_type: "If".to_string(),
            value: None,
            children: Some(vec![
                ast_to_js(cond),
                ast_to_js(then_branch),
                ast_to_js(else_branch),
            ]),
        },
    }
}

/// Generate WGSL code from an expression.
#[wasm_bindgen]
pub fn emit_wgsl(input: &str) -> JsValue {
    use rhizome_dew_scalar::wgsl;

    let result = match Expr::parse(input) {
        Ok(expr) => match wgsl::emit_wgsl(expr.ast()) {
            Ok(wgsl_expr) => JsCodeResult {
                ok: true,
                code: Some(wgsl_expr.code),
                error: None,
            },
            Err(e) => JsCodeResult {
                ok: false,
                code: None,
                error: Some(e.to_string()),
            },
        },
        Err(e) => JsCodeResult {
            ok: false,
            code: None,
            error: Some(e.to_string()),
        },
    };

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

/// Generate Lua code from an expression.
#[wasm_bindgen]
pub fn emit_lua(input: &str) -> JsValue {
    use rhizome_dew_scalar::lua;

    let result = match Expr::parse(input) {
        Ok(expr) => match lua::emit_lua(expr.ast()) {
            Ok(lua_expr) => JsCodeResult {
                ok: true,
                code: Some(lua_expr.code),
                error: None,
            },
            Err(e) => JsCodeResult {
                ok: false,
                code: None,
                error: Some(e.to_string()),
            },
        },
        Err(e) => JsCodeResult {
            ok: false,
            code: None,
            error: Some(e.to_string()),
        },
    };

    serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
}

// Tests are run via wasm-bindgen-test in the browser or node environment
// since they depend on JS interop that can't work on native targets
