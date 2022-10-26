use crate::front::wgsl::ast;
use crate::{FastHashMap, Handle};

pub struct Index<'a> {
    globals: FastHashMap<&'a str, Handle<ast::GlobalDecl<'a>>>,
}

impl<'a> Index<'a> {
    pub fn generate(tu: &ast::TranslationUnit<'a>) -> Self {
        let mut globals = FastHashMap::default();

        for (handle, decl) in tu.decls.iter() {
            let name = match decl.kind {
                ast::GlobalDeclKind::Fn(ref f) => f.name.name,
                ast::GlobalDeclKind::Var(ref v) => v.name.name,
                ast::GlobalDeclKind::Const(ref c) => c.name.name,
                ast::GlobalDeclKind::Struct(ref s) => s.name.name,
                ast::GlobalDeclKind::Type(ref t) => t.name.name,
            };

            globals.insert(name, handle);
        }

        Self { globals }
    }
}
