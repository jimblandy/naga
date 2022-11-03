use crate::front::wgsl::errors::Error;
use crate::front::wgsl::{ast, Span};
use crate::{FastHashMap, Handle};

pub struct Index<'a> {
    globals: FastHashMap<&'a str, Handle<ast::GlobalDecl<'a>>>,
    dependency_order: Vec<Handle<ast::GlobalDecl<'a>>>,
}

impl<'a> Index<'a> {
    pub fn generate(tu: &ast::TranslationUnit<'a>) -> Result<Self, Error<'a>> {
        let mut globals = FastHashMap::default();

        // Populate dependencies
        for (handle, decl) in tu.decls.iter() {
            let name = decl_ident(decl).name;
            globals.insert(name, handle);
        }

        let len = tu.decls.len();
        let solver = DependencySolver {
            globals: &globals,
            module: tu,
            visited: vec![false; len],
            temp_visited: vec![false; len],
            path: Vec::new(),
            out: Vec::with_capacity(len),
        };
        let dependency_order = solver.solve()?;

        Ok(Self {
            globals,
            dependency_order,
        })
    }

    pub fn visit_ordered(&self) -> impl Iterator<Item = Handle<ast::GlobalDecl<'a>>> + '_ {
        self.dependency_order.iter().copied()
    }
}

pub struct ResolvedDependency<'a> {
    pub decl: Handle<ast::GlobalDecl<'a>>,
    pub usage: Span,
}

struct DependencySolver<'source, 'temp> {
    globals: &'temp FastHashMap<&'source str, Handle<ast::GlobalDecl<'source>>>,
    module: &'temp ast::TranslationUnit<'source>,
    visited: Vec<bool>,
    temp_visited: Vec<bool>,
    path: Vec<ResolvedDependency<'source>>,
    out: Vec<Handle<ast::GlobalDecl<'source>>>,
}

impl<'a> DependencySolver<'a, '_> {
    fn solve(mut self) -> Result<Vec<Handle<ast::GlobalDecl<'a>>>, Error<'a>> {
        for id in self.module.decls.iter().map(|x| x.0) {
            if self.visited[id.index()] {
                continue;
            }

            self.dfs(id)?;
        }

        Ok(self.out)
    }

    fn dfs(&mut self, id: Handle<ast::GlobalDecl<'a>>) -> Result<(), Error<'a>> {
        let decl = &self.module.decls[id];
        let id_usize = id.index();

        if self.visited[id_usize] {
            return Ok(());
        }

        self.temp_visited[id_usize] = true;
        for dep in decl.dependencies.iter() {
            if let Some(&dep_id) = self.globals.get(dep.ident) {
                self.path.push(ResolvedDependency {
                    decl: dep_id,
                    usage: dep.usage.clone(),
                });
                let dep_id_usize = dep_id.index();

                if self.temp_visited[dep_id_usize] {
                    // found a cycle.
                    return if dep_id == id {
                        Err(Error::RecursiveDeclaration {
                            ident: decl_ident(decl).span,
                            usage: dep.usage.clone(),
                        })
                    } else {
                        let start_at = self
                            .path
                            .iter()
                            .rev()
                            .enumerate()
                            .find(|&(_, dep)| dep.decl == dep_id)
                            .map(|x| x.0)
                            .unwrap_or(0);

                        Err(Error::CyclicDeclaration {
                            ident: decl_ident(&self.module.decls[dep_id]).span,
                            path: self.path[start_at..]
                                .iter()
                                .map(|curr_dep| {
                                    let curr_id = curr_dep.decl;
                                    let curr_decl = &self.module.decls[curr_id];

                                    (decl_ident(curr_decl).span, curr_dep.usage.clone())
                                })
                                .collect(),
                        })
                    };
                } else if !self.visited[dep_id_usize] {
                    self.dfs(dep_id)?;
                }

                self.path.pop();
            }

            // Ignore unresolved identifiers, they may be inbuilts.
        }

        self.temp_visited[id_usize] = false;
        self.visited[id_usize] = true;
        self.out.push(id);

        Ok(())
    }
}

fn decl_ident<'a>(decl: &ast::GlobalDecl<'a>) -> ast::Ident<'a> {
    match decl.kind {
        ast::GlobalDeclKind::Fn(ref f) => f.name.clone(),
        ast::GlobalDeclKind::Var(ref v) => v.name.clone(),
        ast::GlobalDeclKind::Const(ref c) => c.name.clone(),
        ast::GlobalDeclKind::Struct(ref s) => s.name.clone(),
        ast::GlobalDeclKind::Type(ref t) => t.name.clone(),
    }
}
