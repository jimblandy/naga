use crate::front::wgsl::ast;
use crate::front::wgsl::ast::{GlobalDeclKind, TranslationUnit, TypeKind};
use crate::front::wgsl::errors::{Error, ExpectedToken};
use crate::front::wgsl::index::Index;
use crate::front::wgsl::number::Number;
use crate::proc::{Alignment, Layouter};
use crate::{Arena, FastHashMap, Handle, Span};

enum GlobalDecl {
    Fn(Handle<crate::Function>),
    Var(Handle<crate::GlobalVariable>),
    Const(Handle<crate::Constant>),
    Type(Handle<crate::Type>),
    EntryPoint,
}

struct OutputContext<'source, 'temp, 'out> {
    global_expressions: &'temp Arena<ast::Expression<'source>>,
    globals: &'temp FastHashMap<&'source str, GlobalDecl>,
    module: &'out mut crate::Module,
}

impl<'source> OutputContext<'source, '_, '_> {
    fn reborrow(&mut self) -> OutputContext<'source, '_, '_> {
        OutputContext {
            global_expressions: self.global_expressions,
            globals: self.globals,
            module: self.module,
        }
    }
}

pub struct Lowerer<'source, 'temp> {
    index: &'temp Index<'source>,
    layouter: Layouter,
}

impl<'source, 'temp> Lowerer<'source, 'temp> {
    pub fn new(index: &'temp Index<'source>) -> Self {
        Self {
            index,
            layouter: Layouter::default(),
        }
    }

    pub fn lower(
        &mut self,
        tu: &'temp TranslationUnit<'source>,
    ) -> Result<crate::Module, Error<'source>> {
        let mut module = crate::Module::default();
        let mut globals = FastHashMap::default();

        for decl in self.index.visit_ordered() {
            let span = tu.decls.get_span(decl);
            let decl = &tu.decls[decl];

            match decl.kind {
                GlobalDeclKind::Fn(_) => {}
                GlobalDeclKind::Var(ref v) => {
                    let mut ctx = OutputContext {
                        global_expressions: &tu.global_expressions,
                        globals: &globals,
                        module: &mut module,
                    };

                    let ty = self.resolve_type(&v.ty, ctx.reborrow())?;

                    let init = v
                        .init
                        .map(|init| {
                            self.constant(
                                &ctx.global_expressions[init],
                                ctx.global_expressions.get_span(init),
                                ctx.reborrow(),
                            )
                        })
                        .transpose()?;

                    let handle = ctx.module.global_variables.append(
                        crate::GlobalVariable {
                            name: Some(v.name.name.to_string()),
                            space: v.space,
                            binding: v.binding.clone(),
                            ty,
                            init,
                        },
                        span,
                    );

                    globals.insert(v.name.name, GlobalDecl::Var(handle));
                }
                GlobalDeclKind::Const(ref c) => {
                    let mut ctx = OutputContext {
                        global_expressions: &tu.global_expressions,
                        globals: &globals,
                        module: &mut module,
                    };

                    let inner = self.constant_inner(
                        &ctx.global_expressions[c.init],
                        ctx.global_expressions.get_span(c.init),
                    )?;
                    let inferred_type = match inner {
                        crate::ConstantInner::Scalar { width, value } => self.ensure_type_exists(
                            crate::TypeInner::Scalar {
                                width,
                                kind: value.scalar_kind(),
                            },
                            ctx.reborrow(),
                        ),
                        crate::ConstantInner::Composite { ty, .. } => ty,
                    };

                    let handle = ctx.module.constants.append(
                        crate::Constant {
                            name: Some(c.name.name.to_string()),
                            specialization: None,
                            inner,
                        },
                        span,
                    );

                    let explicit_ty =
                        c.ty.as_ref()
                            .map(|ty| self.resolve_type(ty, ctx.reborrow()))
                            .transpose()?;

                    if let Some(explicit) = explicit_ty {
                        if explicit != inferred_type {
                            return Err(Error::InitializationTypeMismatch(
                                c.name.span.clone(),
                                ctx.module.types[explicit].name.as_ref().unwrap().clone(),
                            ));
                        }
                    }

                    globals.insert(c.name.name, GlobalDecl::Const(handle));
                }
                GlobalDeclKind::Struct(ref s) => {
                    let handle = self.resolve_struct(
                        s,
                        span,
                        OutputContext {
                            global_expressions: &tu.global_expressions,
                            globals: &globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(s.name.name, GlobalDecl::Type(handle));
                }
                GlobalDeclKind::Type(ref alias) => {
                    let ty = self.resolve_type(
                        &alias.ty,
                        OutputContext {
                            global_expressions: &tu.global_expressions,
                            globals: &globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(alias.name.name, GlobalDecl::Type(ty));
                }
            }
        }

        Ok(module)
    }

    fn resolve_struct(
        &mut self,
        s: &ast::Struct<'source>,
        span: Span,
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Type>, Error<'source>> {
        let mut offset = 0;
        let mut struct_alignment = Alignment::ONE;
        let mut members = Vec::with_capacity(s.members.len());

        for member in s.members.iter() {
            let ty = self.resolve_type(&member.ty, ctx.reborrow())?;

            self.layouter
                .update(&ctx.module.types, &ctx.module.constants)
                .unwrap();

            let member_min_size = self.layouter[ty].size;
            let member_min_alignment = self.layouter[ty].alignment;

            let member_size = if let Some((size, span)) = member.size.clone() {
                if size < member_min_size {
                    return Err(Error::SizeAttributeTooLow(span, member_min_size));
                } else {
                    size
                }
            } else {
                member_min_size
            };

            let member_alignment = if let Some((align, span)) = member.align.clone() {
                if let Some(alignment) = Alignment::new(align) {
                    if alignment < member_min_alignment {
                        return Err(Error::AlignAttributeTooLow(span, member_min_alignment));
                    } else {
                        alignment
                    }
                } else {
                    return Err(Error::NonPowerOfTwoAlignAttribute(span));
                }
            } else {
                member_min_alignment
            };

            offset = member_alignment.round_up(offset);
            struct_alignment = struct_alignment.max(member_alignment);

            let mut binding = member.binding.clone();
            if let Some(ref mut binding) = binding {
                binding.apply_default_interpolation(&ctx.module.types[ty].inner);
            }

            members.push(crate::StructMember {
                name: Some(member.name.name.to_owned()),
                ty,
                binding,
                offset,
            });

            offset += member_size;
        }

        let size = struct_alignment.round_up(offset);
        let inner = crate::TypeInner::Struct {
            members,
            span: size,
        };

        let handle = ctx.module.types.insert(
            crate::Type {
                name: Some(s.name.name.to_string()),
                inner,
            },
            span,
        );
        Ok(handle)
    }

    fn resolve_type(
        &mut self,
        ty: &ast::Type<'source>,
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Type>, Error<'source>> {
        let inner = match ty.kind {
            TypeKind::Scalar { kind, width } => crate::TypeInner::Scalar { kind, width },
            TypeKind::Vector { size, kind, width } => {
                crate::TypeInner::Vector { size, kind, width }
            }
            TypeKind::Matrix {
                rows,
                columns,
                width,
            } => crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            },
            TypeKind::Atomic { kind, width } => crate::TypeInner::Atomic { kind, width },
            TypeKind::Pointer { ref base, space } => {
                let base = self.resolve_type(&base, ctx.reborrow())?;
                crate::TypeInner::Pointer { base, space }
            }
            TypeKind::Array { ref base, size } => {
                let base = self.resolve_type(&base, ctx.reborrow())?;
                self.layouter
                    .update(&ctx.module.types, &ctx.module.constants)
                    .unwrap();

                crate::TypeInner::Array {
                    base,
                    size: match size {
                        ast::ArraySize::Constant(constant) => {
                            let expr = &ctx.global_expressions[constant];
                            let constant = self.constant(
                                expr,
                                ctx.global_expressions.get_span(constant),
                                ctx.reborrow(),
                            )?;
                            crate::ArraySize::Constant(constant)
                        }
                        ast::ArraySize::Dynamic => crate::ArraySize::Dynamic,
                    },
                    stride: self.layouter[base].to_stride(),
                }
            }
            TypeKind::Image {
                dim,
                arrayed,
                class,
            } => crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            },
            TypeKind::Sampler { comparison } => crate::TypeInner::Sampler { comparison },
            TypeKind::BindingArray { ref base, size } => {
                let base = self.resolve_type(&base, ctx.reborrow())?;

                crate::TypeInner::BindingArray {
                    base,
                    size: match size {
                        ast::ArraySize::Constant(constant) => {
                            let expr = &ctx.global_expressions[constant];
                            let constant = self.constant(
                                expr,
                                ctx.global_expressions.get_span(constant),
                                ctx.reborrow(),
                            )?;
                            crate::ArraySize::Constant(constant)
                        }
                        ast::ArraySize::Dynamic => crate::ArraySize::Dynamic,
                    },
                }
            }
            TypeKind::User(ref ident) => {
                return match ctx.globals.get(ident.name) {
                    Some(&GlobalDecl::Type(handle)) => Ok(handle),
                    Some(_) => Err(Error::Unexpected(ident.span.clone(), ExpectedToken::Type)),
                    None => Err(Error::UnknownType(ident.span.clone())),
                }
            }
        };

        Ok(self.ensure_type_exists(inner, ctx))
    }

    fn ensure_type_exists(
        &mut self,
        inner: crate::TypeInner,
        ctx: OutputContext<'source, '_, '_>,
    ) -> Handle<crate::Type> {
        let name = inner.to_wgsl(&ctx.module.types, &ctx.module.constants);
        let ty = ctx.module.types.insert(
            crate::Type {
                inner,
                name: Some(name),
            },
            Span::UNDEFINED,
        );

        ty
    }

    fn constant(
        &mut self,
        expr: &ast::Expression<'source>,
        expr_span: Span,
        ctx: OutputContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Constant>, Error<'source>> {
        let inner = self.constant_inner(expr, expr_span)?;
        let c = ctx.module.constants.append(
            crate::Constant {
                name: None,
                specialization: None,
                inner,
            },
            Span::UNDEFINED,
        );
        Ok(c)
    }

    fn constant_inner(
        &mut self,
        expr: &ast::Expression<'source>,
        expr_span: Span,
    ) -> Result<crate::ConstantInner, Error<'source>> {
        match expr {
            ast::Expression::Literal(ref literal) => {
                let inner = match *literal {
                    ast::Literal::Number(Number::F32(f)) => crate::ConstantInner::Scalar {
                        width: 4,
                        value: crate::ScalarValue::Float(f as _),
                    },
                    ast::Literal::Number(Number::I32(i)) => crate::ConstantInner::Scalar {
                        width: 4,
                        value: crate::ScalarValue::Sint(i as _),
                    },
                    ast::Literal::Number(Number::U32(u)) => crate::ConstantInner::Scalar {
                        width: 4,
                        value: crate::ScalarValue::Uint(u as _),
                    },
                    ast::Literal::Number(_) => {
                        unreachable!("got abstract numeric type when not expected");
                    }
                    ast::Literal::Bool(b) => crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(b),
                    },
                };

                Ok(inner)
            }
            _ => Err(Error::ConstExprUnsupported(expr_span.to_range().unwrap())),
        }
    }
}
