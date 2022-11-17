use crate::{Handle, Span};

use super::{ast, Error, ExpressionContext, Lowerer, OutputContext};

enum ConcreteConstructorHandle {
    PartialVector {
        size: crate::VectorSize,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    PartialArray,
    Type(Handle<crate::Type>),
}

impl ConcreteConstructorHandle {
    fn borrow<'a>(&self, module: &'a crate::Module) -> ConcreteConstructor<'a> {
        match *self {
            Self::PartialVector { size } => ConcreteConstructor::PartialVector { size },
            Self::PartialMatrix { columns, rows } => {
                ConcreteConstructor::PartialMatrix { columns, rows }
            }
            Self::PartialArray => ConcreteConstructor::PartialArray,
            Self::Type(handle) => ConcreteConstructor::Type(handle, &module.types[handle].inner),
        }
    }
}

enum ConcreteConstructor<'a> {
    PartialVector {
        size: crate::VectorSize,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    PartialArray,
    Type(Handle<crate::Type>, &'a crate::TypeInner),
}

impl ConcreteConstructorHandle {
    fn to_error_string(&self, ctx: ExpressionContext) -> String {
        match *self {
            Self::PartialVector { size } => {
                format!("vec{}<?>", size as u32,)
            }
            Self::PartialMatrix { columns, rows } => {
                format!("mat{}x{}<?>", columns as u32, rows as u32,)
            }
            Self::PartialArray => "array<?, ?>".to_string(),
            Self::Type(ty) => ctx.fmt_ty(ty),
        }
    }
}

enum ComponentsHandle {
    None,
    One {
        component: Handle<crate::Expression>,
        span: Span,
        ty: Handle<crate::Type>,
    },
    Many {
        components: Vec<Handle<crate::Expression>>,
        spans: Vec<Span>,
        first_component_ty: Handle<crate::Type>,
    },
}

impl ComponentsHandle {
    fn borrow(self, module: &crate::Module) -> Components {
        match self {
            Self::None => Components::None,
            Self::One {
                component,
                span,
                ty,
            } => Components::One {
                component,
                span,
                ty,
                ty_inner: &module.types[ty].inner,
            },
            Self::Many {
                components,
                spans,
                first_component_ty,
            } => Components::Many {
                components,
                spans,
                first_component_ty_inner: &module.types[first_component_ty].inner,
            },
        }
    }
}

enum Components<'a> {
    None,
    One {
        component: Handle<crate::Expression>,
        span: Span,
        ty: Handle<crate::Type>,
        ty_inner: &'a crate::TypeInner,
    },
    Many {
        components: Vec<Handle<crate::Expression>>,
        spans: Vec<Span>,
        first_component_ty_inner: &'a crate::TypeInner,
    },
}

impl Components<'_> {
    fn into_components_vec(self) -> Vec<Handle<crate::Expression>> {
        match self {
            Self::None => vec![],
            Self::One { component, .. } => vec![component],
            Self::Many { components, .. } => components,
        }
    }
}

impl<'source, 'temp> Lowerer<'source, 'temp> {
    pub(super) fn construct(
        &mut self,
        span: Span,
        constructor: &ast::ConstructorType<'source>,
        ty_span: Span,
        components: &[Handle<ast::Expression<'source>>],
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let mut octx = OutputContext {
            read_expressions: None,
            global_expressions: ctx.global_expressions,
            globals: ctx.globals,
            module: ctx.module,
        };
        let constructor_h = self.constructor(constructor, octx.reborrow())?;

        let components_h = match *components {
            [] => ComponentsHandle::None,
            [component] => {
                let span = ctx.read_expressions.get_span(component);
                let component = self.lower_expression(component, ctx.reborrow())?;
                let ty = ctx.resolve_type(component)?;

                ComponentsHandle::One {
                    component,
                    span,
                    ty,
                }
            }
            [component, ref rest @ ..] => {
                let span = ctx.read_expressions.get_span(component);
                let component = self.lower_expression(component, ctx.reborrow())?;
                let ty = ctx.resolve_type(component)?;

                ComponentsHandle::Many {
                    components: std::iter::once(Ok(component))
                        .chain(
                            rest.iter()
                                .map(|&component| self.lower_expression(component, ctx.reborrow())),
                        )
                        .collect::<Result<_, _>>()?,
                    spans: std::iter::once(span)
                        .chain(
                            rest.iter()
                                .map(|&component| ctx.read_expressions.get_span(component)),
                        )
                        .collect(),
                    first_component_ty: ty,
                }
            }
        };

        let (components, constructor) = (
            components_h.borrow(ctx.module),
            constructor_h.borrow(ctx.module),
        );
        let expr = match (components, constructor) {
            // Empty constructor
            (Components::None, dst_ty) => {
                let ty = match dst_ty {
                    ConcreteConstructor::Type(ty, _) => ty,
                    _ => return Err(Error::TypeNotInferrable(ty_span)),
                };

                return match ctx.create_zero_value_constant(ty) {
                    Some(constant) => {
                        Ok(ctx.interrupt_emitter(crate::Expression::Constant(constant), span))
                    }
                    None => Err(Error::TypeNotConstructible(ty_span)),
                };
            }

            // Scalar constructor & conversion (scalar -> scalar)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(_, &crate::TypeInner::Scalar { kind, width }),
            ) => crate::Expression::As {
                expr: component,
                kind,
                convert: Some(width),
            },

            // Vector conversion (vector -> vector)
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Vector { size: src_size, .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector {
                        size: dst_size,
                        kind: dst_kind,
                        width: dst_width,
                    },
                ),
            ) if dst_size == src_size => crate::Expression::As {
                expr: component,
                kind: dst_kind,
                convert: Some(dst_width),
            },

            // Vector conversion (vector -> vector) - partial
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Vector {
                            size: src_size,
                            kind: src_kind,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::PartialVector { size: dst_size },
            ) if dst_size == src_size => crate::Expression::As {
                expr: component,
                kind: src_kind,
                convert: None,
            },

            // Matrix conversion (matrix -> matrix)
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Matrix {
                            columns: src_columns,
                            rows: src_rows,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns: dst_columns,
                        rows: dst_rows,
                        width: dst_width,
                    },
                ),
            ) if dst_columns == src_columns && dst_rows == src_rows => crate::Expression::As {
                expr: component,
                kind: crate::ScalarKind::Float,
                convert: Some(dst_width),
            },

            // Matrix conversion (matrix -> matrix) - partial
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Matrix {
                            columns: src_columns,
                            rows: src_rows,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::PartialMatrix {
                    columns: dst_columns,
                    rows: dst_rows,
                },
            ) if dst_columns == src_columns && dst_rows == src_rows => crate::Expression::As {
                expr: component,
                kind: crate::ScalarKind::Float,
                convert: None,
            },

            // Vector constructor (splat) - infer type
            (
                Components::One {
                    component,
                    ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::PartialVector { size },
            ) => crate::Expression::Splat {
                size,
                value: component,
            },

            // Vector constructor (splat)
            (
                Components::One {
                    component,
                    ty_inner:
                        &crate::TypeInner::Scalar {
                            kind: src_kind,
                            width: src_width,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector {
                        size,
                        kind: dst_kind,
                        width: dst_width,
                    },
                ),
            ) if dst_kind == src_kind || dst_width == src_width => crate::Expression::Splat {
                size,
                value: component,
            },

            // Vector constructor (by elements)
            (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        &crate::TypeInner::Scalar { kind, width }
                        | &crate::TypeInner::Vector { kind, width, .. },
                    ..
                },
                ConcreteConstructor::PartialVector { size },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        &crate::TypeInner::Scalar { .. } | &crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(_, &crate::TypeInner::Vector { size, width, kind }),
            ) => {
                let inner = crate::TypeInner::Vector { size, kind, width };
                let ty = ctx.ensure_type_exists(inner);
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by elements)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Scalar { width, .. },
                    ..
                },
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
            ) => {
                let vec_ty = ctx.ensure_type_exists(crate::TypeInner::Vector {
                    width,
                    kind: crate::ScalarKind::Float,
                    size: rows,
                });

                let components = components
                    .chunks(rows as usize)
                    .map(|vec_components| {
                        ctx.expressions.append(
                            crate::Expression::Compose {
                                ty: vec_ty,
                                components: Vec::from(vec_components),
                            },
                            Default::default(),
                        )
                    })
                    .collect();

                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by columns)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Vector { width, .. },
                    ..
                },
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: &crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
            ) => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
                crate::Expression::Compose { ty, components }
            }

            // Array constructor - infer type
            (components, ConcreteConstructor::PartialArray) => {
                let components = components.into_components_vec();

                let base = ctx.resolve_type(components[0])?;

                let size = crate::Constant {
                    name: None,
                    specialization: None,
                    inner: crate::ConstantInner::Scalar {
                        width: 4,
                        value: crate::ScalarValue::Uint(components.len() as _),
                    },
                };

                let inner = crate::TypeInner::Array {
                    base,
                    size: crate::ArraySize::Constant(
                        ctx.module.constants.fetch_or_append(size, Span::UNDEFINED),
                    ),
                    stride: {
                        self.layouter
                            .update(&ctx.module.types, &ctx.module.constants)
                            .unwrap();
                        self.layouter[base].to_stride()
                    },
                };
                let ty = ctx.ensure_type_exists(inner);

                crate::Expression::Compose { ty, components }
            }

            // Array constructor
            (components, ConcreteConstructor::Type(ty, &crate::TypeInner::Array { .. })) => {
                let components = components.into_components_vec();
                crate::Expression::Compose { ty, components }
            }

            // Struct constructor
            (components, ConcreteConstructor::Type(ty, &crate::TypeInner::Struct { .. })) => {
                crate::Expression::Compose {
                    ty,
                    components: components.into_components_vec(),
                }
            }

            // ERRORS

            // Bad conversion (type cast)
            (
                Components::One {
                    span, ty: src_ty, ..
                },
                _,
            ) => {
                let from_type = ctx.fmt_ty(src_ty);
                return Err(Error::BadTypeCast {
                    span,
                    from_type,
                    to_type: constructor_h.to_error_string(ctx.reborrow()),
                });
            }

            // Too many parameters for scalar constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(_, &crate::TypeInner::Scalar { .. }),
            ) => {
                let span = spans[1].until(spans.last().unwrap());
                return Err(Error::UnexpectedComponents(span));
            }

            // Parameters are of the wrong type for vector or matrix constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(
                    _,
                    &crate::TypeInner::Vector { .. } | &crate::TypeInner::Matrix { .. },
                )
                | ConcreteConstructor::PartialVector { .. }
                | ConcreteConstructor::PartialMatrix { .. },
            ) => {
                return Err(Error::InvalidConstructorComponentType(spans[0], 0));
            }

            // Other types can't be constructed
            _ => return Err(Error::TypeNotConstructible(ty_span)),
        };

        let expr = ctx.expressions.append(expr, span);
        Ok(expr)
    }

    pub(super) fn const_construct(
        &mut self,
        span: Span,
        constructor: &ast::ConstructorType<'source>,
        components: &[Handle<ast::Expression<'source>>],
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<crate::ConstantInner, Error<'source>> {
        // TODO: Support zero values, splatting and inference.

        let constructor = self.constructor(constructor, ctx.reborrow())?;

        let c = match constructor {
            ConcreteConstructorHandle::Type(ty) => {
                let components = components
                    .iter()
                    .map(|&expr| {
                        let arena = match ctx.read_expressions {
                            Some(arena) => arena,
                            None => ctx.global_expressions,
                        };
                        self.constant(&arena[expr], arena.get_span(expr), ctx.reborrow())
                    })
                    .collect::<Result<_, _>>()?;

                crate::ConstantInner::Composite { ty, components }
            }
            _ => return Err(Error::ConstExprUnsupported(span)),
        };
        Ok(c)
    }

    fn constructor<'out>(
        &mut self,
        constructor: &ast::ConstructorType<'source>,
        mut ctx: OutputContext<'source, '_, 'out>,
    ) -> Result<ConcreteConstructorHandle, Error<'source>> {
        let c = match *constructor {
            ast::ConstructorType::Scalar { width, kind } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Scalar { width, kind });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialVector { size } => {
                ConcreteConstructorHandle::PartialVector { size }
            }
            ast::ConstructorType::Vector { size, kind, width } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Vector { size, kind, width });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialMatrix { rows, columns } => {
                ConcreteConstructorHandle::PartialMatrix { rows, columns }
            }
            ast::ConstructorType::Matrix {
                rows,
                columns,
                width,
            } => {
                let ty = ctx.ensure_type_exists(crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::PartialArray => ConcreteConstructorHandle::PartialArray,
            ast::ConstructorType::Array { ref base, size } => {
                let base = self.resolve_type(base, ctx.reborrow())?;
                let size = match size {
                    ast::ArraySize::Constant(expr) => {
                        let span = ctx.global_expressions.get_span(expr);
                        let expr = &ctx.global_expressions[expr];
                        crate::ArraySize::Constant(self.constant(expr, span, ctx.reborrow())?)
                    }
                    ast::ArraySize::Dynamic => crate::ArraySize::Dynamic,
                };

                self.layouter
                    .update(&ctx.module.types, &ctx.module.constants)
                    .unwrap();
                let ty = ctx.ensure_type_exists(crate::TypeInner::Array {
                    base,
                    size,
                    stride: self.layouter[base].to_stride(),
                });
                ConcreteConstructorHandle::Type(ty)
            }
            ast::ConstructorType::Type(ty) => ConcreteConstructorHandle::Type(ty),
        };

        Ok(c)
    }
}
