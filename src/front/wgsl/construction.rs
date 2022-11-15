use crate::front::wgsl;
use crate::front::wgsl::ast;
use crate::front::wgsl::ast::{ArraySize, ConstructorType};
use crate::front::wgsl::Error;
use crate::front::wgsl::{ExpressionContext, Lowerer, OutputContext};
use crate::{Handle, Span};

enum ConcreteConstructor {
    PartialVector {
        size: crate::VectorSize,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    PartialArray,
    Type(Handle<crate::Type>, crate::TypeInner),
}

impl ConcreteConstructor {
    fn to_error_string(&self, ctx: ExpressionContext) -> String {
        match *self {
            ConcreteConstructor::PartialVector { size } => {
                format!("vec{}<?>", size as u32,)
            }
            ConcreteConstructor::PartialMatrix { columns, rows } => {
                format!("mat{}x{}<?>", columns as u32, rows as u32,)
            }
            ConcreteConstructor::PartialArray => "array<?, ?>".to_string(),
            ConcreteConstructor::Type(ty, _) => ctx.fmt_ty(ty).to_string(),
        }
    }
}

enum Components {
    None,
    One {
        component: Handle<crate::Expression>,
        span: Span,
        ty: Handle<crate::Type>,
        ty_inner: crate::TypeInner,
    },
    Many {
        components: Vec<Handle<crate::Expression>>,
        spans: Vec<Span>,
        first_component_ty_inner: crate::TypeInner,
    },
}

impl Components {
    fn into_components_vec(self) -> Vec<Handle<crate::Expression>> {
        match self {
            Components::None => vec![],
            Components::One { component, .. } => vec![component],
            Components::Many { components, .. } => components,
        }
    }
}

impl<'source, 'temp> Lowerer<'source, 'temp> {
    pub(super) fn construct(
        &mut self,
        span: Span,
        constructor: &ConstructorType<'source>,
        c_span: wgsl::Span,
        components: &[Handle<ast::Expression<'source>>],
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let mut octx = OutputContext {
            read_expressions: None,
            global_expressions: ctx.global_expressions,
            globals: ctx.globals,
            module: ctx.module,
        };
        let constructor = self.constructor(constructor, octx.reborrow())?;

        let components = match *components {
            [] => Components::None,
            [component] => {
                let span = ctx.read_expressions.get_span(component);
                let component = self.lower_expression(component, ctx.reborrow())?;
                let ty = ctx.resolve_type(component)?;

                Components::One {
                    component,
                    span,
                    ty,
                    ty_inner: ctx.module.types[ty].inner.clone(),
                }
            }
            [component, ref rest @ ..] => {
                let span = ctx.read_expressions.get_span(component);
                let component = self.lower_expression(component, ctx.reborrow())?;
                let ty = ctx.resolve_type(component)?;

                Components::Many {
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
                    first_component_ty_inner: ctx.module.types[ty].inner.clone(),
                }
            }
        };

        let expr = match (components, constructor) {
            // Empty constructor
            (Components::None, dst_ty) => {
                let ty = match dst_ty {
                    ConcreteConstructor::Type(ty, _) => ty,
                    _ => return Err(Error::TypeNotInferrable(c_span)),
                };

                return match ctx.create_zero_value_constant(ty) {
                    Some(constant) => {
                        Ok(ctx.interrupt_emitter(crate::Expression::Constant(constant), span))
                    }
                    None => Err(Error::TypeNotConstructible(c_span)),
                };
            }

            // Scalar constructor & conversion (scalar -> scalar)
            (
                Components::One {
                    component,
                    ty_inner: crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(_, crate::TypeInner::Scalar { kind, width }),
            ) => crate::Expression::As {
                expr: component,
                kind,
                convert: Some(width),
            },

            // Vector conversion (vector -> vector)
            (
                Components::One {
                    component,
                    ty_inner: crate::TypeInner::Vector { size: src_size, .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Vector {
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
                        crate::TypeInner::Vector {
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
                        crate::TypeInner::Matrix {
                            columns: src_columns,
                            rows: src_rows,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Matrix {
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
                        crate::TypeInner::Matrix {
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
                    ty_inner: crate::TypeInner::Scalar { .. },
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
                        crate::TypeInner::Scalar {
                            kind: src_kind,
                            width: src_width,
                            ..
                        },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Vector {
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
                        crate::TypeInner::Scalar { kind, width }
                        | crate::TypeInner::Vector { kind, width, .. },
                    ..
                },
                ConcreteConstructor::PartialVector { size },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner:
                        crate::TypeInner::Scalar { .. } | crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(_, crate::TypeInner::Vector { size, width, kind }),
            ) => {
                let inner = crate::TypeInner::Vector { size, kind, width };
                let ty = ctx.module.types.insert(
                    crate::Type {
                        name: Some(inner.to_wgsl(&ctx.module.types, &ctx.module.constants)),
                        inner,
                    },
                    Span::UNDEFINED,
                );
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by elements)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: crate::TypeInner::Scalar { width, .. },
                    ..
                },
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: crate::TypeInner::Scalar { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
            ) => {
                let inner = crate::TypeInner::Vector {
                    width,
                    kind: crate::ScalarKind::Float,
                    size: rows,
                };
                let vec_ty = ctx.module.types.insert(
                    crate::Type {
                        name: Some(inner.to_wgsl(&ctx.module.types, &ctx.module.constants)),
                        inner,
                    },
                    Default::default(),
                );

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

                let inner = crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                };
                let ty = ctx.module.types.insert(
                    crate::Type {
                        name: Some(inner.to_wgsl(&ctx.module.types, &ctx.module.constants)),
                        inner,
                    },
                    Default::default(),
                );
                crate::Expression::Compose { ty, components }
            }

            // Matrix constructor (by columns)
            (
                Components::Many {
                    components,
                    first_component_ty_inner: crate::TypeInner::Vector { width, .. },
                    ..
                },
                ConcreteConstructor::PartialMatrix { columns, rows },
            )
            | (
                Components::Many {
                    components,
                    first_component_ty_inner: crate::TypeInner::Vector { .. },
                    ..
                },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                ),
            ) => {
                let inner = crate::TypeInner::Matrix {
                    columns,
                    rows,
                    width,
                };
                let ty = ctx.module.types.insert(
                    crate::Type {
                        name: Some(inner.to_wgsl(&ctx.module.types, &ctx.module.constants)),
                        inner,
                    },
                    Default::default(),
                );
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
                let ty = ctx.module.types.insert(
                    crate::Type {
                        name: Some(inner.to_wgsl(&ctx.module.types, &ctx.module.constants)),
                        inner,
                    },
                    Span::UNDEFINED,
                );

                crate::Expression::Compose { ty, components }
            }

            // Array constructor
            (components, ConcreteConstructor::Type(ty, crate::TypeInner::Array { .. })) => {
                let components = components.into_components_vec();
                crate::Expression::Compose { ty, components }
            }

            // Struct constructor
            (components, ConcreteConstructor::Type(ty, crate::TypeInner::Struct { .. })) => {
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
                dst_ty,
            ) => {
                let from_type = ctx.fmt_ty(src_ty).to_string();
                return Err(Error::BadTypeCast {
                    span: span.to_range().unwrap(),
                    from_type,
                    to_type: dst_ty.to_error_string(ctx.reborrow()),
                });
            }

            // Too many parameters for scalar constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(_, crate::TypeInner::Scalar { .. }),
            ) => {
                return Err(Error::UnexpectedComponents(wgsl::Span {
                    start: spans[1].to_range().unwrap().start,
                    end: spans.last().unwrap().to_range().unwrap().end,
                }));
            }

            // Parameters are of the wrong type for vector or matrix constructor
            (
                Components::Many { spans, .. },
                ConcreteConstructor::Type(
                    _,
                    crate::TypeInner::Vector { .. } | crate::TypeInner::Matrix { .. },
                )
                | ConcreteConstructor::PartialVector { .. }
                | ConcreteConstructor::PartialMatrix { .. },
            ) => {
                return Err(Error::InvalidConstructorComponentType(
                    spans[0].to_range().unwrap(),
                    0,
                ));
            }

            // Other types can't be constructed
            _ => return Err(Error::TypeNotConstructible(c_span)),
        };

        let expr = ctx.expressions.append(expr, span);
        Ok(expr)
    }

    pub(super) fn const_construct(
        &mut self,
        span: Span,
        constructor: &ConstructorType<'source>,
        components: &[Handle<ast::Expression<'source>>],
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<crate::ConstantInner, Error<'source>> {
        // TODO: Support zero values, splatting and inference.

        let constructor = self.constructor(constructor, ctx.reborrow())?;

        let c = match constructor {
            ConcreteConstructor::Type(ty, _) => {
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
            _ => return Err(Error::ConstExprUnsupported(span.to_range().unwrap())),
        };
        Ok(c)
    }

    fn constructor(
        &mut self,
        constructor: &ConstructorType<'source>,
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<ConcreteConstructor, Error<'source>> {
        let c = match *constructor {
            ConstructorType::Scalar { width, kind } => {
                let ty = self
                    .ensure_type_exists(crate::TypeInner::Scalar { width, kind }, ctx.reborrow());
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::PartialVector { size } => ConcreteConstructor::PartialVector { size },
            ConstructorType::Vector { size, kind, width } => {
                let ty = self.ensure_type_exists(
                    crate::TypeInner::Vector { size, kind, width },
                    ctx.reborrow(),
                );
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::PartialMatrix { rows, columns } => {
                ConcreteConstructor::PartialMatrix { rows, columns }
            }
            ConstructorType::Matrix {
                rows,
                columns,
                width,
            } => {
                let ty = self.ensure_type_exists(
                    crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    },
                    ctx.reborrow(),
                );
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::PartialArray => ConcreteConstructor::PartialArray,
            ConstructorType::Array { ref base, size } => {
                let base = self.resolve_type(base, ctx.reborrow())?;
                let size = match size {
                    ArraySize::Constant(expr) => {
                        let span = ctx.global_expressions.get_span(expr);
                        let expr = &ctx.global_expressions[expr];
                        crate::ArraySize::Constant(self.constant(expr, span, ctx.reborrow())?)
                    }
                    ArraySize::Dynamic => crate::ArraySize::Dynamic,
                };

                self.layouter
                    .update(&ctx.module.types, &ctx.module.constants)
                    .unwrap();
                let ty = self.ensure_type_exists(
                    crate::TypeInner::Array {
                        base,
                        size,
                        stride: self.layouter[base].to_stride(),
                    },
                    ctx.reborrow(),
                );
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::Type(ty) => {
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
        };

        Ok(c)
    }
}
