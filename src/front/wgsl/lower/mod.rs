use crate::front::wgsl::ast::{
    ConstructorType, GlobalDeclKind, TranslationUnit, TypeKind, VarDecl,
};
use crate::front::wgsl::errors::{Error, ExpectedToken, InvalidAssignmentType};
use crate::front::wgsl::index::Index;
use crate::front::wgsl::number::Number;
use crate::front::wgsl::{ast, conv};
use crate::front::{Emitter, Typifier};
use crate::proc::{ensure_block_returns, Alignment, Layouter, ResolveContext, TypeResolution};
use crate::{Arena, FastHashMap, Handle, NamedExpressions, Span};

mod construct;

enum GlobalDecl {
    Function(Handle<crate::Function>),
    Var(Handle<crate::GlobalVariable>),
    Const(Handle<crate::Constant>),
    Type(Handle<crate::Type>),
    EntryPoint,
}

struct OutputContext<'source, 'temp, 'out> {
    read_expressions: Option<&'temp Arena<ast::Expression<'source>>>,
    global_expressions: &'temp Arena<ast::Expression<'source>>,
    globals: &'temp mut FastHashMap<&'source str, GlobalDecl>,
    module: &'out mut crate::Module,
}

impl<'source> OutputContext<'source, '_, '_> {
    fn reborrow(&mut self) -> OutputContext<'source, '_, '_> {
        OutputContext {
            read_expressions: self.read_expressions.as_deref(),
            global_expressions: self.global_expressions,
            globals: self.globals,
            module: self.module,
        }
    }
}

struct StatementContext<'source, 'temp, 'out> {
    local_table: &'temp mut FastHashMap<Handle<ast::Local>, TypedExpression>,
    globals: &'temp mut FastHashMap<&'source str, GlobalDecl>,
    global_expressions: &'temp Arena<ast::Expression<'source>>,
    read_expressions: &'temp Arena<ast::Expression<'source>>,
    typifier: &'temp mut Typifier,
    variables: &'out mut Arena<crate::LocalVariable>,
    expressions: &'out mut Arena<crate::Expression>,
    named_expressions: &'out mut FastHashMap<Handle<crate::Expression>, String>,
    arguments: &'out [crate::FunctionArgument],
    module: &'out mut crate::Module,
}

impl<'a, 'temp> StatementContext<'a, 'temp, '_> {
    fn reborrow(&mut self) -> StatementContext<'a, '_, '_> {
        StatementContext {
            local_table: self.local_table,
            globals: self.globals,
            global_expressions: self.global_expressions,
            read_expressions: self.read_expressions,
            typifier: self.typifier,
            variables: self.variables,
            expressions: self.expressions,
            named_expressions: self.named_expressions,
            arguments: self.arguments,
            module: self.module,
        }
    }

    fn as_expression<'t>(
        &'t mut self,
        block: &'t mut crate::Block,
        emitter: &'t mut Emitter,
    ) -> ExpressionContext<'a, 't, '_>
    where
        'temp: 't,
    {
        ExpressionContext {
            local_table: self.local_table,
            globals: self.globals,
            global_expressions: self.global_expressions,
            read_expressions: self.read_expressions,
            typifier: self.typifier,
            expressions: self.expressions,
            module: self.module,
            local_vars: self.variables,
            arguments: self.arguments,
            block,
            emitter,
        }
    }
}

struct SamplingContext {
    image: Handle<crate::Expression>,
    arrayed: bool,
}

struct ExpressionContext<'source, 'temp, 'out> {
    local_table: &'temp mut FastHashMap<Handle<ast::Local>, TypedExpression>,
    globals: &'temp mut FastHashMap<&'source str, GlobalDecl>,
    global_expressions: &'temp Arena<ast::Expression<'source>>,
    read_expressions: &'temp Arena<ast::Expression<'source>>,
    typifier: &'temp mut Typifier,
    expressions: &'out mut Arena<crate::Expression>,
    local_vars: &'out Arena<crate::LocalVariable>,
    arguments: &'out [crate::FunctionArgument],
    module: &'out mut crate::Module,
    block: &'temp mut crate::Block,
    emitter: &'temp mut Emitter,
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            local_table: self.local_table,
            globals: self.globals,
            global_expressions: self.global_expressions,
            read_expressions: self.read_expressions,
            typifier: self.typifier,
            expressions: self.expressions,
            module: self.module,
            local_vars: self.local_vars,
            arguments: self.arguments,
            block: self.block,
            emitter: self.emitter,
        }
    }

    fn resolve_type(
        &mut self,
        handle: Handle<crate::Expression>,
    ) -> Result<Handle<crate::Type>, Error<'a>> {
        let resolve_ctx = ResolveContext {
            constants: &self.module.constants,
            types: &self.module.types,
            global_vars: &self.module.global_variables,
            local_vars: self.local_vars,
            functions: &self.module.functions,
            arguments: self.arguments,
        };
        match self.typifier.grow(handle, self.expressions, &resolve_ctx) {
            Err(e) => Err(Error::InvalidResolve(e)),
            Ok(()) => {
                let res = &self.typifier[handle];
                match *res {
                    TypeResolution::Handle(handle) => Ok(handle),
                    TypeResolution::Value(ref inner) => {
                        let name = inner.to_wgsl(&self.module.types, &self.module.constants);
                        let handle = self.module.types.insert(
                            crate::Type {
                                name: Some(name),
                                inner: inner.clone(),
                            },
                            Span::UNDEFINED,
                        );
                        Ok(handle)
                    }
                }
            }
        }
    }

    fn prepare_sampling(
        &mut self,
        image: Handle<crate::Expression>,
        span: Span,
    ) -> Result<SamplingContext, Error<'a>> {
        Ok(SamplingContext {
            image,
            arrayed: {
                let ty = self.resolve_type(image)?;
                match self.module.types[ty].inner {
                    crate::TypeInner::Image { arrayed, .. } => arrayed,
                    _ => return Err(Error::BadTexture(span.to_range().unwrap())),
                }
            },
        })
    }

    fn prepare_args<'b>(
        &mut self,
        args: &'b [Handle<ast::Expression<'a>>],
        min_args: u32,
        span: Span,
    ) -> ArgumentContext<'b, 'a> {
        ArgumentContext {
            args: args.iter(),
            min_args,
            args_used: 0,
            total_args: args.len() as u32,
            span: span.to_range().unwrap(),
        }
    }

    /// Insert splats, if needed by the non-'*' operations.
    fn binary_op_splat(
        &mut self,
        op: crate::BinaryOperator,
        left: &mut Handle<crate::Expression>,
        right: &mut Handle<crate::Expression>,
    ) -> Result<(), Error<'a>> {
        if op != crate::BinaryOperator::Multiply {
            let ty = self.resolve_type(*left)?;
            let left_size = match self.module.types[ty].inner {
                crate::TypeInner::Vector { size, .. } => Some(size),
                _ => None,
            };

            let ty = self.resolve_type(*right)?;
            match (left_size, &self.module.types[ty].inner) {
                (Some(size), &crate::TypeInner::Scalar { .. }) => {
                    *right = self.expressions.append(
                        crate::Expression::Splat {
                            size,
                            value: *right,
                        },
                        self.expressions.get_span(*right),
                    );
                }
                (None, &crate::TypeInner::Vector { size, .. }) => {
                    *left = self.expressions.append(
                        crate::Expression::Splat { size, value: *left },
                        self.expressions.get_span(*left),
                    );
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Add a single expression to the expression table that is not covered by `self.emitter`.
    ///
    /// This is useful for `CallResult` and `AtomicResult` expressions, which should not be covered by
    /// `Emit` statements.
    fn interrupt_emitter(
        &mut self,
        expression: crate::Expression,
        span: Span,
    ) -> Handle<crate::Expression> {
        self.block.extend(self.emitter.finish(self.expressions));
        let result = self.expressions.append(expression, span);
        self.emitter.start(self.expressions);
        result
    }

    /// Apply the WGSL Load Rule to `expr`.
    ///
    /// If `expr` is has type `ref<SC, T, A>`, perform a load to produce a value of type
    /// `T`. Otherwise, return `expr` unchanged.
    fn apply_load_rule(&mut self, expr: TypedExpression) -> Handle<crate::Expression> {
        if expr.is_reference {
            let load = crate::Expression::Load {
                pointer: expr.handle,
            };
            let span = self.expressions.get_span(expr.handle);
            self.expressions.append(load, span)
        } else {
            expr.handle
        }
    }

    /// Creates a zero value constant of type `ty`
    ///
    /// Returns `None` if the given `ty` is not a constructible type
    fn create_zero_value_constant(
        &mut self,
        ty: Handle<crate::Type>,
    ) -> Option<Handle<crate::Constant>> {
        let inner = match self.module.types[ty].inner {
            crate::TypeInner::Scalar { kind, width } => {
                let value = match kind {
                    crate::ScalarKind::Sint => crate::ScalarValue::Sint(0),
                    crate::ScalarKind::Uint => crate::ScalarValue::Uint(0),
                    crate::ScalarKind::Float => crate::ScalarValue::Float(0.),
                    crate::ScalarKind::Bool => crate::ScalarValue::Bool(false),
                };
                crate::ConstantInner::Scalar { width, value }
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_ty = self.module.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Scalar { width, kind },
                    },
                    Default::default(),
                );
                let component = self.create_zero_value_constant(scalar_ty);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..size as u8).map(|_| component).collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let vec_ty = self.module.types.insert(
                    crate::Type {
                        name: None,
                        inner: crate::TypeInner::Vector {
                            width,
                            kind: crate::ScalarKind::Float,
                            size: rows,
                        },
                    },
                    Default::default(),
                );
                let component = self.create_zero_value_constant(vec_ty);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..columns as u8)
                        .map(|_| component)
                        .collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(size),
                ..
            } => {
                let component = self.create_zero_value_constant(base);
                crate::ConstantInner::Composite {
                    ty,
                    components: (0..self.module.constants[size].to_array_length().unwrap())
                        .map(|_| component)
                        .collect::<Option<_>>()?,
                }
            }
            crate::TypeInner::Struct { ref members, .. } => {
                let members = members.clone();
                crate::ConstantInner::Composite {
                    ty,
                    components: members
                        .iter()
                        .map(|member| self.create_zero_value_constant(member.ty))
                        .collect::<Option<_>>()?,
                }
            }
            _ => return None,
        };

        let constant = self.module.constants.fetch_or_append(
            crate::Constant {
                name: None,
                specialization: None,
                inner,
            },
            Span::default(),
        );
        Some(constant)
    }

    /// Format type `ty`.
    fn fmt_ty(&self, ty: Handle<crate::Type>) -> &str {
        self.module.types[ty]
            .name
            .as_ref()
            .map(|s| &**s)
            .unwrap_or("unknown")
    }
}

struct ArgumentContext<'ctx, 'source> {
    args: std::slice::Iter<'ctx, Handle<ast::Expression<'source>>>,
    min_args: u32,
    args_used: u32,
    total_args: u32,
    span: super::Span,
}

impl<'source> ArgumentContext<'_, 'source> {
    pub fn finish(self) -> Result<(), Error<'source>> {
        if self.args.len() == 0 {
            Ok(())
        } else {
            Err(Error::WrongArgumentCount {
                found: self.total_args,
                expected: self.min_args..self.args_used + 1,
                span: self.span,
            })
        }
    }

    pub fn next(&mut self) -> Result<Handle<ast::Expression<'source>>, Error<'source>> {
        match self.args.next().copied() {
            Some(arg) => {
                self.args_used += 1;
                Ok(arg)
            }
            None => Err(Error::WrongArgumentCount {
                found: self.total_args,
                expected: self.min_args..self.args_used + 1,
                span: self.span.clone(),
            }),
        }
    }
}

/// A Naga [`Expression`] handle, with WGSL type information.
///
/// Naga and WGSL types are very close, but Naga lacks WGSL's 'reference' types,
/// which we need to know to apply the Load Rule. This struct carries a Naga
/// `Handle<Expression>` along with enough information to determine its WGSL type.
///
/// [`Expression`]: crate::Expression
#[derive(Debug, Copy, Clone)]
struct TypedExpression {
    /// The handle of the Naga expression.
    handle: Handle<crate::Expression>,

    /// True if this expression's WGSL type is a reference.
    ///
    /// When this is true, `handle` must be a pointer.
    is_reference: bool,
}

enum Composition {
    Single(u32),
    Multi(crate::VectorSize, [crate::SwizzleComponent; 4]),
}

impl Composition {
    const fn letter_component(letter: char) -> Option<crate::SwizzleComponent> {
        use crate::SwizzleComponent as Sc;
        match letter {
            'x' | 'r' => Some(Sc::X),
            'y' | 'g' => Some(Sc::Y),
            'z' | 'b' => Some(Sc::Z),
            'w' | 'a' => Some(Sc::W),
            _ => None,
        }
    }

    fn extract_impl(name: &str, name_span: Span) -> Result<u32, Error> {
        let ch = name
            .chars()
            .next()
            .ok_or_else(|| Error::BadAccessor(name_span.to_range().unwrap()))?;
        match Self::letter_component(ch) {
            Some(sc) => Ok(sc as u32),
            None => Err(Error::BadAccessor(name_span.to_range().unwrap())),
        }
    }

    fn make(name: &str, name_span: Span) -> Result<Self, Error> {
        if name.len() > 1 {
            let mut components = [crate::SwizzleComponent::X; 4];
            for (comp, ch) in components.iter_mut().zip(name.chars()) {
                *comp = Self::letter_component(ch)
                    .ok_or_else(|| Error::BadAccessor(name_span.to_range().unwrap()))?;
            }

            let size = match name.len() {
                2 => crate::VectorSize::Bi,
                3 => crate::VectorSize::Tri,
                4 => crate::VectorSize::Quad,
                _ => return Err(Error::BadAccessor(name_span.to_range().unwrap())),
            };
            Ok(Composition::Multi(size, components))
        } else {
            Self::extract_impl(name, name_span).map(Composition::Single)
        }
    }
}

enum ConstantOrInner {
    Constant(Handle<crate::Constant>),
    Inner(crate::ConstantInner),
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
                GlobalDeclKind::Fn(ref f) => {
                    let decl = self.lower_fn(
                        f,
                        span,
                        OutputContext {
                            read_expressions: None,
                            global_expressions: &tu.global_expressions,
                            globals: &mut globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(f.name.name, decl);
                }
                GlobalDeclKind::Var(ref v) => {
                    let mut ctx = OutputContext {
                        read_expressions: None,
                        global_expressions: &tu.global_expressions,
                        globals: &mut globals,
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
                        read_expressions: None,
                        global_expressions: &tu.global_expressions,
                        globals: &mut globals,
                        module: &mut module,
                    };

                    let inner = self.constant_inner(
                        &ctx.global_expressions[c.init],
                        ctx.global_expressions.get_span(c.init),
                        ctx.reborrow(),
                    )?;
                    let inner = match inner {
                        ConstantOrInner::Constant(c) => ctx.module.constants[c].inner.clone(),
                        ConstantOrInner::Inner(inner) => inner,
                    };

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
                                ctx.module.types[inferred_type]
                                    .name
                                    .as_ref()
                                    .unwrap()
                                    .clone(),
                            ));
                        }
                    }

                    globals.insert(c.name.name, GlobalDecl::Const(handle));
                }
                GlobalDeclKind::Struct(ref s) => {
                    let handle = self.lower_struct(
                        s,
                        span,
                        OutputContext {
                            read_expressions: None,
                            global_expressions: &tu.global_expressions,
                            globals: &mut globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(s.name.name, GlobalDecl::Type(handle));
                }
                GlobalDeclKind::Type(ref alias) => {
                    let ty = self.resolve_type(
                        &alias.ty,
                        OutputContext {
                            read_expressions: None,
                            global_expressions: &tu.global_expressions,
                            globals: &mut globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(alias.name.name, GlobalDecl::Type(ty));
                }
            }
        }

        Ok(module)
    }

    fn lower_fn(
        &mut self,
        f: &ast::Function<'source>,
        span: Span,
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<GlobalDecl, Error<'source>> {
        let mut local_table = FastHashMap::default();
        let mut local_variables = Arena::new();
        let mut expressions = Arena::new();
        let mut named_expressions = NamedExpressions::default();

        let arguments = f
            .arguments
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let ty = self.resolve_type(&arg.ty, ctx.reborrow())?;
                let expr = expressions.append(
                    crate::Expression::FunctionArgument(i as u32),
                    arg.name.span.clone().into(),
                );
                local_table.insert(
                    arg.handle,
                    TypedExpression {
                        handle: expr,
                        is_reference: false,
                    },
                );
                named_expressions.insert(expr, arg.name.name.to_string());

                Ok(crate::FunctionArgument {
                    name: Some(arg.name.name.to_string()),
                    ty,
                    binding: self.interpolate_default(&arg.binding, ty, ctx.reborrow()),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let result = f
            .result
            .as_ref()
            .map(|res| {
                self.resolve_type(&res.ty, ctx.reborrow())
                    .map(|ty| crate::FunctionResult {
                        ty,
                        binding: self.interpolate_default(&res.binding, ty, ctx.reborrow()),
                    })
            })
            .transpose()?;

        let mut body = self.lower_block(
            &f.body,
            StatementContext {
                local_table: &mut local_table,
                globals: ctx.globals,
                global_expressions: ctx.global_expressions,
                read_expressions: &f.expressions,
                typifier: &mut Default::default(),
                variables: &mut local_variables,
                expressions: &mut expressions,
                named_expressions: &mut named_expressions,
                module: ctx.module,
                arguments: &arguments,
            },
        )?;
        ensure_block_returns(&mut body);

        let function = crate::Function {
            name: Some(f.name.name.to_string()),
            arguments,
            result,
            local_variables,
            expressions,
            named_expressions,
            body,
        };

        if let Some(ref entry) = f.entry_point {
            ctx.module.entry_points.push(crate::EntryPoint {
                name: f.name.name.to_string(),
                stage: entry.stage,
                early_depth_test: entry.early_depth_test,
                workgroup_size: entry.workgroup_size,
                function,
            });
            Ok(GlobalDecl::EntryPoint)
        } else {
            let handle = ctx.module.functions.append(function, span);
            Ok(GlobalDecl::Function(handle))
        }
    }

    fn lower_block(
        &mut self,
        b: &ast::Block<'source>,
        mut ctx: StatementContext<'source, '_, '_>,
    ) -> Result<crate::Block, Error<'source>> {
        let mut block = crate::Block::default();

        for stmt in b.stmts.iter() {
            self.lower_statement(stmt, &mut block, ctx.reborrow())?;
        }

        Ok(block)
    }

    fn lower_statement(
        &mut self,
        stmt: &ast::Statement<'source>,
        block: &mut crate::Block,
        mut ctx: StatementContext<'source, '_, '_>,
    ) -> Result<(), Error<'source>> {
        let out = match stmt.kind {
            ast::StatementKind::Block(ref block) => {
                let block = self.lower_block(block, ctx.reborrow())?;
                crate::Statement::Block(block)
            }
            ast::StatementKind::VarDecl(ref decl) => match **decl {
                VarDecl::Let(ref l) => {
                    let mut emitter = Emitter::default();
                    emitter.start(ctx.expressions);

                    let value =
                        self.lower_expression(l.init, ctx.as_expression(block, &mut emitter))?;

                    let explicit_ty =
                        l.ty.as_ref()
                            .map(|ty| {
                                self.resolve_type(
                                    ty,
                                    OutputContext {
                                        read_expressions: None,
                                        global_expressions: ctx.global_expressions,
                                        globals: ctx.globals,
                                        module: ctx.module,
                                    },
                                )
                            })
                            .transpose()?;

                    if let Some(ty) = explicit_ty {
                        let mut ctx = ctx.as_expression(block, &mut emitter);
                        let init_ty = ctx.resolve_type(value)?;
                        if !ctx.module.types[ty]
                            .inner
                            .equivalent(&ctx.module.types[init_ty].inner, &ctx.module.types)
                        {
                            return Err(Error::InitializationTypeMismatch(
                                l.name.span.clone(),
                                ctx.fmt_ty(ty).to_string(),
                                ctx.fmt_ty(init_ty).to_string(),
                            ));
                        }
                    }

                    block.extend(emitter.finish(ctx.expressions));
                    ctx.local_table.insert(
                        l.handle,
                        TypedExpression {
                            handle: value,
                            is_reference: false,
                        },
                    );
                    ctx.named_expressions.insert(value, l.name.name.to_string());

                    return Ok(());
                }
                VarDecl::Var(ref v) => {
                    let mut emitter = Emitter::default();
                    emitter.start(ctx.expressions);

                    let value = v
                        .init
                        .map(|init| {
                            self.lower_expression(init, ctx.as_expression(block, &mut emitter))
                        })
                        .transpose()?;

                    let inferred_ty = value
                        .map(|value| {
                            let mut ctx = ctx.as_expression(block, &mut emitter);
                            ctx.resolve_type(value)
                        })
                        .transpose()?;

                    let explicit_ty =
                        v.ty.as_ref()
                            .map(|ty| {
                                self.resolve_type(
                                    ty,
                                    OutputContext {
                                        read_expressions: None,
                                        global_expressions: ctx.global_expressions,
                                        globals: ctx.globals,
                                        module: ctx.module,
                                    },
                                )
                            })
                            .transpose()?;

                    let ty = match (explicit_ty, inferred_ty) {
                        (Some(explicit), Some(inferred)) => {
                            let ctx = ctx.as_expression(block, &mut emitter);
                            if !ctx.module.types[explicit]
                                .inner
                                .equivalent(&ctx.module.types[inferred].inner, &ctx.module.types)
                            {
                                return Err(Error::InitializationTypeMismatch(
                                    v.name.span.clone(),
                                    ctx.fmt_ty(explicit).to_string(),
                                    ctx.fmt_ty(inferred).to_string(),
                                ));
                            }
                            explicit
                        }
                        (Some(explicit), None) => explicit,
                        (None, Some(inferred)) => inferred,
                        (None, None) => {
                            return Err(Error::MissingType(v.name.span.clone()));
                        }
                    };

                    let var = ctx.variables.append(
                        crate::LocalVariable {
                            name: Some(v.name.name.to_string()),
                            ty,
                            init: None,
                        },
                        stmt.span.clone().into(),
                    );

                    let handle = ctx
                        .as_expression(block, &mut emitter)
                        .interrupt_emitter(crate::Expression::LocalVariable(var), Span::UNDEFINED);
                    block.extend(emitter.finish(ctx.expressions));
                    ctx.local_table.insert(
                        v.handle,
                        TypedExpression {
                            handle,
                            is_reference: true,
                        },
                    );

                    match value {
                        Some(value) => crate::Statement::Store {
                            pointer: handle,
                            value,
                        },
                        None => return Ok(()),
                    }
                }
                VarDecl::Const(_) => return Err(Error::ConstExprUnsupported(stmt.span.clone())),
            },
            ast::StatementKind::If {
                condition,
                ref accept,
                ref reject,
            } => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let condition =
                    self.lower_expression(condition, ctx.as_expression(block, &mut emitter))?;
                block.extend(emitter.finish(ctx.expressions));

                let accept = self.lower_block(accept, ctx.reborrow())?;
                let reject = self.lower_block(reject, ctx.reborrow())?;

                crate::Statement::If {
                    condition,
                    accept,
                    reject,
                }
            }
            ast::StatementKind::Switch {
                selector,
                ref cases,
            } => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let mut ectx = ctx.as_expression(block, &mut emitter);
                let selector = self.lower_expression(selector, ectx.reborrow())?;

                let ty = ectx.resolve_type(selector)?;
                let uint =
                    ectx.module.types[ty].inner.scalar_kind() == Some(crate::ScalarKind::Uint);
                block.extend(emitter.finish(ctx.expressions));

                let cases = cases
                    .iter()
                    .map(|case| {
                        Ok(crate::SwitchCase {
                            value: match case.value {
                                ast::SwitchValue::I32(num) if !uint => {
                                    crate::SwitchValue::Integer(num)
                                }
                                ast::SwitchValue::U32(num) if uint => {
                                    crate::SwitchValue::Integer(num as i32)
                                }
                                ast::SwitchValue::Default => crate::SwitchValue::Default,
                                _ => {
                                    return Err(Error::InvalidSwitchValue {
                                        uint,
                                        span: case.value_span.clone(),
                                    });
                                }
                            },
                            body: self.lower_block(&case.body, ctx.reborrow())?,
                            fall_through: case.fall_through,
                        })
                    })
                    .collect::<Result<_, _>>()?;

                crate::Statement::Switch { selector, cases }
            }
            ast::StatementKind::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                let body = self.lower_block(body, ctx.reborrow())?;
                let mut continuing = self.lower_block(continuing, ctx.reborrow())?;

                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);
                let break_if = break_if
                    .map(|expr| self.lower_expression(expr, ctx.as_expression(block, &mut emitter)))
                    .transpose()?;
                continuing.extend(emitter.finish(ctx.expressions));

                crate::Statement::Loop {
                    body,
                    continuing,
                    break_if,
                }
            }
            ast::StatementKind::Break => crate::Statement::Break,
            ast::StatementKind::Continue => crate::Statement::Continue,
            ast::StatementKind::Return { value } => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let value = value
                    .map(|expr| self.lower_expression(expr, ctx.as_expression(block, &mut emitter)))
                    .transpose()?;
                block.extend(emitter.finish(ctx.expressions));

                crate::Statement::Return { value }
            }
            ast::StatementKind::Kill => crate::Statement::Kill,
            ast::StatementKind::Call {
                ref function,
                ref arguments,
            } => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let _ = self.call(
                    stmt.span.clone().into(),
                    function,
                    arguments,
                    ctx.as_expression(block, &mut emitter),
                )?;
                block.extend(emitter.finish(ctx.expressions));
                return Ok(());
            }
            ast::StatementKind::Assign { target, op, value } => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let target = self.lower_expression_for_reference(
                    target,
                    ctx.as_expression(block, &mut emitter),
                )?;
                let mut value =
                    self.lower_expression(value, ctx.as_expression(block, &mut emitter))?;

                if !target.is_reference {
                    let ty = if ctx.named_expressions.contains_key(&target.handle) {
                        InvalidAssignmentType::ImmutableBinding
                    } else {
                        match ctx.expressions[target.handle] {
                            crate::Expression::Swizzle { .. } => InvalidAssignmentType::Swizzle,
                            _ => InvalidAssignmentType::Other,
                        }
                    };

                    return Err(Error::InvalidAssignment {
                        span: ctx.expressions.get_span(target.handle).to_range().unwrap(),
                        ty,
                    });
                }

                let value = match op {
                    Some(op) => {
                        let mut ctx = ctx.as_expression(block, &mut emitter);
                        let mut left = ctx.apply_load_rule(target);
                        ctx.binary_op_splat(op, &mut left, &mut value)?;
                        ctx.expressions.append(
                            crate::Expression::Binary {
                                op,
                                left,
                                right: value,
                            },
                            Span::from(stmt.span.clone()),
                        )
                    }
                    None => value,
                };
                block.extend(emitter.finish(ctx.expressions));

                crate::Statement::Store {
                    pointer: target.handle,
                    value,
                }
            }
            ast::StatementKind::Increment(value) | ast::StatementKind::Decrement(value) => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let op = match stmt.kind {
                    ast::StatementKind::Increment(_) => crate::BinaryOperator::Add,
                    ast::StatementKind::Decrement(_) => crate::BinaryOperator::Subtract,
                    _ => unreachable!(),
                };

                let value_span = ctx.read_expressions.get_span(value).to_range().unwrap();
                let reference = self.lower_expression_for_reference(
                    value,
                    ctx.as_expression(block, &mut emitter),
                )?;
                let mut ectx = ctx.as_expression(block, &mut emitter);

                let ty = ectx.resolve_type(reference.handle)?;
                let (kind, width) = match ectx.module.types[ty].inner {
                    crate::TypeInner::ValuePointer {
                        size: None,
                        kind,
                        width,
                        ..
                    } => (kind, width),
                    crate::TypeInner::Pointer { base, .. } => match ectx.module.types[base].inner {
                        crate::TypeInner::Scalar { kind, width } => (kind, width),
                        _ => return Err(Error::BadIncrDecrReferenceType(value_span.clone())),
                    },
                    _ => return Err(Error::BadIncrDecrReferenceType(value_span.clone())),
                };
                let constant_inner = crate::ConstantInner::Scalar {
                    width,
                    value: match kind {
                        crate::ScalarKind::Sint => crate::ScalarValue::Sint(1),
                        crate::ScalarKind::Uint => crate::ScalarValue::Uint(1),
                        _ => return Err(Error::BadIncrDecrReferenceType(value_span.clone())),
                    },
                };
                let constant = ectx.module.constants.append(
                    crate::Constant {
                        name: None,
                        specialization: None,
                        inner: constant_inner,
                    },
                    Span::UNDEFINED,
                );

                let left = ectx.expressions.append(
                    crate::Expression::Load {
                        pointer: reference.handle,
                    },
                    value_span.into(),
                );
                let right =
                    ectx.interrupt_emitter(crate::Expression::Constant(constant), Span::UNDEFINED);
                let value = ectx.expressions.append(
                    crate::Expression::Binary { op, left, right },
                    stmt.span.clone().into(),
                );

                block.extend(emitter.finish(ctx.expressions));
                crate::Statement::Store {
                    pointer: reference.handle,
                    value,
                }
            }
            ast::StatementKind::Ignore(expr) => {
                let mut emitter = Emitter::default();
                emitter.start(ctx.expressions);

                let _ = self.lower_expression(expr, ctx.as_expression(block, &mut emitter))?;
                block.extend(emitter.finish(ctx.expressions));
                return Ok(());
            }
        };

        block.push(out, Span::from(stmt.span.clone()));

        Ok(())
    }

    fn lower_expression(
        &mut self,
        expr: Handle<ast::Expression<'source>>,
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let expr = self.lower_expression_for_reference(expr, ctx.reborrow())?;
        Ok(ctx.apply_load_rule(expr))
    }

    fn lower_expression_for_reference(
        &mut self,
        expr: Handle<ast::Expression<'source>>,
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<TypedExpression, Error<'source>> {
        let span = ctx.read_expressions.get_span(expr);
        let expr = &ctx.read_expressions[expr];

        let (expr, is_reference) = match *expr {
            ast::Expression::Literal(literal) => {
                let inner = match literal {
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
                    ast::Literal::Number(x) => {
                        panic!("got abstract numeric type when not expected: {:?}", x);
                    }
                    ast::Literal::Bool(b) => crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(b),
                    },
                };
                let handle = ctx.module.constants.fetch_or_append(
                    crate::Constant {
                        name: None,
                        specialization: None,
                        inner,
                    },
                    Span::UNDEFINED,
                );
                let handle = ctx.interrupt_emitter(crate::Expression::Constant(handle), span);
                return Ok(TypedExpression {
                    handle,
                    is_reference: false,
                });
            }
            ast::Expression::Ident(ast::IdentExpr::Local(local)) => {
                return Ok(ctx.local_table[&local])
            }
            ast::Expression::Ident(ast::IdentExpr::Unresolved(name)) => {
                return if let Some(global) = ctx.globals.get(name) {
                    let (expr, is_reference) = match *global {
                        GlobalDecl::Var(handle) => (
                            crate::Expression::GlobalVariable(handle),
                            ctx.module.global_variables[handle].space
                                != crate::AddressSpace::Handle,
                        ),
                        GlobalDecl::Const(handle) => (crate::Expression::Constant(handle), false),
                        _ => {
                            return Err(Error::Unexpected(
                                span.to_range().unwrap(),
                                ExpectedToken::Variable,
                            ));
                        }
                    };

                    let handle = ctx.interrupt_emitter(expr, span);
                    Ok(TypedExpression {
                        handle,
                        is_reference,
                    })
                } else {
                    Err(Error::UnknownIdent(span.to_range().unwrap(), name))
                }
            }
            ast::Expression::Construct {
                ref ty,
                ref ty_span,
                ref components,
            } => {
                let handle =
                    self.construct(span, ty, ty_span.clone(), components, ctx.reborrow())?;
                return Ok(TypedExpression {
                    handle,
                    is_reference: false,
                });
            }
            ast::Expression::Unary { op, expr } => {
                let expr = self.lower_expression(expr, ctx.reborrow())?;
                (crate::Expression::Unary { op, expr }, false)
            }
            ast::Expression::AddrOf(expr) => {
                // The `&` operator simply converts a reference to a pointer. And since a
                // reference is required, the Load Rule is not applied.
                let expr = self.lower_expression_for_reference(expr, ctx.reborrow())?;
                if !expr.is_reference {
                    return Err(Error::NotReference(
                        "the operand of the `&` operator",
                        span.to_range().unwrap(),
                    ));
                }

                // No code is generated. We just declare the pointer a reference now.
                return Ok(TypedExpression {
                    is_reference: false,
                    ..expr
                });
            }
            ast::Expression::Deref(expr) => {
                // The pointer we dereference must be loaded.
                let pointer = self.lower_expression(expr, ctx.reborrow())?;

                let ty = ctx.resolve_type(pointer)?;
                if ctx.module.types[ty].inner.pointer_space().is_none() {
                    return Err(Error::NotPointer(span.to_range().unwrap()));
                }

                return Ok(TypedExpression {
                    handle: pointer,
                    is_reference: true,
                });
            }
            ast::Expression::Binary { op, left, right } => {
                // Load both operands.
                let mut left = self.lower_expression(left, ctx.reborrow())?;
                let mut right = self.lower_expression(right, ctx.reborrow())?;
                ctx.binary_op_splat(op, &mut left, &mut right)?;
                (crate::Expression::Binary { op, left, right }, false)
            }
            ast::Expression::Call {
                ref function,
                ref arguments,
            } => {
                let handle = self
                    .call(span, function, arguments, ctx.reborrow())?
                    .ok_or(Error::FunctionReturnsVoid(function.span.clone()))?;
                return Ok(TypedExpression {
                    handle,
                    is_reference: false,
                });
            }
            ast::Expression::Index { base, index } => {
                let base = self.lower_expression_for_reference(base, ctx.reborrow())?;
                let index = self.lower_expression(index, ctx.reborrow())?;

                let ty = ctx.resolve_type(base.handle)?;
                let wgsl_pointer =
                    ctx.module.types[ty].inner.pointer_space().is_some() && !base.is_reference;

                if wgsl_pointer {
                    return Err(Error::Pointer(
                        "the value indexed by a `[]` subscripting expression",
                        ctx.expressions.get_span(base.handle).to_range().unwrap(),
                    ));
                }

                if let crate::Expression::Constant(constant) = ctx.expressions[index] {
                    use std::convert::TryFrom;
                    let span = ctx.expressions.get_span(index).to_range().unwrap();
                    let index = match ctx.module.constants[constant].inner {
                        crate::ConstantInner::Scalar {
                            value: crate::ScalarValue::Uint(int),
                            ..
                        } => u32::try_from(int).map_err(|_| Error::BadU32Constant(span)),
                        crate::ConstantInner::Scalar {
                            value: crate::ScalarValue::Sint(int),
                            ..
                        } => u32::try_from(int).map_err(|_| Error::BadU32Constant(span)),
                        _ => Err(Error::BadU32Constant(span)),
                    }?;

                    (
                        crate::Expression::AccessIndex {
                            base: base.handle,
                            index,
                        },
                        base.is_reference,
                    )
                } else {
                    (
                        crate::Expression::Access {
                            base: base.handle,
                            index,
                        },
                        base.is_reference,
                    )
                }
            }
            ast::Expression::Member { base, ref field } => {
                let TypedExpression {
                    handle,
                    is_reference,
                } = self.lower_expression_for_reference(base, ctx.reborrow())?;

                let ty = ctx.resolve_type(handle)?;
                let temp_inner;
                let (composite, wgsl_pointer) = match ctx.module.types[ty].inner {
                    crate::TypeInner::Pointer { base, .. } => {
                        (&ctx.module.types[base].inner, !is_reference)
                    }
                    crate::TypeInner::ValuePointer {
                        size: None,
                        kind,
                        width,
                        ..
                    } => {
                        temp_inner = crate::TypeInner::Scalar { kind, width };
                        (&temp_inner, !is_reference)
                    }
                    crate::TypeInner::ValuePointer {
                        size: Some(size),
                        kind,
                        width,
                        ..
                    } => {
                        temp_inner = crate::TypeInner::Vector { size, kind, width };
                        (&temp_inner, !is_reference)
                    }
                    ref other => (other, false),
                };

                if wgsl_pointer {
                    return Err(Error::Pointer(
                        "the value accessed by a `.member` expression",
                        ctx.expressions.get_span(handle).to_range().unwrap(),
                    ));
                }

                let access = match *composite {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let index = members
                            .iter()
                            .position(|m| m.name.as_deref() == Some(field.name))
                            .ok_or(Error::BadAccessor(field.span.clone()))?
                            as u32;

                        (
                            crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            },
                            is_reference,
                        )
                    }
                    crate::TypeInner::Vector { .. } | crate::TypeInner::Matrix { .. } => {
                        match Composition::make(field.name, Span::from(field.span.clone()))? {
                            Composition::Multi(size, pattern) => {
                                let vector = ctx.apply_load_rule(TypedExpression {
                                    handle,
                                    is_reference,
                                });

                                (
                                    crate::Expression::Swizzle {
                                        size,
                                        vector,
                                        pattern,
                                    },
                                    false,
                                )
                            }
                            Composition::Single(index) => (
                                crate::Expression::AccessIndex {
                                    base: handle,
                                    index,
                                },
                                is_reference,
                            ),
                        }
                    }
                    _ => return Err(Error::BadAccessor(field.span.clone())),
                };

                access
            }
            ast::Expression::Bitcast { expr, ref to } => {
                let expr = self.lower_expression(expr, ctx.reborrow())?;
                let to_resolved = self.resolve_type(
                    to,
                    OutputContext {
                        read_expressions: None,
                        global_expressions: ctx.global_expressions,
                        globals: ctx.globals,
                        module: ctx.module,
                    },
                )?;

                let kind = match ctx.module.types[to_resolved].inner {
                    crate::TypeInner::Scalar { kind, .. } => kind,
                    crate::TypeInner::Vector { kind, .. } => kind,
                    _ => {
                        let ty = ctx.resolve_type(expr)?;
                        return Err(Error::BadTypeCast {
                            from_type: format!("{}", ctx.fmt_ty(ty)),
                            span: to.span.clone(),
                            to_type: format!("{}", ctx.fmt_ty(to_resolved)),
                        });
                    }
                };

                (
                    crate::Expression::As {
                        expr,
                        kind,
                        convert: None,
                    },
                    false,
                )
            }
        };

        let handle = ctx.expressions.append(expr, span);
        Ok(TypedExpression {
            handle,
            is_reference,
        })
    }

    fn call(
        &mut self,
        span: Span,
        function: &ast::Ident<'source>,
        arguments: &[Handle<ast::Expression<'source>>],
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Option<Handle<crate::Expression>>, Error<'source>> {
        match ctx.globals.get(function.name) {
            Some(&GlobalDecl::Type(ty)) => {
                let handle = self.construct(
                    span,
                    &ConstructorType::Type(ty),
                    function.span.clone(),
                    arguments,
                    ctx.reborrow(),
                )?;
                Ok(Some(handle))
            }
            Some(&GlobalDecl::Const(_) | &GlobalDecl::Var(_)) => Err(Error::Unexpected(
                function.span.clone(),
                ExpectedToken::Function,
            )),
            Some(&GlobalDecl::EntryPoint) => Err(Error::CalledEntryPoint(function.span.clone())),
            Some(&GlobalDecl::Function(function)) => {
                let arguments = arguments
                    .iter()
                    .map(|&arg| self.lower_expression(arg, ctx.reborrow()))
                    .collect::<Result<Vec<_>, _>>()?;

                ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                let result = ctx.module.functions[function].result.is_some().then(|| {
                    ctx.expressions
                        .append(crate::Expression::CallResult(function), span)
                });
                ctx.emitter.start(ctx.expressions);
                ctx.block.push(
                    crate::Statement::Call {
                        function,
                        arguments,
                        result,
                    },
                    span,
                );

                Ok(result)
            }
            None => {
                let span = function.span.clone().into();
                let expr = if let Some(fun) = conv::map_relational_fun(function.name) {
                    let mut args = ctx.prepare_args(arguments, 1, span);
                    let argument = self.lower_expression(args.next()?, ctx.reborrow())?;
                    args.finish()?;

                    crate::Expression::Relational { fun, argument }
                } else if let Some(axis) = conv::map_derivative_axis(function.name) {
                    let mut args = ctx.prepare_args(arguments, 1, span);
                    let expr = self.lower_expression(args.next()?, ctx.reborrow())?;
                    args.finish()?;

                    crate::Expression::Derivative { axis, expr }
                } else if let Some(fun) = conv::map_standard_fun(function.name) {
                    let expected = fun.argument_count() as _;
                    let mut args = ctx.prepare_args(arguments, expected, span);

                    let arg = self.lower_expression(args.next()?, ctx.reborrow())?;
                    let arg1 = args
                        .next()
                        .map(|x| self.lower_expression(x, ctx.reborrow()))
                        .ok()
                        .transpose()?;
                    let arg2 = args
                        .next()
                        .map(|x| self.lower_expression(x, ctx.reborrow()))
                        .ok()
                        .transpose()?;
                    let arg3 = args
                        .next()
                        .map(|x| self.lower_expression(x, ctx.reborrow()))
                        .ok()
                        .transpose()?;

                    args.finish()?;

                    crate::Expression::Math {
                        fun,
                        arg,
                        arg1,
                        arg2,
                        arg3,
                    }
                } else {
                    match function.name {
                        "select" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let reject = self.lower_expression(args.next()?, ctx.reborrow())?;
                            let accept = self.lower_expression(args.next()?, ctx.reborrow())?;
                            let condition = self.lower_expression(args.next()?, ctx.reborrow())?;

                            args.finish()?;

                            crate::Expression::Select {
                                reject,
                                accept,
                                condition,
                            }
                        }
                        "arrayLength" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let expr = self.lower_expression(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            crate::Expression::ArrayLength(expr)
                        }
                        "atomicLoad" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let pointer = self.atomic_pointer(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            crate::Expression::Load { pointer }
                        }
                        "atomicStore" => {
                            let mut args = ctx.prepare_args(arguments, 2, span);
                            let pointer = self.atomic_pointer(args.next()?, ctx.reborrow())?;
                            let value = self.lower_expression(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                            ctx.emitter.start(ctx.expressions);
                            ctx.block
                                .push(crate::Statement::Store { pointer, value }, span);
                            return Ok(None);
                        }
                        "atomicAdd" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::Add,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicSub" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::Subtract,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicAnd" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::And,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicOr" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::InclusiveOr,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicXor" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::ExclusiveOr,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicMin" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::Min,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicMax" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::Max,
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicExchange" => {
                            return Ok(Some(self.atomic_helper(
                                span,
                                crate::AtomicFunction::Exchange { compare: None },
                                arguments,
                                ctx.reborrow(),
                            )?))
                        }
                        "atomicCompareExchangeWeak" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let pointer = self.atomic_pointer(args.next()?, ctx.reborrow())?;

                            let compare = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let value = args.next()?;
                            let value_span = ctx.read_expressions.get_span(value);
                            let value = self.lower_expression(value, ctx.reborrow())?;
                            let ty = ctx.resolve_type(pointer)?;

                            args.finish()?;

                            let expression = match ctx.module.types[ty].inner {
                                crate::TypeInner::Scalar { kind, width } => {
                                    crate::Expression::AtomicResult {
                                        kind,
                                        width,
                                        comparison: false,
                                    }
                                }
                                _ => {
                                    return Err(Error::InvalidAtomicOperandType(
                                        value_span.to_range().unwrap(),
                                    ))
                                }
                            };

                            let result = ctx.interrupt_emitter(expression, span.clone().into());
                            ctx.block.push(
                                crate::Statement::Atomic {
                                    pointer,
                                    fun: crate::AtomicFunction::Exchange {
                                        compare: Some(compare),
                                    },
                                    value,
                                    result,
                                },
                                span.into(),
                            );
                            return Ok(Some(result));
                        }
                        "storageBarrier" => {
                            ctx.prepare_args(arguments, 0, span).finish()?;

                            ctx.block
                                .push(crate::Statement::Barrier(crate::Barrier::STORAGE), span);
                            return Ok(None);
                        }
                        "workgroupBarrier" => {
                            ctx.prepare_args(arguments, 0, span).finish()?;

                            ctx.block
                                .push(crate::Statement::Barrier(crate::Barrier::WORK_GROUP), span);
                            return Ok(None);
                        }
                        "textureStore" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let value = self.lower_expression(args.next()?, ctx.reborrow())?;

                            args.finish()?;

                            ctx.block.extend(ctx.emitter.finish(ctx.expressions));
                            ctx.emitter.start(ctx.expressions);
                            let stmt = crate::Statement::ImageStore {
                                image,
                                coordinate,
                                array_index,
                                value,
                            };
                            ctx.block.push(stmt, span);
                            return Ok(None);
                        }
                        "textureSample" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Auto,
                                depth_ref: None,
                            }
                        }
                        "textureSampleLevel" => {
                            let mut args = ctx.prepare_args(arguments, 5, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let level = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Exact(level),
                                depth_ref: None,
                            }
                        }
                        "textureSampleBias" => {
                            let mut args = ctx.prepare_args(arguments, 5, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let bias = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Bias(bias),
                                depth_ref: None,
                            }
                        }
                        "textureSampleGrad" => {
                            let mut args = ctx.prepare_args(arguments, 6, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let x = self.lower_expression(args.next()?, ctx.reborrow())?;
                            let y = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Gradient { x, y },
                                depth_ref: None,
                            }
                        }
                        "textureSampleCompare" => {
                            let mut args = ctx.prepare_args(arguments, 5, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let reference = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Auto,
                                depth_ref: Some(reference),
                            }
                        }
                        "textureSampleCompareLevel" => {
                            let mut args = ctx.prepare_args(arguments, 5, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let reference = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: None,
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Zero,
                                depth_ref: Some(reference),
                            }
                        }
                        "textureGather" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let mut image_or_component = args.next()?;
                            let component =
                                match self.gather_component(image_or_component, ctx.reborrow())? {
                                    Some(x) => {
                                        image_or_component = args.next()?;
                                        x
                                    }
                                    None => crate::SwizzleComponent::X,
                                };

                            let image = image_or_component;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: Some(component),
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Zero,
                                depth_ref: None,
                            }
                        }
                        "textureGatherCompare" => {
                            let mut args = ctx.prepare_args(arguments, 4, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let sampler = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let sc = ctx.prepare_sampling(image, image_span.clone().into())?;
                            let array_index = if sc.arrayed {
                                Some(self.lower_expression(args.next()?, ctx.reborrow())?)
                            } else {
                                None
                            };

                            let reference = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let offset = args
                                .next()
                                .map(|arg| {
                                    self.constant(
                                        &ctx.read_expressions[arg],
                                        ctx.read_expressions.get_span(arg),
                                        OutputContext {
                                            read_expressions: Some(ctx.read_expressions),
                                            global_expressions: ctx.global_expressions,
                                            globals: ctx.globals,
                                            module: ctx.module,
                                        },
                                    )
                                })
                                .ok()
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageSample {
                                image: sc.image,
                                sampler,
                                gather: Some(crate::SwizzleComponent::X),
                                coordinate,
                                array_index,
                                offset,
                                level: crate::SampleLevel::Zero,
                                depth_ref: Some(reference),
                            }
                        }
                        "textureLoad" => {
                            let mut args = ctx.prepare_args(arguments, 3, span);

                            let image = args.next()?;
                            let image_span =
                                ctx.read_expressions.get_span(image).to_range().unwrap();
                            let image = self.lower_expression(image, ctx.reborrow())?;

                            let coordinate = self.lower_expression(args.next()?, ctx.reborrow())?;

                            let ty = ctx.resolve_type(image)?;
                            let (class, arrayed) = match ctx.module.types[ty].inner {
                                crate::TypeInner::Image { class, arrayed, .. } => (class, arrayed),
                                _ => return Err(Error::BadTexture(image_span)),
                            };
                            let array_index = arrayed
                                .then(|| self.lower_expression(args.next()?, ctx.reborrow()))
                                .transpose()?;

                            let level = class
                                .is_mipmapped()
                                .then(|| self.lower_expression(args.next()?, ctx.reborrow()))
                                .transpose()?;

                            let sample = class
                                .is_multisampled()
                                .then(|| self.lower_expression(args.next()?, ctx.reborrow()))
                                .transpose()?;

                            args.finish()?;

                            crate::Expression::ImageLoad {
                                image,
                                coordinate,
                                array_index,
                                level,
                                sample,
                            }
                        }
                        "textureDimensions" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let image = self.lower_expression(args.next()?, ctx.reborrow())?;
                            let level = args
                                .next()
                                .map(|arg| self.lower_expression(arg, ctx.reborrow()))
                                .ok()
                                .transpose()?;
                            args.finish()?;

                            crate::Expression::ImageQuery {
                                image,
                                query: crate::ImageQuery::Size { level },
                            }
                        }
                        "textureNumLevels" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let image = self.lower_expression(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            crate::Expression::ImageQuery {
                                image,
                                query: crate::ImageQuery::NumLevels,
                            }
                        }
                        "textureNumLayers" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let image = self.lower_expression(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            crate::Expression::ImageQuery {
                                image,
                                query: crate::ImageQuery::NumLayers,
                            }
                        }
                        "textureNumSamples" => {
                            let mut args = ctx.prepare_args(arguments, 1, span);
                            let image = self.lower_expression(args.next()?, ctx.reborrow())?;
                            args.finish()?;

                            crate::Expression::ImageQuery {
                                image,
                                query: crate::ImageQuery::NumSamples,
                            }
                        }
                        _ => {
                            return Err(Error::UnknownIdent(
                                function.span.clone(),
                                function.name.clone(),
                            ))
                        }
                    }
                };

                let expr = ctx.expressions.append(expr, span);
                Ok(Some(expr))
            }
        }
    }

    fn atomic_pointer(
        &mut self,
        expr: Handle<ast::Expression<'source>>,
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let span = ctx.read_expressions.get_span(expr).to_range().unwrap();
        let pointer = self.lower_expression(expr, ctx.reborrow())?;

        let ty = ctx.resolve_type(pointer)?;
        match ctx.module.types[ty].inner {
            crate::TypeInner::Pointer { base, .. } => match ctx.module.types[base].inner {
                crate::TypeInner::Atomic { .. } => Ok(pointer),
                ref other => {
                    log::error!("Pointer type to {:?} passed to atomic op", other);
                    Err(Error::InvalidAtomicPointer(span))
                }
            },
            ref other => {
                log::error!("Type {:?} passed to atomic op", other);
                Err(Error::InvalidAtomicPointer(span))
            }
        }
    }

    fn atomic_helper(
        &mut self,
        span: Span,
        fun: crate::AtomicFunction,
        args: &[Handle<ast::Expression<'source>>],
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        let mut args = ctx.prepare_args(args, 2, span);

        let pointer = self.atomic_pointer(args.next()?, ctx.reborrow())?;

        let value = args.next()?;
        let value_span = ctx.read_expressions.get_span(value);
        let value = self.lower_expression(value, ctx.reborrow())?;

        let ty = ctx.resolve_type(value)?;

        args.finish()?;

        let expression = match ctx.module.types[ty].inner {
            crate::TypeInner::Scalar { kind, width } => crate::Expression::AtomicResult {
                kind,
                width,
                comparison: false,
            },
            _ => {
                return Err(Error::InvalidAtomicOperandType(
                    value_span.to_range().unwrap(),
                ))
            }
        };

        let result = ctx.interrupt_emitter(expression, span.clone().into());
        ctx.block.push(
            crate::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            },
            span.into(),
        );
        Ok(result)
    }

    fn gather_component(
        &mut self,
        expr: Handle<ast::Expression<'source>>,
        ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Option<crate::SwizzleComponent>, Error<'source>> {
        let span = ctx.read_expressions.get_span(expr);

        let constant = match self
            .constant_inner(
                &ctx.read_expressions[expr],
                span,
                OutputContext {
                    read_expressions: Some(ctx.read_expressions),
                    globals: ctx.globals,
                    global_expressions: ctx.global_expressions,
                    module: ctx.module,
                },
            )
            .ok()
        {
            Some(ConstantOrInner::Constant(c)) => ctx.module.constants[c].inner.clone(),
            Some(ConstantOrInner::Inner(inner)) => inner,
            None => return Ok(None),
        };

        let int = match constant {
            crate::ConstantInner::Scalar {
                value: crate::ScalarValue::Sint(i),
                ..
            } if i >= 0 => i as u64,
            crate::ConstantInner::Scalar {
                value: crate::ScalarValue::Uint(i),
                ..
            } => i,
            _ => {
                return Err(Error::InvalidGatherComponent(span.to_range().unwrap()));
            }
        };

        crate::SwizzleComponent::XYZW
            .get(int as usize)
            .copied()
            .map(Some)
            .ok_or(Error::InvalidGatherComponent(span.to_range().unwrap()))
    }

    fn lower_struct(
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

            let binding = self.interpolate_default(&member.binding, ty, ctx.reborrow());

            offset = member_alignment.round_up(offset);
            struct_alignment = struct_alignment.max(member_alignment);

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
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Constant>, Error<'source>> {
        let inner = match self.constant_inner(expr, expr_span, ctx.reborrow())? {
            ConstantOrInner::Constant(c) => return Ok(c),
            ConstantOrInner::Inner(inner) => inner,
        };

        let c = ctx.module.constants.fetch_or_append(
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
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<ConstantOrInner, Error<'source>> {
        let span = expr_span.to_range().unwrap();

        let inner = match *expr {
            ast::Expression::Literal(literal) => {
                let inner = match literal {
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
                    ast::Literal::Number(x) => {
                        panic!("got abstract numeric type when not expected: {:?}", x);
                    }
                    ast::Literal::Bool(b) => crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(b),
                    },
                };

                inner
            }
            ast::Expression::Ident(ast::IdentExpr::Local(_)) => {
                return Err(Error::Unexpected(span, ExpectedToken::Constant))
            }
            ast::Expression::Ident(ast::IdentExpr::Unresolved(name)) => {
                return if let Some(global) = ctx.globals.get(name) {
                    match *global {
                        GlobalDecl::Const(handle) => Ok(ConstantOrInner::Constant(handle)),
                        _ => Err(Error::Unexpected(span, ExpectedToken::Constant)),
                    }
                } else {
                    Err(Error::UnknownIdent(span, name))
                }
            }
            ast::Expression::Construct {
                ref ty,
                ref components,
                ..
            } => self.const_construct(expr_span, ty, components, ctx.reborrow())?,
            ast::Expression::Call {
                ref function,
                ref arguments,
            } => match ctx.globals.get(function.name) {
                Some(&GlobalDecl::Type(ty)) => self.const_construct(
                    expr_span,
                    &ConstructorType::Type(ty),
                    arguments,
                    ctx.reborrow(),
                )?,
                Some(_) => return Err(Error::ConstExprUnsupported(span)),
                None => {
                    return Err(Error::UnknownIdent(
                        function.span.clone(),
                        function.name.clone(),
                    ))
                }
            },
            _ => return Err(Error::ConstExprUnsupported(span)),
        };

        Ok(ConstantOrInner::Inner(inner))
    }

    fn interpolate_default(
        &mut self,
        binding: &Option<crate::Binding>,
        ty: Handle<crate::Type>,
        ctx: OutputContext<'source, '_, '_>,
    ) -> Option<crate::Binding> {
        let mut binding = binding.clone();
        if let Some(ref mut binding) = binding {
            binding.apply_default_interpolation(&ctx.module.types[ty].inner);
        }

        binding
    }
}
