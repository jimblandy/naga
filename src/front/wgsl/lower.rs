use crate::front::wgsl::ast;
use crate::front::wgsl::ast::{
    ArraySize, ConstructorType, GlobalDeclKind, TranslationUnit, TypeKind, VarDecl,
};
use crate::front::wgsl::errors::{Error, ExpectedToken, InvalidAssignmentType};
use crate::front::wgsl::index::Index;
use crate::front::wgsl::number::Number;
use crate::front::{Emitter, Typifier};
use crate::proc::{ensure_block_returns, Alignment, Layouter, ResolveContext, TypeResolution};
use crate::{Arena, FastHashMap, Handle, NamedExpressions, Span};

enum GlobalDecl {
    Function(Handle<crate::Function>),
    Var(Handle<crate::GlobalVariable>),
    Const(Handle<crate::Constant>),
    Type(Handle<crate::Type>),
    EntryPoint,
}

struct OutputContext<'source, 'temp, 'out> {
    global_expressions: &'temp Arena<ast::Expression<'source>>,
    globals: &'temp mut FastHashMap<&'source str, GlobalDecl>,
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
                            global_expressions: &tu.global_expressions,
                            globals: &mut globals,
                            module: &mut module,
                        },
                    )?;
                    globals.insert(f.name.name, decl);
                }
                GlobalDeclKind::Var(ref v) => {
                    let mut ctx = OutputContext {
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
                        global_expressions: &tu.global_expressions,
                        globals: &mut globals,
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
                    let handle = self.lower_struct(
                        s,
                        span,
                        OutputContext {
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
        let mut emitter = Emitter::default();
        emitter.start(ctx.expressions);

        let out = match stmt.kind {
            ast::StatementKind::Block(ref block) => {
                let block = self.lower_block(block, ctx.reborrow())?;
                crate::Statement::Block(block)
            }
            ast::StatementKind::VarDecl(ref decl) => match **decl {
                VarDecl::Let(ref l) => {
                    let value =
                        self.lower_expression(l.init, ctx.as_expression(block, &mut emitter))?;

                    let explicit_ty =
                        l.ty.as_ref()
                            .map(|ty| {
                                self.resolve_type(
                                    ty,
                                    OutputContext {
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
                        .expressions
                        .append(crate::Expression::LocalVariable(var), Span::UNDEFINED);
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
                let expr = ctx.as_expression(block, &mut emitter);
                let condition = self.lower_expression(condition, expr)?;
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
            ast::StatementKind::Ignore(expr) => {
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
                    ast::Literal::Number(_) => {
                        unreachable!("got abstract numeric type when not expected");
                    }
                    ast::Literal::Bool(b) => crate::ConstantInner::Scalar {
                        width: 1,
                        value: crate::ScalarValue::Bool(b),
                    },
                };
                let handle = ctx.module.constants.append(
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
                if let Some(global) = ctx.globals.get(name) {
                    match *global {
                        GlobalDecl::Var(handle) => {
                            (crate::Expression::GlobalVariable(handle), true)
                        }
                        GlobalDecl::Const(handle) => (crate::Expression::Constant(handle), false),
                        _ => {
                            return Err(Error::Unexpected(
                                span.to_range().unwrap(),
                                ExpectedToken::Variable,
                            ));
                        }
                    }
                } else {
                    return Err(Error::UnknownIdent(span.to_range().unwrap(), name));
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
                let handle = self.call(span, function, arguments, ctx.reborrow())?;
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
                let base = self.lower_expression_for_reference(base, ctx.reborrow())?;

                let ty = ctx.resolve_type(base.handle)?;
                let base_ty = &ctx.module.types[ty].inner;
                let wgsl_pointer = base_ty.pointer_space().is_some() && !base.is_reference;

                if wgsl_pointer {
                    return Err(Error::Pointer(
                        "the value accessed by a `.member` expression",
                        ctx.expressions.get_span(base.handle).to_range().unwrap(),
                    ));
                }

                let access = match *base_ty {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let index = members
                            .iter()
                            .position(|m| m.name.as_deref() == Some(field.name))
                            .ok_or(Error::BadAccessor(field.span.clone()))?
                            as u32;

                        (
                            crate::Expression::AccessIndex {
                                base: base.handle,
                                index,
                            },
                            base.is_reference,
                        )
                    }
                    crate::TypeInner::Vector { .. } | crate::TypeInner::Matrix { .. } => {
                        match Composition::make(field.name, Span::from(field.span.clone()))? {
                            Composition::Multi(size, pattern) => {
                                let vector = ctx.apply_load_rule(base);

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
                                    base: base.handle,
                                    index,
                                },
                                base.is_reference,
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
                            to_type: format!("{:?}", ctx.fmt_ty(to_resolved)),
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
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
        match ctx.globals.get(function.name) {
            Some(&GlobalDecl::Type(ty)) => {
                let handle = self.construct(
                    span,
                    &ConstructorType::Type(ty),
                    function.span.clone(),
                    arguments,
                    ctx.reborrow(),
                )?;
                Ok(handle)
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
                let handle = ctx.interrupt_emitter(crate::Expression::CallResult(function), span);

                ctx.block.push(
                    crate::Statement::Call {
                        function,
                        arguments,
                        result: Some(handle),
                    },
                    span,
                );

                Ok(handle)
            }
            None => Err(Error::UnknownIdent(
                function.span.clone(),
                function.name.clone(),
            )),
        }
    }

    fn construct(
        &mut self,
        span: Span,
        constructor: &ConstructorType<'source>,
        c_span: super::Span,
        components: &[Handle<ast::Expression<'source>>],
        mut ctx: ExpressionContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Expression>, Error<'source>> {
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

        let mut octx = OutputContext {
            global_expressions: ctx.global_expressions,
            globals: ctx.globals,
            module: ctx.module,
        };
        let constructor = match *constructor {
            ConstructorType::Scalar { width, kind } => {
                let ty = self
                    .ensure_type_exists(crate::TypeInner::Scalar { width, kind }, octx.reborrow());
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::PartialVector { size } => ConcreteConstructor::PartialVector { size },
            ConstructorType::Vector { size, kind, width } => {
                let ty = self.ensure_type_exists(
                    crate::TypeInner::Vector { size, kind, width },
                    octx.reborrow(),
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
                    octx.reborrow(),
                );
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::PartialArray => ConcreteConstructor::PartialArray,
            ConstructorType::Array { ref base, size } => {
                let base = self.resolve_type(base, octx.reborrow())?;
                let size = match size {
                    ArraySize::Constant(expr) => {
                        let span = octx.global_expressions.get_span(expr);
                        let expr = &octx.global_expressions[expr];
                        crate::ArraySize::Constant(self.constant(expr, span, octx.reborrow())?)
                    }
                    ArraySize::Dynamic => crate::ArraySize::Dynamic,
                };

                self.layouter
                    .update(&octx.module.types, &octx.module.constants)
                    .unwrap();
                let ty = self.ensure_type_exists(
                    crate::TypeInner::Array {
                        base,
                        size,
                        stride: self.layouter[base].to_stride(),
                    },
                    octx.reborrow(),
                );
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
            ConstructorType::Type(ty) => {
                ConcreteConstructor::Type(ty, ctx.module.types[ty].inner.clone())
            }
        };

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
                return Err(Error::UnexpectedComponents(super::Span {
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
        ctx: OutputContext<'source, '_, '_>,
    ) -> Result<Handle<crate::Constant>, Error<'source>> {
        let inner = self.constant_inner(expr, expr_span)?;
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
    ) -> Result<crate::ConstantInner, Error<'source>> {
        match *expr {
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
