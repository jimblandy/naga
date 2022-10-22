use super::{
    ast, conv,
    errors::{Error, ExpectedToken, ParseError},
    lexer::Lexer,
    number::Number,
    NumberType, Span, Token,
};
use crate::{
    arena::{Arena, Handle},
    front::SymbolTable,
    span::Span as NagaSpan,
    FastHashSet,
};
use std::convert::TryFrom;

struct ExpressionContext<'input, 'temp, 'out> {
    expressions: &'out mut Arena<ast::Expression<'input>>,
    global_expressions: Option<&'out mut Arena<ast::Expression<'input>>>,
    local_table: &'temp mut SymbolTable<&'input str, Handle<ast::Local>>,
    locals: &'out mut Arena<ast::Local>,
    unresolved: &'out mut FastHashSet<&'input str>,
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ExpressionContext<'a, '_, '_> {
        ExpressionContext {
            expressions: self.expressions,
            global_expressions: self.global_expressions.as_mut().map(|r| &mut **r),
            local_table: self.local_table,
            locals: self.locals,
            unresolved: self.unresolved,
        }
    }

    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            ExpressionContext<'a, '_, '_>,
        ) -> Result<Handle<ast::Expression<'a>>, Error<'a>>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let start = lexer.start_byte_offset() as u32;
        let mut accumulator = parser(lexer, self.reborrow())?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            let left = accumulator;
            let right = parser(lexer, self.reborrow())?;
            let end = lexer.end_byte_offset() as u32;
            accumulator = self.expressions.append(
                ast::Expression::Binary { op, left, right },
                NagaSpan::new(start, end),
            );
        }
        Ok(accumulator)
    }

    fn global_expressions(&mut self) -> &mut Arena<ast::Expression<'a>> {
        self.global_expressions
            .as_mut()
            .map(|r| &mut **r)
            .unwrap_or(self.expressions)
    }
}

#[derive(Default)]
struct TypeAttributes {
    // Although WGSL nas no type attributes at the moment, it had them in the past
    // (`[[stride]]`) and may as well acquire some again in the future.
    // Therefore, we are leaving the plumbing in for now.
}

/// Which grammar rule we are in the midst of parsing.
///
/// This is used for error checking. `Parser` maintains a stack of
/// these and (occasionally) checks that it is being pushed and popped
/// as expected.
#[derive(Clone, Debug, PartialEq)]
enum Rule {
    Attribute,
    VariableDecl,
    TypeDecl,
    FunctionDecl,
    Block,
    Statement,
    PrimaryExpr,
    SingularExpr,
    UnaryExpr,
    GeneralExpr,
}

#[derive(Default)]
struct BindingParser {
    location: Option<u32>,
    built_in: Option<crate::BuiltIn>,
    interpolation: Option<crate::Interpolation>,
    sampling: Option<crate::Sampling>,
    invariant: bool,
}

impl BindingParser {
    fn parse<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
    ) -> Result<(), Error<'a>> {
        match name {
            "location" => {
                lexer.expect(Token::Paren('('))?;
                self.location = Some(Parser::parse_non_negative_i32_literal(lexer)?);
                lexer.expect(Token::Paren(')'))?;
            }
            "builtin" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.built_in = Some(conv::map_built_in(raw, span)?);
                lexer.expect(Token::Paren(')'))?;
            }
            "interpolate" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.interpolation = Some(conv::map_interpolation(raw, span)?);
                if lexer.skip(Token::Separator(',')) {
                    let (raw, span) = lexer.next_ident_with_span()?;
                    self.sampling = Some(conv::map_sampling(raw, span)?);
                }
                lexer.expect(Token::Paren(')'))?;
            }
            "invariant" => self.invariant = true,
            _ => return Err(Error::UnknownAttribute(name_span)),
        }
        Ok(())
    }

    const fn finish<'a>(self, span: Span) -> Result<Option<crate::Binding>, Error<'a>> {
        match (
            self.location,
            self.built_in,
            self.interpolation,
            self.sampling,
            self.invariant,
        ) {
            (None, None, None, None, false) => Ok(None),
            (Some(location), None, interpolation, sampling, false) => {
                // Before handing over the completed `Module`, we call
                // `apply_default_interpolation` to ensure that the interpolation and
                // sampling have been explicitly specified on all vertex shader output and fragment
                // shader input user bindings, so leaving them potentially `None` here is fine.
                Ok(Some(crate::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                }))
            }
            (None, Some(crate::BuiltIn::Position { .. }), None, None, invariant) => {
                Ok(Some(crate::Binding::BuiltIn(crate::BuiltIn::Position {
                    invariant,
                })))
            }
            (None, Some(built_in), None, None, false) => {
                Ok(Some(crate::Binding::BuiltIn(built_in)))
            }
            (_, _, _, _, _) => Err(Error::InconsistentBinding(span)),
        }
    }
}

struct ParsedVariable<'a> {
    name: &'a str,
    name_span: Span,
    space: Option<crate::AddressSpace>,
    ty: ast::Type<'a>,
    init: Option<Handle<ast::Expression<'a>>>,
}

pub struct Parser {
    rules: Vec<(Rule, usize)>,
}

impl Parser {
    pub fn new() -> Self {
        Parser { rules: Vec::new() }
    }

    fn reset(&mut self) {
        self.rules.clear();
    }

    fn push_rule_span(&mut self, rule: Rule, lexer: &mut Lexer<'_>) {
        self.rules.push((rule, lexer.start_byte_offset()));
    }

    fn pop_rule_span(&mut self, lexer: &Lexer<'_>) -> Span {
        let (_, initial) = self.rules.pop().unwrap();
        lexer.span_from(initial)
    }

    fn peek_rule_span(&mut self, lexer: &Lexer<'_>) -> Span {
        let &(_, initial) = self.rules.last().unwrap();
        lexer.span_from(initial)
    }

    fn parse_switch_value<'a>(lexer: &mut Lexer<'a>) -> Result<i32, Error<'a>> {
        let token_span = lexer.next();
        match token_span.0 {
            Token::Number(Ok(Number::U32(num))) => Ok(num as i32),
            Token::Number(Ok(Number::I32(num))) => Ok(num),
            Token::Number(Err(e)) => Err(Error::BadNumber(token_span.1, e)),
            _ => Err(Error::Unexpected(token_span.1, ExpectedToken::Integer)),
        }
    }

    /// Parse a non-negative signed integer literal.
    /// This is for attributes like `size`, `location` and others.
    fn parse_non_negative_i32_literal<'a>(lexer: &mut Lexer<'a>) -> Result<u32, Error<'a>> {
        match lexer.next() {
            (Token::Number(Ok(Number::I32(num))), span) => {
                u32::try_from(num).map_err(|_| Error::NegativeInt(span))
            }
            (Token::Number(Err(e)), span) => Err(Error::BadNumber(span, e)),
            other => Err(Error::Unexpected(
                other.1,
                ExpectedToken::Number(NumberType::I32),
            )),
        }
    }

    /// Parse a non-negative integer literal that may be either signed or unsigned.
    /// This is for the `workgroup_size` attribute and array lengths.
    /// Note: these values should be no larger than [`i32::MAX`], but this is not checked here.
    fn parse_generic_non_negative_int_literal<'a>(lexer: &mut Lexer<'a>) -> Result<u32, Error<'a>> {
        match lexer.next() {
            (Token::Number(Ok(Number::I32(num))), span) => {
                u32::try_from(num).map_err(|_| Error::NegativeInt(span))
            }
            (Token::Number(Ok(Number::U32(num))), _) => Ok(num),
            (Token::Number(Err(e)), span) => Err(Error::BadNumber(span, e)),
            other => Err(Error::Unexpected(
                other.1,
                ExpectedToken::Number(NumberType::I32),
            )),
        }
    }

    fn parse_constructor_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        word: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ast::ConstructorType<'a>>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(ast::ConstructorType::Scalar { kind, width }));
        }

        let partial = match word {
            "vec2" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Bi,
            },
            "vec3" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Tri,
            },
            "vec4" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Quad,
            },
            "mat2x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Bi,
            },
            "mat2x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Tri,
            },
            "mat2x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Quad,
            },
            "mat3x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Bi,
            },
            "mat3x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
            },
            "mat3x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Quad,
            },
            "mat4x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Bi,
            },
            "mat4x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
            },
            "mat4x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Quad,
            },
            "array" => ast::ConstructorType::PartialArray,
            _ => return Ok(None),
        };

        // parse component type if present
        match (lexer.peek().0, partial) {
            (Token::Paren('<'), ast::ConstructorType::PartialVector { size }) => {
                let (kind, width) = lexer.next_scalar_generic()?;
                Ok(Some(ast::ConstructorType::Vector { size, kind, width }))
            }
            (Token::Paren('<'), ast::ConstructorType::PartialMatrix { columns, rows }) => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                match kind {
                    crate::ScalarKind::Float => Ok(Some(ast::ConstructorType::Matrix {
                        columns,
                        rows,
                        width,
                    })),
                    _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
                }
            }
            (Token::Paren('<'), ast::ConstructorType::PartialArray) => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, ctx.reborrow())?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let expr = self.parse_general_expression(lexer, ctx)?;
                    ast::ArraySize::Constant(expr)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                Ok(Some(ast::ConstructorType::Array { base, size }))
            }
            (_, partial) => Ok(Some(partial)),
        }
    }

    /// Expects `name` to be consumed (not in lexer).
    fn parse_inbuilt_or_user_function<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<(ast::Ident<'a>, Vec<Handle<ast::Expression<'a>>>), Error<'a>> {
        lexer.open_arguments()?;
        let mut arguments = Vec::new();
        loop {
            if !arguments.is_empty() {
                if !lexer.next_argument()? {
                    break;
                }
            } else {
                if lexer.skip(Token::Paren(')')) {
                    break;
                }
            }
            let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
            arguments.push(arg);
        }

        Ok((
            ast::Ident {
                name,
                span: name_span,
            },
            arguments,
        ))
    }

    /// Expects [`Rule::PrimaryExpr`] or [`Rule::SingularExpr`] on top; does not pop it.
    /// Expects `name` to be consumed (not in lexer).
    fn parse_function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        assert!(self.rules.last().is_some());

        let expr = if let Some(ty) = self.parse_constructor_type(lexer, name, ctx.reborrow())? {
            lexer.open_arguments()?;
            let mut components = Vec::new();
            loop {
                if !components.is_empty() {
                    if !lexer.next_argument()? {
                        break;
                    }
                }
                let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
                components.push(arg);
            }
            lexer.close_arguments()?;

            ast::Expression::Construct { ty, components }
        } else {
            match name {
                "bitcast" => {
                    lexer.expect_generic_paren('<')?;
                    let to = self.parse_type_decl(lexer, ctx.reborrow())?;
                    lexer.expect_generic_paren('>')?;

                    lexer.open_arguments()?;
                    let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.close_arguments()?;

                    ast::Expression::Bitcast { expr, to }
                }
                // everything else can be handled later, since they can be hidden by user-defined functions.
                _ => {
                    let (function, arguments) = self.parse_inbuilt_or_user_function(
                        lexer,
                        name,
                        name_span,
                        ctx.reborrow(),
                    )?;
                    ast::Expression::Call {
                        function,
                        arguments,
                    }
                }
            }
        };

        let span = NagaSpan::from(self.peek_rule_span(lexer));
        let expr = ctx.expressions.append(expr, span);
        Ok(expr)
    }

    fn parse_ident_expr<'a>(
        &mut self,
        name: &'a str,
        ctx: ExpressionContext<'a, '_, '_>,
    ) -> ast::IdentExpr<'a> {
        match ctx.local_table.lookup(name) {
            Some(&local) => ast::IdentExpr::Local(local),
            None => {
                ctx.unresolved.insert(name);
                ast::IdentExpr::Unresolved(name)
            }
        }
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.push_rule_span(Rule::PrimaryExpr, lexer);

        let expr = match lexer.peek() {
            (Token::Paren('('), _) => {
                let _ = lexer.next();
                let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                lexer.expect(Token::Paren(')'))?;
                self.pop_rule_span(lexer);
                return Ok(expr);
            }
            (Token::Word("true"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Bool(true))
            }
            (Token::Word("false"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Bool(false))
            }
            (Token::Number(res), span) => {
                let _ = lexer.next();
                let num = res.map_err(|err| Error::BadNumber(span, err))?;
                ast::Expression::Literal(ast::Literal::Number(num))
            }
            (Token::Word(word), span) => {
                let _ = lexer.next();
                match lexer.peek().0 {
                    Token::Paren('(') => {
                        self.pop_rule_span(lexer);
                        return self.parse_function_call(lexer, word, span, ctx);
                    }
                    _ => {
                        let ident = self.parse_ident_expr(word, ctx.reborrow());
                        ast::Expression::Ident(ident)
                    }
                }
            }
            other => return Err(Error::Unexpected(other.1, ExpectedToken::PrimaryExpression)),
        };

        let span = self.pop_rule_span(lexer);
        let expr = ctx.expressions.append(expr, NagaSpan::from(span));
        Ok(expr)
    }

    fn parse_postfix<'a>(
        &mut self,
        span_start: usize,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
        expr: Handle<ast::Expression<'a>>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let mut expr = expr;

        loop {
            let expression = match lexer.peek().0 {
                Token::Separator('.') => {
                    let _ = lexer.next();
                    let (name, name_span) = lexer.next_ident_with_span()?;

                    ast::Expression::Member {
                        base: expr,
                        field: ast::Ident {
                            name,
                            span: name_span,
                        },
                    }
                }
                Token::Paren('[') => {
                    let _ = lexer.next();
                    let index = self.parse_general_expression(lexer, ctx.reborrow())?;
                    lexer.expect(Token::Paren(']'))?;

                    ast::Expression::Index { base: expr, index }
                }
                _ => break,
            };

            let span = lexer.span_from(span_start);
            expr = ctx.expressions.append(expression, NagaSpan::from(span));
        }

        Ok(expr)
    }

    /// Parse a `unary_expression`.
    fn parse_unary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.push_rule_span(Rule::UnaryExpr, lexer);
        //TODO: refactor this to avoid backing up
        let expr = match lexer.peek().0 {
            Token::Operation('-') => {
                let _ = lexer.next();
                let expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ast::Expression::Unary {
                    op: crate::UnaryOperator::Negate,
                    expr,
                };
                let span = NagaSpan::from(self.peek_rule_span(lexer));
                ctx.expressions.append(expr, span)
            }
            Token::Operation('!' | '~') => {
                let _ = lexer.next();
                let expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ast::Expression::Unary {
                    op: crate::UnaryOperator::Not,
                    expr,
                };
                let span = NagaSpan::from(self.peek_rule_span(lexer));
                ctx.expressions.append(expr, span)
            }
            Token::Operation('*') => {
                let _ = lexer.next();
                let expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ast::Expression::Deref(expr);
                let span = NagaSpan::from(self.peek_rule_span(lexer));
                ctx.expressions.append(expr, span)
            }
            Token::Operation('&') => {
                let _ = lexer.next();
                let expr = self.parse_unary_expression(lexer, ctx.reborrow())?;
                let expr = ast::Expression::AddrOf(expr);
                let span = NagaSpan::from(self.peek_rule_span(lexer));
                ctx.expressions.append(expr, span)
            }
            _ => self.parse_singular_expression(lexer, ctx.reborrow())?,
        };

        self.pop_rule_span(lexer);
        Ok(expr)
    }

    /// Parse a `singular_expression`.
    fn parse_singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let start = lexer.start_byte_offset();
        self.push_rule_span(Rule::SingularExpr, lexer);
        let primary_expr = self.parse_primary_expression(lexer, ctx.reborrow())?;
        let singular_expr = self.parse_postfix(start, lexer, ctx.reborrow(), primary_expr)?;
        self.pop_rule_span(lexer);

        Ok(singular_expr)
    }

    fn parse_equality_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        // equality_expression
        context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('=') => Some(crate::BinaryOperator::Equal),
                Token::LogicalOperation('!') => Some(crate::BinaryOperator::NotEqual),
                _ => None,
            },
            // relational_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::Paren('<') => Some(crate::BinaryOperator::Less),
                        Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                        Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                        Token::LogicalOperation('>') => Some(crate::BinaryOperator::GreaterEqual),
                        _ => None,
                    },
                    // shift_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::ShiftOperation('<') => {
                                    Some(crate::BinaryOperator::ShiftLeft)
                                }
                                Token::ShiftOperation('>') => {
                                    Some(crate::BinaryOperator::ShiftRight)
                                }
                                _ => None,
                            },
                            // additive_expression
                            |lexer, mut context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('+') => Some(crate::BinaryOperator::Add),
                                        Token::Operation('-') => {
                                            Some(crate::BinaryOperator::Subtract)
                                        }
                                        _ => None,
                                    },
                                    // multiplicative_expression
                                    |lexer, mut context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('*') => {
                                                    Some(crate::BinaryOperator::Multiply)
                                                }
                                                Token::Operation('/') => {
                                                    Some(crate::BinaryOperator::Divide)
                                                }
                                                Token::Operation('%') => {
                                                    Some(crate::BinaryOperator::Modulo)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_unary_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )
    }

    fn parse_general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.parse_general_expression_with_span(lexer, ctx.reborrow())
            .map(|(expr, _)| expr)
    }

    fn parse_general_expression_with_span<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, '_>,
    ) -> Result<(Handle<ast::Expression<'a>>, Span), Error<'a>> {
        self.push_rule_span(Rule::GeneralExpr, lexer);
        // logical_or_expression
        let handle = context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('|') => Some(crate::BinaryOperator::LogicalOr),
                _ => None,
            },
            // logical_and_expression
            |lexer, mut context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::LogicalOperation('&') => Some(crate::BinaryOperator::LogicalAnd),
                        _ => None,
                    },
                    // inclusive_or_expression
                    |lexer, mut context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('|') => Some(crate::BinaryOperator::InclusiveOr),
                                _ => None,
                            },
                            // exclusive_or_expression
                            |lexer, mut context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('^') => {
                                            Some(crate::BinaryOperator::ExclusiveOr)
                                        }
                                        _ => None,
                                    },
                                    // and_expression
                                    |lexer, mut context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('&') => {
                                                    Some(crate::BinaryOperator::And)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.parse_equality_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )?;
        Ok((handle, self.pop_rule_span(lexer)))
    }

    fn parse_variable_ident_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<(ast::Ident<'a>, ast::Type<'a>), Error<'a>> {
        let (name, span) = lexer.next_ident_with_span()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, ctx.reborrow())?;
        Ok((ast::Ident { name, span }, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<ParsedVariable<'a>, Error<'a>> {
        self.push_rule_span(Rule::VariableDecl, lexer);
        let mut space = None;

        if lexer.skip(Token::Paren('<')) {
            let (class_str, span) = lexer.next_ident_with_span()?;
            space = Some(match class_str {
                "storage" => {
                    let access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        // defaulting to `read`
                        crate::StorageAccess::LOAD
                    };
                    crate::AddressSpace::Storage { access }
                }
                _ => conv::map_address_space(class_str, span)?,
            });
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, ctx.reborrow())?;

        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.parse_general_expression(lexer, ctx.reborrow())?;
            Some(handle)
        } else {
            None
        };
        lexer.expect(Token::Separator(';'))?;
        let name_span = self.pop_rule_span(lexer);
        Ok(ParsedVariable {
            name,
            name_span,
            space,
            ty,
            init,
        })
    }

    fn parse_struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Vec<ast::StructMember<'a>>, Error<'a>> {
        let mut members = Vec::new();

        lexer.expect(Token::Paren('{'))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren('}')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let (mut size, mut align) = (None, None);
            self.push_rule_span(Rule::Attribute, lexer);
            let mut bind_parser = BindingParser::default();
            while lexer.skip(Token::Attribute) {
                match lexer.next_ident_with_span()? {
                    ("size", _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (value, span) =
                            lexer.capture_span(Self::parse_non_negative_i32_literal)?;
                        lexer.expect(Token::Paren(')'))?;
                        size = Some((value, span));
                    }
                    ("align", _) => {
                        lexer.expect(Token::Paren('('))?;
                        let (value, span) =
                            lexer.capture_span(Self::parse_non_negative_i32_literal)?;
                        lexer.expect(Token::Paren(')'))?;
                        align = Some((value, span));
                    }
                    (word, word_span) => bind_parser.parse(lexer, word, word_span)?,
                }
            }

            let bind_span = self.pop_rule_span(lexer);
            let binding = bind_parser.finish(bind_span)?;

            let (name, span) = match lexer.next() {
                (Token::Word(word), span) => (word, span),
                other => return Err(Error::Unexpected(other.1, ExpectedToken::FieldName)),
            };
            if crate::keywords::wgsl::RESERVED.contains(&name) {
                return Err(Error::ReservedKeyword(span));
            }
            lexer.expect(Token::Separator(':'))?;
            let ty = self.parse_type_decl(lexer, ctx.reborrow())?;
            ready = lexer.skip(Token::Separator(','));

            members.push(ast::StructMember {
                name: ast::Ident { name, span },
                ty,
                binding,
                size,
                align,
            });
        }

        Ok(members)
    }

    fn parse_matrix_scalar_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    ) -> Result<ast::TypeKind<'a>, Error<'a>> {
        let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
        match kind {
            crate::ScalarKind::Float => Ok(ast::TypeKind::Matrix {
                columns,
                rows,
                width,
            }),
            _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
        }
    }

    fn parse_type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        _attribute: TypeAttributes,
        word: &'a str,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ast::TypeKind<'a>>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(ast::TypeKind::Scalar { kind, width }));
        }

        Ok(Some(match word {
            "vec2" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::TypeKind::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            "vec3" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::TypeKind::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            "vec4" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::TypeKind::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            "mat2x2" => {
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Bi)?
            }
            "mat2x3" => {
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Tri)?
            }
            "mat2x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Bi,
                crate::VectorSize::Quad,
            )?,
            "mat3x2" => {
                self.parse_matrix_scalar_type(lexer, crate::VectorSize::Tri, crate::VectorSize::Bi)?
            }
            "mat3x3" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Tri,
                crate::VectorSize::Tri,
            )?,
            "mat3x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Tri,
                crate::VectorSize::Quad,
            )?,
            "mat4x2" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Bi,
            )?,
            "mat4x3" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Tri,
            )?,
            "mat4x4" => self.parse_matrix_scalar_type(
                lexer,
                crate::VectorSize::Quad,
                crate::VectorSize::Quad,
            )?,
            "atomic" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::TypeKind::Atomic { kind, width }
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let (ident, span) = lexer.next_ident_with_span()?;
                let mut space = conv::map_address_space(ident, span)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.parse_type_decl(lexer, ctx)?;
                if let crate::AddressSpace::Storage { ref mut access } = space {
                    *access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        crate::StorageAccess::LOAD
                    };
                }
                lexer.expect_generic_paren('>')?;
                ast::TypeKind::Pointer {
                    base: Box::new(base),
                    space,
                }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, ctx.reborrow())?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let size = self.parse_general_expression(
                        lexer,
                        ExpressionContext {
                            expressions: ctx.global_expressions(),
                            global_expressions: None,
                            local_table: &mut SymbolTable::default(),
                            locals: &mut Arena::new(),
                            unresolved: &mut FastHashSet::default(),
                        },
                    )?;
                    ast::ArraySize::Constant(size)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                ast::TypeKind::Array {
                    base: Box::new(base),
                    size,
                }
            }
            "binding_array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, ctx.reborrow())?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let size = self.parse_general_expression(
                        lexer,
                        ExpressionContext {
                            expressions: ctx.global_expressions(),
                            global_expressions: None,
                            local_table: &mut SymbolTable::default(),
                            locals: &mut Arena::new(),
                            unresolved: &mut FastHashSet::default(),
                        },
                    )?;
                    ast::ArraySize::Constant(size)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                ast::TypeKind::BindingArray {
                    base: Box::new(base),
                    size,
                }
            }
            "sampler" => ast::TypeKind::Sampler { comparison: false },
            "sampler_comparison" => ast::TypeKind::Sampler { comparison: true },
            "texture_1d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_1d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_3d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_multisampled_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_multisampled_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_depth_2d" => ast::TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_2d_array" => ast::TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube" => ast::TypeKind::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube_array" => ast::TypeKind::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_multisampled_2d" => ast::TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: true },
            },
            "texture_storage_1d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_1d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_3d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::TypeKind::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            _ => return Ok(None),
        }))
    }

    const fn check_texture_sample_type(
        kind: crate::ScalarKind,
        width: u8,
        span: Span,
    ) -> Result<(), Error<'static>> {
        use crate::ScalarKind::*;
        // Validate according to https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
        match (kind, width) {
            (Float | Sint | Uint, 4) => Ok(()),
            _ => Err(Error::BadTextureSampleType { span, kind, width }),
        }
    }

    /// Parse type declaration of a given name and attribute.
    fn parse_type_decl_name<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        attribute: TypeAttributes,
        ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::Type<'a>, Error<'a>> {
        let span = name_span.start..lexer.end_byte_offset();

        Ok(
            match self.parse_type_decl_impl(lexer, attribute, name, ctx)? {
                Some(kind) => ast::Type { kind, span },
                None => ast::Type {
                    kind: ast::TypeKind::User(ast::Ident {
                        name,
                        span: name_span,
                    }),
                    span,
                },
            },
        )
    }

    fn parse_type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::Type<'a>, Error<'a>> {
        self.push_rule_span(Rule::TypeDecl, lexer);
        let attribute = TypeAttributes::default();

        if lexer.skip(Token::Attribute) {
            let other = lexer.next();
            return Err(Error::Unexpected(other.1, ExpectedToken::TypeAttribute));
        }

        let (name, name_span) = lexer.next_ident_with_span()?;
        let ty = self.parse_type_decl_name(lexer, name, name_span, attribute, ctx)?;
        self.pop_rule_span(lexer);

        Ok(ty)
    }

    fn parse_assignment_op_and_rhs<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
        target: Handle<ast::Expression<'a>>,
        span_start: usize,
    ) -> Result<(), Error<'a>> {
        use crate::BinaryOperator as Bo;

        let op = lexer.next();
        if !matches!(
            op.0,
            Token::Operation('=')
                | Token::AssignmentOperation(_)
                | Token::IncrementOperation
                | Token::DecrementOperation
        ) {
            return Err(Error::Unexpected(op.1, ExpectedToken::Assignment));
        }

        let (op, value) = match op {
            (Token::Operation('='), _) => {
                let value = self.parse_general_expression(lexer, ctx.reborrow())?;
                (None, value)
            }
            (Token::AssignmentOperation(c), _) => {
                let op = match c {
                    '<' => Bo::ShiftLeft,
                    '>' => Bo::ShiftRight,
                    '+' => Bo::Add,
                    '-' => Bo::Subtract,
                    '*' => Bo::Multiply,
                    '/' => Bo::Divide,
                    '%' => Bo::Modulo,
                    '&' => Bo::And,
                    '|' => Bo::InclusiveOr,
                    '^' => Bo::ExclusiveOr,
                    // Note: `consume_token` shouldn't produce any other assignment ops
                    _ => unreachable!(),
                };

                let value = self.parse_general_expression(lexer, ctx.reborrow())?;
                (Some(op), value)
            }
            token @ (Token::IncrementOperation | Token::DecrementOperation, _) => {
                let op = match token.0 {
                    Token::IncrementOperation => Bo::Add,
                    Token::DecrementOperation => Bo::Subtract,
                    _ => unreachable!(),
                };
                let op_span = token.1;

                let value = ctx.expressions.append(
                    ast::Expression::Literal(ast::Literal::Number(Number::AbstractInt(1))),
                    op_span.into(),
                );

                (Some(op), value)
            }
            other => return Err(Error::Unexpected(other.1, ExpectedToken::SwitchItem)),
        };

        let span_end = lexer.end_byte_offset();
        block.stmts.push(ast::Statement {
            kind: ast::StatementKind::Assign { target, op, value },
            span: span_start..span_end,
        });
        Ok(())
    }

    /// Parse an assignment statement (will also parse increment and decrement statements)
    fn parse_assignment_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        let span_start = lexer.start_byte_offset();
        let target = self.parse_general_expression(lexer, ctx.reborrow())?;
        self.parse_assignment_op_and_rhs(lexer, ctx, block, target, span_start)
    }

    /// Parse a function call statement.
    /// Expects `ident` to be consumed (not in the lexer).
    fn parse_function_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ident: &'a str,
        ident_span: Span,
        mut context: ExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        self.push_rule_span(Rule::SingularExpr, lexer);

        let (function, arguments) = self.parse_inbuilt_or_user_function(
            lexer,
            ident,
            ident_span.clone(),
            context.reborrow(),
        )?;
        let span_end = lexer.end_byte_offset();

        block.stmts.push(ast::Statement {
            kind: ast::StatementKind::Call {
                function,
                arguments,
            },
            span: ident_span.start..span_end,
        });

        self.pop_rule_span(lexer);

        Ok(())
    }

    fn parse_function_call_or_assignment_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        let span_start = lexer.start_byte_offset();
        match lexer.peek() {
            (Token::Word(name), span) => {
                let _ = lexer.next();
                match lexer.peek() {
                    (Token::Paren('('), _) => {
                        self.parse_function_statement(lexer, name, span, context.reborrow(), block)
                    }
                    _ => {
                        let expr =
                            ast::Expression::Ident(self.parse_ident_expr(name, context.reborrow()));
                        let target = context.expressions.append(expr, NagaSpan::from(span));
                        self.parse_assignment_op_and_rhs(lexer, context, block, target, span_start)
                    }
                }
            }
            _ => self.parse_assignment_statement(lexer, context.reborrow(), block),
        }
    }

    fn parse_switch_case_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<(bool, ast::Block<'a>), Error<'a>> {
        let mut body = ast::Block::default();

        lexer.expect(Token::Paren('{'))?;

        ctx.local_table.push_scope();

        let fall_through = loop {
            // default statements
            if lexer.skip(Token::Word("fallthrough")) {
                lexer.expect(Token::Separator(';'))?;
                lexer.expect(Token::Paren('}'))?;
                break true;
            }
            if lexer.skip(Token::Paren('}')) {
                break false;
            }
            self.parse_statement(lexer, ctx.reborrow(), &mut body)?;
        };

        ctx.local_table.pop_scope();

        Ok((fall_through, body))
    }

    fn parse_statement<'a, 'out>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        self.push_rule_span(Rule::Statement, lexer);
        match lexer.peek() {
            (Token::Separator(';'), _) => {
                let _ = lexer.next();
                self.pop_rule_span(lexer);
                return Ok(());
            }
            (Token::Paren('{'), _) => {
                let (inner, span) = self.parse_block(lexer, ctx.reborrow())?;
                block.stmts.push(ast::Statement {
                    kind: ast::StatementKind::Block(inner),
                    span,
                });
                self.pop_rule_span(lexer);
                return Ok(());
            }
            (Token::Word(word), _) => {
                let kind = match word {
                    "_" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Operation('='))?;
                        let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                        lexer.expect(Token::Separator(';'))?;

                        ast::StatementKind::Ignore(expr)
                    }
                    "let" => {
                        let _ = lexer.next();
                        let (name, name_span) = lexer.next_ident_with_span()?;
                        if crate::keywords::wgsl::RESERVED.contains(&name) {
                            return Err(Error::ReservedKeyword(name_span));
                        }
                        let given_ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.parse_type_decl(lexer, ctx.reborrow())?;
                            Some(ty)
                        } else {
                            None
                        };
                        lexer.expect(Token::Operation('='))?;
                        let expr_id = self.parse_general_expression(lexer, ctx.reborrow())?;
                        lexer.expect(Token::Separator(';'))?;

                        let handle = ctx
                            .locals
                            .append(ast::Local, NagaSpan::from(name_span.clone()));
                        if let Some(old) = ctx.local_table.add(name, handle) {
                            return Err(Error::Redefinition {
                                previous: ctx.locals.get_span(old).to_range().unwrap(),
                                current: name_span.clone(),
                            });
                        }

                        ast::StatementKind::VarDecl(Box::new(ast::VarDecl::Let(ast::Let {
                            name: ast::Ident {
                                name,
                                span: name_span,
                            },
                            ty: given_ty,
                            init: expr_id,
                            handle,
                        })))
                    }
                    "var" => {
                        let _ = lexer.next();

                        let (name, name_span) = lexer.next_ident_with_span()?;
                        if crate::keywords::wgsl::RESERVED.contains(&name) {
                            return Err(Error::ReservedKeyword(name_span));
                        }
                        let ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.parse_type_decl(lexer, ctx.reborrow())?;
                            Some(ty)
                        } else {
                            None
                        };

                        let init = if lexer.skip(Token::Operation('=')) {
                            let init = self.parse_general_expression(lexer, ctx.reborrow())?;
                            Some(init)
                        } else {
                            None
                        };

                        lexer.expect(Token::Separator(';'))?;

                        let handle = ctx
                            .locals
                            .append(ast::Local, NagaSpan::from(name_span.clone()));
                        if let Some(old) = ctx.local_table.add(name, handle) {
                            return Err(Error::Redefinition {
                                previous: ctx.locals.get_span(old).to_range().unwrap(),
                                current: name_span.clone(),
                            });
                        }

                        ast::StatementKind::VarDecl(Box::new(ast::VarDecl::Var(
                            ast::LocalVariable {
                                name: ast::Ident {
                                    name,
                                    span: name_span,
                                },
                                ty,
                                init,
                                handle,
                            },
                        )))
                    }
                    "return" => {
                        let _ = lexer.next();
                        let value = if lexer.peek().0 != Token::Separator(';') {
                            let handle = self.parse_general_expression(lexer, ctx.reborrow())?;
                            Some(handle)
                        } else {
                            None
                        };
                        lexer.expect(Token::Separator(';'))?;
                        ast::StatementKind::Return { value }
                    }
                    "if" => {
                        let _ = lexer.next();
                        let condition = self.parse_general_expression(lexer, ctx.reborrow())?;

                        let accept = self.parse_block(lexer, ctx.reborrow())?.0;

                        let mut elsif_stack = Vec::new();
                        let mut elseif_span_start = lexer.start_byte_offset();
                        let mut reject = loop {
                            if !lexer.skip(Token::Word("else")) {
                                break ast::Block::default();
                            }

                            if !lexer.skip(Token::Word("if")) {
                                // ... else { ... }
                                break self.parse_block(lexer, ctx.reborrow())?.0;
                            }

                            // ... else if (...) { ... }
                            let other_condition =
                                self.parse_general_expression(lexer, ctx.reborrow())?;
                            let other_block = self.parse_block(lexer, ctx.reborrow())?;
                            elsif_stack.push((elseif_span_start, other_condition, other_block));
                            elseif_span_start = lexer.start_byte_offset();
                        };

                        let span_end = lexer.end_byte_offset();
                        // reverse-fold the else-if blocks
                        //Note: we may consider uplifting this to the IR
                        for (other_span_start, other_cond, other_block) in
                            elsif_stack.into_iter().rev()
                        {
                            let sub_stmt = ast::StatementKind::If {
                                condition: other_cond,
                                accept: other_block.0,
                                reject,
                            };
                            reject = ast::Block::default();
                            reject.stmts.push(ast::Statement {
                                kind: sub_stmt,
                                span: other_span_start..span_end,
                            })
                        }

                        ast::StatementKind::If {
                            condition,
                            accept,
                            reject,
                        }
                    }
                    "switch" => {
                        let _ = lexer.next();
                        let selector = self.parse_general_expression(lexer, ctx.reborrow())?;
                        lexer.expect(Token::Paren('{'))?;
                        let mut cases = Vec::new();

                        loop {
                            // cases + default
                            match lexer.next() {
                                (Token::Word("case"), _) => {
                                    // parse a list of values
                                    let value = loop {
                                        let value = Self::parse_switch_value(lexer)?;
                                        if lexer.skip(Token::Separator(',')) {
                                            if lexer.skip(Token::Separator(':')) {
                                                break value;
                                            }
                                        } else {
                                            lexer.skip(Token::Separator(':'));
                                            break value;
                                        }
                                        cases.push(ast::SwitchCase {
                                            value: crate::SwitchValue::Integer(value),
                                            body: ast::Block::default(),
                                            fall_through: true,
                                        });
                                    };

                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, ctx.reborrow())?;

                                    cases.push(ast::SwitchCase {
                                        value: crate::SwitchValue::Integer(value),
                                        body,
                                        fall_through,
                                    });
                                }
                                (Token::Word("default"), _) => {
                                    lexer.skip(Token::Separator(':'));
                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, ctx.reborrow())?;
                                    cases.push(ast::SwitchCase {
                                        value: crate::SwitchValue::Default,
                                        body,
                                        fall_through,
                                    });
                                }
                                (Token::Paren('}'), _) => break,
                                other => {
                                    return Err(Error::Unexpected(
                                        other.1,
                                        ExpectedToken::SwitchItem,
                                    ))
                                }
                            }
                        }

                        ast::StatementKind::Switch { selector, cases }
                    }
                    "loop" => self.parse_loop(lexer, ctx.reborrow())?,
                    "while" => {
                        let _ = lexer.next();
                        let mut body = ast::Block::default();

                        let (condition, span) = lexer.capture_span(|lexer| {
                            let condition = self.parse_general_expression(lexer, ctx.reborrow())?;
                            lexer.expect(Token::Paren('{'))?;
                            Ok(condition)
                        })?;
                        let mut reject = ast::Block::default();
                        reject.stmts.push(ast::Statement {
                            kind: ast::StatementKind::Break,
                            span: span.clone(),
                        });

                        body.stmts.push(ast::Statement {
                            kind: ast::StatementKind::If {
                                condition,
                                accept: ast::Block::default(),
                                reject,
                            },
                            span,
                        });

                        ctx.local_table.push_scope();

                        while !lexer.skip(Token::Paren('}')) {
                            self.parse_statement(lexer, ctx.reborrow(), &mut body)?;
                        }

                        ctx.local_table.pop_scope();

                        ast::StatementKind::Loop {
                            body,
                            continuing: ast::Block::default(),
                            break_if: None,
                        }
                    }
                    "for" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Paren('('))?;

                        ctx.local_table.push_scope();

                        if !lexer.skip(Token::Separator(';')) {
                            let num_statements = block.stmts.len();
                            let (_, span) = lexer.capture_span(|lexer| {
                                self.parse_statement(lexer, ctx.reborrow(), block)
                            })?;

                            if block.stmts.len() != num_statements {
                                match block.stmts.last().unwrap().kind {
                                    ast::StatementKind::Call { .. }
                                    | ast::StatementKind::Assign { .. }
                                    | ast::StatementKind::VarDecl(_) => {}
                                    _ => return Err(Error::InvalidForInitializer(span)),
                                }
                            }
                        };

                        let mut body = ast::Block::default();
                        if !lexer.skip(Token::Separator(';')) {
                            let (condition, span) = lexer.capture_span(|lexer| {
                                let condition =
                                    self.parse_general_expression(lexer, ctx.reborrow())?;
                                lexer.expect(Token::Separator(';'))?;
                                Ok(condition)
                            })?;
                            let mut reject = ast::Block::default();
                            reject.stmts.push(ast::Statement {
                                kind: ast::StatementKind::Break,
                                span: span.clone(),
                            });
                            body.stmts.push(ast::Statement {
                                kind: ast::StatementKind::If {
                                    condition,
                                    accept: ast::Block::default(),
                                    reject,
                                },
                                span,
                            });
                        };

                        let mut continuing = ast::Block::default();
                        if !lexer.skip(Token::Paren(')')) {
                            self.parse_function_call_or_assignment_statement(
                                lexer,
                                ctx.reborrow(),
                                &mut continuing,
                            )?;
                            lexer.expect(Token::Paren(')'))?;
                        }
                        lexer.expect(Token::Paren('{'))?;

                        while !lexer.skip(Token::Paren('}')) {
                            self.parse_statement(lexer, ctx.reborrow(), &mut body)?;
                        }

                        ctx.local_table.pop_scope();

                        ast::StatementKind::Loop {
                            body,
                            continuing,
                            break_if: None,
                        }
                    }
                    "break" => {
                        let (_, mut span) = lexer.next();
                        // Check if the next token is an `if`, this indicates
                        // that the user tried to type out a `break if` which
                        // is illegal in this position.
                        let (peeked_token, peeked_span) = lexer.peek();
                        if let Token::Word("if") = peeked_token {
                            span.end = peeked_span.end;
                            return Err(Error::InvalidBreakIf(span));
                        }
                        ast::StatementKind::Break
                    }
                    "continue" => {
                        let _ = lexer.next();
                        ast::StatementKind::Continue
                    }
                    "discard" => {
                        let _ = lexer.next();
                        ast::StatementKind::Kill
                    }
                    // assignment or a function call
                    _ => {
                        self.parse_function_call_or_assignment_statement(
                            lexer,
                            ctx.reborrow(),
                            block,
                        )?;
                        lexer.expect(Token::Separator(';'))?;
                        self.pop_rule_span(lexer);
                        return Ok(());
                    }
                };

                let span = self.pop_rule_span(lexer);
                block.stmts.push(ast::Statement { kind, span });
            }
            _ => {
                self.parse_assignment_statement(lexer, ctx.reborrow(), block)?;
                self.pop_rule_span(lexer);
            }
        }
        Ok(())
    }

    fn parse_loop<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::StatementKind<'a>, Error<'a>> {
        let _ = lexer.next();
        let mut body = ast::Block::default();
        let mut continuing = ast::Block::default();
        let mut break_if = None;

        lexer.expect(Token::Paren('{'))?;

        ctx.local_table.push_scope();

        loop {
            if lexer.skip(Token::Word("continuing")) {
                // Branch for the `continuing` block, this must be
                // the last thing in the loop body

                // Expect a opening brace to start the continuing block
                lexer.expect(Token::Paren('{'))?;
                loop {
                    if lexer.skip(Token::Word("break")) {
                        // Branch for the `break if` statement, this statement
                        // has the form `break if <expr>;` and must be the last
                        // statement in a continuing block

                        // The break must be followed by an `if` to form
                        // the break if
                        lexer.expect(Token::Word("if"))?;

                        let condition = self.parse_general_expression(lexer, ctx.reborrow())?;
                        // Set the condition of the break if to the newly parsed
                        // expression
                        break_if = Some(condition);

                        // Expect a semicolon to close the statement
                        lexer.expect(Token::Separator(';'))?;
                        // Expect a closing brace to close the continuing block,
                        // since the break if must be the last statement
                        lexer.expect(Token::Paren('}'))?;
                        // Stop parsing the continuing block
                        break;
                    } else if lexer.skip(Token::Paren('}')) {
                        // If we encounter a closing brace it means we have reached
                        // the end of the continuing block and should stop processing
                        break;
                    } else {
                        // Otherwise try to parse a statement
                        self.parse_statement(lexer, ctx.reborrow(), &mut continuing)?;
                    }
                }
                // Since the continuing block must be the last part of the loop body,
                // we expect to see a closing brace to end the loop body
                lexer.expect(Token::Paren('}'))?;
                break;
            }
            if lexer.skip(Token::Paren('}')) {
                // If we encounter a closing brace it means we have reached
                // the end of the loop body and should stop processing
                break;
            }
            // Otherwise try to parse a statement
            self.parse_statement(lexer, ctx.reborrow(), &mut body)?;
        }

        ctx.local_table.pop_scope();

        Ok(ast::StatementKind::Loop {
            body,
            continuing,
            break_if,
        })
    }

    fn parse_block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ExpressionContext<'a, '_, '_>,
    ) -> Result<(ast::Block<'a>, Span), Error<'a>> {
        self.push_rule_span(Rule::Block, lexer);

        ctx.local_table.push_scope();

        let _ = lexer.next();
        let mut statements = ast::Block::default();
        while !lexer.skip(Token::Paren('}')) {
            self.parse_statement(lexer, ctx.reborrow(), &mut statements)?;
        }

        ctx.local_table.pop_scope();

        let span = self.pop_rule_span(lexer);
        Ok((statements, span))
    }

    fn parse_varying_binding<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
    ) -> Result<Option<crate::Binding>, Error<'a>> {
        let mut bind_parser = BindingParser::default();
        self.push_rule_span(Rule::Attribute, lexer);

        while lexer.skip(Token::Attribute) {
            let (word, span) = lexer.next_ident_with_span()?;
            bind_parser.parse(lexer, word, span)?;
        }

        let span = self.pop_rule_span(lexer);
        bind_parser.finish(span)
    }

    fn parse_function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        out: &mut ast::TranslationUnit<'a>,
        dependencies: &mut FastHashSet<&'a str>,
    ) -> Result<ast::Function<'a>, Error<'a>> {
        self.push_rule_span(Rule::FunctionDecl, lexer);
        // read function name
        let (fun_name, span) = lexer.next_ident_with_span()?;
        if crate::keywords::wgsl::RESERVED.contains(&fun_name) {
            return Err(Error::ReservedKeyword(span));
        }

        let mut locals = Arena::new();

        // read parameter list
        let mut arguments = Vec::new();
        lexer.expect(Token::Paren('('))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren(')')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let binding = self.parse_varying_binding(lexer)?;
            let (param_name, param_type) = self.parse_variable_ident_decl(
                lexer,
                ExpressionContext {
                    expressions: &mut out.global_expressions,
                    global_expressions: None,
                    local_table: &mut SymbolTable::default(),
                    locals: &mut locals,
                    unresolved: dependencies,
                },
            )?;
            if crate::keywords::wgsl::RESERVED.contains(&param_name.name) {
                return Err(Error::ReservedKeyword(param_name.span));
            }
            arguments.push(ast::FunctionArgument {
                name: param_name,
                ty: param_type,
                binding,
            });
            ready = lexer.skip(Token::Separator(','));
        }
        // read return type
        let result = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let binding = self.parse_varying_binding(lexer)?;
            let ty = self.parse_type_decl(
                lexer,
                ExpressionContext {
                    expressions: &mut out.global_expressions,
                    global_expressions: None,
                    local_table: &mut SymbolTable::default(),
                    locals: &mut locals,
                    unresolved: dependencies,
                },
            )?;
            Some(ast::FunctionResult { ty, binding })
        } else {
            None
        };

        let mut expressions = Arena::new();
        // read body
        let body = self
            .parse_block(
                lexer,
                ExpressionContext {
                    expressions: &mut expressions,
                    global_expressions: Some(&mut out.global_expressions),
                    local_table: &mut SymbolTable::default(),
                    locals: &mut locals,
                    unresolved: dependencies,
                },
            )?
            .0;

        let fun = ast::Function {
            entry_point: None,
            name: ast::Ident {
                name: fun_name,
                span,
            },
            arguments,
            result,
            expressions,
            body,
            locals,
        };

        // done
        self.pop_rule_span(lexer);

        Ok(fun)
    }

    fn parse_global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        out: &mut ast::TranslationUnit<'a>,
    ) -> Result<bool, Error<'a>> {
        // read attributes
        let mut binding = None;
        let mut stage = None;
        let mut workgroup_size = [0u32; 3];
        let mut early_depth_test = None;
        let (mut bind_index, mut bind_group) = (None, None);

        self.push_rule_span(Rule::Attribute, lexer);
        while lexer.skip(Token::Attribute) {
            match lexer.next_ident_with_span()? {
                ("binding", _) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_index = Some(Self::parse_non_negative_i32_literal(lexer)?);
                    lexer.expect(Token::Paren(')'))?;
                }
                ("group", _) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_group = Some(Self::parse_non_negative_i32_literal(lexer)?);
                    lexer.expect(Token::Paren(')'))?;
                }
                ("vertex", _) => {
                    stage = Some(crate::ShaderStage::Vertex);
                }
                ("fragment", _) => {
                    stage = Some(crate::ShaderStage::Fragment);
                }
                ("compute", _) => {
                    stage = Some(crate::ShaderStage::Compute);
                }
                ("workgroup_size", _) => {
                    lexer.expect(Token::Paren('('))?;
                    workgroup_size = [1u32; 3];
                    for (i, size) in workgroup_size.iter_mut().enumerate() {
                        *size = Self::parse_generic_non_negative_int_literal(lexer)?;
                        match lexer.next() {
                            (Token::Paren(')'), _) => break,
                            (Token::Separator(','), _) if i != 2 => (),
                            other => {
                                return Err(Error::Unexpected(
                                    other.1,
                                    ExpectedToken::WorkgroupSizeSeparator,
                                ))
                            }
                        }
                    }
                }
                ("early_depth_test", _) => {
                    let conservative = if lexer.skip(Token::Paren('(')) {
                        let (ident, ident_span) = lexer.next_ident_with_span()?;
                        let value = conv::map_conservative_depth(ident, ident_span)?;
                        lexer.expect(Token::Paren(')'))?;
                        Some(value)
                    } else {
                        None
                    };
                    early_depth_test = Some(crate::EarlyDepthTest { conservative });
                }
                (_, word_span) => return Err(Error::UnknownAttribute(word_span)),
            }
        }

        let attrib_span = self.pop_rule_span(lexer);
        match (bind_group, bind_index) {
            (Some(group), Some(index)) => {
                binding = Some(crate::ResourceBinding {
                    group,
                    binding: index,
                });
            }
            (Some(_), None) => return Err(Error::MissingAttribute("binding", attrib_span)),
            (None, Some(_)) => return Err(Error::MissingAttribute("group", attrib_span)),
            (None, None) => {}
        }

        let mut dependencies = FastHashSet::default();

        // read item
        let start = lexer.start_byte_offset();
        let decl = match lexer.next() {
            (Token::Separator(';'), _) => None,
            (Token::Word("struct"), _) => {
                let (name, span) = lexer.next_ident_with_span()?;
                if crate::keywords::wgsl::RESERVED.contains(&name) {
                    return Err(Error::ReservedKeyword(span));
                }
                let members = self.parse_struct_body(
                    lexer,
                    ExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                let type_span = lexer.span_from(start);
                Some(ast::GlobalDecl {
                    kind: ast::GlobalDeclKind::Struct(ast::Struct {
                        name: ast::Ident { name, span },
                        members,
                    }),
                    dependencies,
                    span: type_span,
                })
            }
            (Token::Word("type"), _) => {
                let (name, span) = lexer.next_ident_with_span()?;
                lexer.expect(Token::Operation('='))?;
                let ty = self.parse_type_decl(
                    lexer,
                    ExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                lexer.expect(Token::Separator(';'))?;
                let type_span = lexer.span_from(start);

                Some(ast::GlobalDecl {
                    kind: ast::GlobalDeclKind::Type(ast::TypeAlias {
                        name: ast::Ident { name, span },
                        ty,
                    }),
                    dependencies,
                    span: type_span,
                })
            }
            (Token::Word("const"), _) => {
                let (name, name_span) = lexer.next_ident_with_span()?;
                if crate::keywords::wgsl::RESERVED.contains(&name) {
                    return Err(Error::ReservedKeyword(name_span));
                }

                let ty = if lexer.skip(Token::Separator(':')) {
                    let ty = self.parse_type_decl(
                        lexer,
                        ExpressionContext {
                            expressions: &mut out.global_expressions,
                            global_expressions: None,
                            local_table: &mut SymbolTable::default(),
                            locals: &mut Arena::new(),
                            unresolved: &mut dependencies,
                        },
                    )?;
                    Some(ty)
                } else {
                    None
                };

                lexer.expect(Token::Operation('='))?;
                let init = self.parse_general_expression(
                    lexer,
                    ExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                lexer.expect(Token::Separator(';'))?;

                Some(ast::GlobalDecl {
                    kind: ast::GlobalDeclKind::Const(ast::Const {
                        name: ast::Ident {
                            name,
                            span: name_span,
                        },
                        ty,
                        init,
                    }),
                    dependencies,
                    span: lexer.span_from(start),
                })
            }
            (Token::Word("var"), _) => {
                let pvar = self.parse_variable_decl(
                    lexer,
                    ExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                if crate::keywords::wgsl::RESERVED.contains(&pvar.name) {
                    return Err(Error::ReservedKeyword(pvar.name_span));
                }

                let span = lexer.span_from(start);

                Some(ast::GlobalDecl {
                    kind: ast::GlobalDeclKind::Var(ast::GlobalVariable {
                        name: ast::Ident {
                            name: pvar.name,
                            span: pvar.name_span,
                        },
                        space: pvar.space.unwrap_or(crate::AddressSpace::Handle),
                        binding: binding.take(),
                        ty: pvar.ty,
                        init: pvar.init,
                    }),
                    dependencies,
                    span,
                })
            }
            (Token::Word("fn"), _) => {
                let function = self.parse_function_decl(lexer, out, &mut dependencies)?;
                Some(ast::GlobalDecl {
                    kind: ast::GlobalDeclKind::Fn(ast::Function {
                        entry_point: stage.map(|stage| ast::EntryPoint {
                            stage,
                            early_depth_test,
                            workgroup_size,
                        }),
                        ..function
                    }),
                    dependencies,
                    span: lexer.span_from(start),
                })
            }
            (Token::End, _) => return Ok(false),
            other => return Err(Error::Unexpected(other.1, ExpectedToken::GlobalItem)),
        };

        if let Some(decl) = decl {
            out.decls.push(decl);
        }

        match binding {
            None => Ok(true),
            // we had the attribute but no var?
            Some(_) => Err(Error::Other),
        }
    }

    pub fn parse<'a>(&mut self, source: &'a str) -> Result<ast::TranslationUnit<'a>, ParseError> {
        self.reset();

        let mut lexer = Lexer::new(source);
        let mut tu = ast::TranslationUnit::default();
        loop {
            match self.parse_global_decl(&mut lexer, &mut tu) {
                Err(error) => return Err(error.as_parse_error(lexer.source)),
                Ok(true) => {}
                Ok(false) => {
                    if !self.rules.is_empty() {
                        log::error!("Reached the end of file, but rule stack is not empty");
                        log::error!("Rules: {:?}", self.rules);
                        return Err(Error::Other.as_parse_error(lexer.source));
                    };
                    break;
                }
            }
        }

        Ok(tu)
    }
}
