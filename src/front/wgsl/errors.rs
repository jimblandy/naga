use std::borrow::Cow;
use thiserror::Error;

use super::{NumberType, Span, Token};
use crate::{
    proc::{Alignment, ResolveError},
    SourceLocation, Span as NagaSpan,
};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, NoColor, StandardStream},
    },
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    Number(NumberType),
    Integer,
    Constant,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: assignment, increment/decrement expression
    Assignment,
    /// Expected: '}', identifier
    FieldName,
    /// Expected: attribute for a type
    TypeAttribute,
    /// Expected: ';', '{', word
    Statement,
    /// Expected: 'case', 'default', '}'
    SwitchItem,
    /// Expected: ',', ')'
    WorkgroupSizeSeparator,
    /// Expected: 'struct', 'let', 'var', 'type', ';', 'fn', eof
    GlobalItem,
    /// Expected a type.
    Type,
    /// Access of `var`, `let`, `const`.
    Variable,
    /// Access of a function
    Function,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum NumberError {
    #[error("invalid numeric literal format")]
    Invalid,
    #[error("numeric literal not representable by target type")]
    NotRepresentable,
    #[error("unimplemented f16 type")]
    UnimplementedF16,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InvalidAssignmentType {
    Other,
    Swizzle,
    ImmutableBinding,
}

#[derive(Clone, Debug)]
pub enum Error<'a> {
    Unexpected(Span, ExpectedToken<'a>),
    UnexpectedComponents(Span),
    BadNumber(Span, NumberError),
    /// A negative signed integer literal where both signed and unsigned,
    /// but only non-negative literals are allowed.
    NegativeInt(Span),
    BadU32Constant(Span),
    BadMatrixScalarKind(Span, crate::ScalarKind, u8),
    BadAccessor(Span),
    BadTexture(Span),
    BadTypeCast {
        span: Span,
        from_type: String,
        to_type: String,
    },
    BadTextureSampleType {
        span: Span,
        kind: crate::ScalarKind,
        width: u8,
    },
    BadIncrDecrReferenceType(Span),
    InvalidResolve(ResolveError),
    InvalidForInitializer(Span),
    /// A break if appeared outside of a continuing block
    InvalidBreakIf(Span),
    InvalidGatherComponent(Span, u32),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownShaderStage(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    SizeAttributeTooLow(Span, u32),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    UnknownLocalFunction(Span),
    TypeNotConstructible(Span),
    TypeNotInferrable(Span),
    InitializationTypeMismatch(Span, String),
    MissingType(Span),
    MissingAttribute(&'static str, Span),
    InvalidAtomicPointer(Span),
    InvalidAtomicOperandType(Span),
    Pointer(&'static str, Span),
    NotPointer(Span),
    NotReference(&'static str, Span),
    InvalidAssignment {
        span: Span,
        ty: InvalidAssignmentType,
    },
    ReservedKeyword(Span),
    Redefinition {
        previous: Span,
        current: Span,
    },
    RecursiveDeclaration {
        ident: Span,
        usage: Span,
    },
    CyclicDeclaration {
        ident: Span,
        path: Vec<(Span, Span)>,
    },
    ConstExprUnsupported(Span),
    InvalidSwitchValue {
        uint: bool,
        span: Span,
    },
    CalledEntryPoint(Span),
    Other,
}

impl<'a> Error<'a> {
    pub fn as_parse_error(&self, source: &'a str) -> ParseError {
        match *self {
            Error::Unexpected(ref unexpected_span, expected) => {
                let expected_str = match expected {
                    ExpectedToken::Token(token) => {
                        match token {
                            Token::Separator(c) => format!("'{}'", c),
                            Token::Paren(c) => format!("'{}'", c),
                            Token::Attribute => "@".to_string(),
                            Token::Number(_) => "number".to_string(),
                            Token::Word(s) => s.to_string(),
                            Token::Operation(c) => format!("operation ('{}')", c),
                            Token::LogicalOperation(c) => format!("logical operation ('{}')", c),
                            Token::ShiftOperation(c) => format!("bitshift ('{}{}')", c, c),
                            Token::AssignmentOperation(c) if c=='<' || c=='>' => format!("bitshift ('{}{}=')", c, c),
                            Token::AssignmentOperation(c) => format!("operation ('{}=')", c),
                            Token::IncrementOperation => "increment operation".to_string(),
                            Token::DecrementOperation => "decrement operation".to_string(),
                            Token::Arrow => "->".to_string(),
                            Token::Unknown(c) => format!("unknown ('{}')", c),
                            Token::Trivia => "trivia".to_string(),
                            Token::End => "end".to_string(),
                        }
                    }
                    ExpectedToken::Identifier => "identifier".to_string(),
                    ExpectedToken::Number(ty) => {
                        match ty {
                            NumberType::I32 => "32-bit signed integer literal",
                            NumberType::U32 => "32-bit unsigned integer literal",
                            NumberType::F32 => "32-bit floating-point literal",
                        }.to_string()
                    },
                    ExpectedToken::Integer => "unsigned/signed integer literal".to_string(),
                    ExpectedToken::Constant => "constant".to_string(),
                    ExpectedToken::PrimaryExpression => "expression".to_string(),
                    ExpectedToken::Assignment => "assignment or increment/decrement".to_string(),
                    ExpectedToken::FieldName => "field name or a closing curly bracket to signify the end of the struct".to_string(),
                    ExpectedToken::TypeAttribute => "type attribute".to_string(),
                    ExpectedToken::Statement => "statement".to_string(),
                    ExpectedToken::SwitchItem => "switch item ('case' or 'default') or a closing curly bracket to signify the end of the switch statement ('}')".to_string(),
                    ExpectedToken::WorkgroupSizeSeparator => "workgroup size separator (',') or a closing parenthesis".to_string(),
                    ExpectedToken::GlobalItem => "global item ('struct', 'const', 'var', 'type', ';', 'fn') or the end of the file".to_string(),
                    ExpectedToken::Type => "type".to_string(),
                    ExpectedToken::Variable => "variable access".to_string(),
                    ExpectedToken::Function => "function name".to_string(),
                };
                ParseError {
                    message: format!(
                        "expected {}, found '{}'",
                        expected_str,
                        &source[unexpected_span.clone()],
                    ),
                    labels: vec![(
                        unexpected_span.clone(),
                        format!("expected {}", expected_str).into(),
                    )],
                    notes: vec![],
                }
            }
            Error::UnexpectedComponents(ref bad_span) => ParseError {
                message: "unexpected components".to_string(),
                labels: vec![(bad_span.clone(), "unexpected components".into())],
                notes: vec![],
            },
            Error::BadNumber(ref bad_span, ref err) => ParseError {
                message: format!("{}: `{}`", err, &source[bad_span.clone()],),
                labels: vec![(bad_span.clone(), err.to_string().into())],
                notes: vec![],
            },
            Error::NegativeInt(ref bad_span) => ParseError {
                message: format!(
                    "expected non-negative integer literal, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected non-negative integer".into())],
                notes: vec![],
            },
            Error::BadU32Constant(ref bad_span) => ParseError {
                message: format!(
                    "expected unsigned integer constant expression, found `{}`",
                    &source[bad_span.clone()],
                ),
                labels: vec![(bad_span.clone(), "expected unsigned integer".into())],
                notes: vec![],
            },
            Error::BadMatrixScalarKind(ref span, kind, width) => ParseError {
                message: format!(
                    "matrix scalar type must be floating-point, but found `{}`",
                    kind.to_wgsl(width)
                ),
                labels: vec![(span.clone(), "must be floating-point (e.g. `f32`)".into())],
                notes: vec![],
            },
            Error::BadAccessor(ref accessor_span) => ParseError {
                message: format!(
                    "invalid field accessor `{}`",
                    &source[accessor_span.clone()],
                ),
                labels: vec![(accessor_span.clone(), "invalid accessor".into())],
                notes: vec![],
            },
            Error::UnknownIdent(ref ident_span, ident) => ParseError {
                message: format!("no definition in scope for identifier: '{}'", ident),
                labels: vec![(ident_span.clone(), "unknown identifier".into())],
                notes: vec![],
            },
            Error::UnknownScalarType(ref bad_span) => ParseError {
                message: format!("unknown scalar type: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown scalar type".into())],
                notes: vec!["Valid scalar types are f16, f32, f64, \
                             i8, i16, i32, i64, \
                             u8, u16, u32, u64, bool"
                    .into()],
            },
            Error::BadTextureSampleType {
                ref span,
                kind,
                width,
            } => ParseError {
                message: format!(
                    "texture sample type must be one of f32, i32 or u32, but found {}",
                    kind.to_wgsl(width)
                ),
                labels: vec![(span.clone(), "must be one of f32, i32 or u32".into())],
                notes: vec![],
            },
            Error::BadIncrDecrReferenceType(ref span) => ParseError {
                message:
                    "increment/decrement operation requires reference type to be one of i32 or u32"
                        .to_string(),
                labels: vec![(
                    span.clone(),
                    "must be a reference type of i32 or u32".into(),
                )],
                notes: vec![],
            },
            Error::BadTexture(ref bad_span) => ParseError {
                message: format!(
                    "expected an image, but found '{}' which is not an image",
                    &source[bad_span.clone()]
                ),
                labels: vec![(bad_span.clone(), "not an image".into())],
                notes: vec![],
            },
            Error::BadTypeCast {
                ref span,
                ref from_type,
                ref to_type,
            } => {
                let msg = format!("cannot cast a {} to a {}", from_type, to_type);
                ParseError {
                    message: msg.clone(),
                    labels: vec![(span.clone(), msg.into())],
                    notes: vec![],
                }
            }
            Error::InvalidResolve(ref resolve_error) => ParseError {
                message: resolve_error.to_string(),
                labels: vec![],
                notes: vec![],
            },
            Error::InvalidForInitializer(ref bad_span) => ParseError {
                message: format!(
                    "for(;;) initializer is not an assignment or a function call: '{}'",
                    &source[bad_span.clone()]
                ),
                labels: vec![(
                    bad_span.clone(),
                    "not an assignment or function call".into(),
                )],
                notes: vec![],
            },
            Error::InvalidBreakIf(ref bad_span) => ParseError {
                message: "A break if is only allowed in a continuing block".to_string(),
                labels: vec![(bad_span.clone(), "not in a continuing block".into())],
                notes: vec![],
            },
            Error::InvalidGatherComponent(ref bad_span, component) => ParseError {
                message: format!(
                    "textureGather component {} doesn't exist, must be 0, 1, 2, or 3",
                    component
                ),
                labels: vec![(bad_span.clone(), "invalid component".into())],
                notes: vec![],
            },
            Error::InvalidConstructorComponentType(ref bad_span, component) => ParseError {
                message: format!(
                    "invalid type for constructor component at index [{}]",
                    component
                ),
                labels: vec![(bad_span.clone(), "invalid component type".into())],
                notes: vec![],
            },
            Error::InvalidIdentifierUnderscore(ref bad_span) => ParseError {
                message: "Identifier can't be '_'".to_string(),
                labels: vec![(bad_span.clone(), "invalid identifier".into())],
                notes: vec![
                    "Use phony assignment instead ('_ =' notice the absence of 'let' or 'var')"
                        .to_string(),
                ],
            },
            Error::ReservedIdentifierPrefix(ref bad_span) => ParseError {
                message: format!(
                    "Identifier starts with a reserved prefix: '{}'",
                    &source[bad_span.clone()]
                ),
                labels: vec![(bad_span.clone(), "invalid identifier".into())],
                notes: vec![],
            },
            Error::UnknownAddressSpace(ref bad_span) => ParseError {
                message: format!("unknown address space: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown address space".into())],
                notes: vec![],
            },
            Error::UnknownAttribute(ref bad_span) => ParseError {
                message: format!("unknown attribute: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown attribute".into())],
                notes: vec![],
            },
            Error::UnknownBuiltin(ref bad_span) => ParseError {
                message: format!("unknown builtin: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown builtin".into())],
                notes: vec![],
            },
            Error::UnknownAccess(ref bad_span) => ParseError {
                message: format!("unknown access: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown access".into())],
                notes: vec![],
            },
            Error::UnknownShaderStage(ref bad_span) => ParseError {
                message: format!("unknown shader stage: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown shader stage".into())],
                notes: vec![],
            },
            Error::UnknownStorageFormat(ref bad_span) => ParseError {
                message: format!("unknown storage format: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown storage format".into())],
                notes: vec![],
            },
            Error::UnknownConservativeDepth(ref bad_span) => ParseError {
                message: format!(
                    "unknown conservative depth: '{}'",
                    &source[bad_span.clone()]
                ),
                labels: vec![(bad_span.clone(), "unknown conservative depth".into())],
                notes: vec![],
            },
            Error::UnknownType(ref bad_span) => ParseError {
                message: format!("unknown type: '{}'", &source[bad_span.clone()]),
                labels: vec![(bad_span.clone(), "unknown type".into())],
                notes: vec![],
            },
            Error::SizeAttributeTooLow(ref bad_span, min_size) => ParseError {
                message: format!("struct member size must be at least {}", min_size),
                labels: vec![(
                    bad_span.clone(),
                    format!("must be at least {}", min_size).into(),
                )],
                notes: vec![],
            },
            Error::AlignAttributeTooLow(ref bad_span, min_align) => ParseError {
                message: format!("struct member alignment must be at least {}", min_align),
                labels: vec![(
                    bad_span.clone(),
                    format!("must be at least {}", min_align).into(),
                )],
                notes: vec![],
            },
            Error::NonPowerOfTwoAlignAttribute(ref bad_span) => ParseError {
                message: "struct member alignment must be a power of 2".to_string(),
                labels: vec![(bad_span.clone(), "must be a power of 2".into())],
                notes: vec![],
            },
            Error::InconsistentBinding(ref span) => ParseError {
                message: "input/output binding is not consistent".to_string(),
                labels: vec![(
                    span.clone(),
                    "input/output binding is not consistent".into(),
                )],
                notes: vec![],
            },
            Error::UnknownLocalFunction(ref span) => ParseError {
                message: format!("unknown local function `{}`", &source[span.clone()]),
                labels: vec![(span.clone(), "unknown local function".into())],
                notes: vec![],
            },
            Error::TypeNotConstructible(ref span) => ParseError {
                message: format!("type `{}` is not constructible", &source[span.clone()]),
                labels: vec![(span.clone(), "type is not constructible".into())],
                notes: vec![],
            },
            Error::TypeNotInferrable(ref span) => ParseError {
                message: "type can't be inferred".to_string(),
                labels: vec![(span.clone(), "type can't be inferred".into())],
                notes: vec![],
            },
            Error::InitializationTypeMismatch(ref name_span, ref expected_ty) => ParseError {
                message: format!(
                    "the type of `{}` is expected to be `{}`",
                    &source[name_span.clone()],
                    expected_ty
                ),
                labels: vec![(
                    name_span.clone(),
                    format!("definition of `{}`", &source[name_span.clone()]).into(),
                )],
                notes: vec![],
            },
            Error::MissingType(ref name_span) => ParseError {
                message: format!("variable `{}` needs a type", &source[name_span.clone()]),
                labels: vec![(
                    name_span.clone(),
                    format!("definition of `{}`", &source[name_span.clone()]).into(),
                )],
                notes: vec![],
            },
            Error::MissingAttribute(name, ref name_span) => ParseError {
                message: format!(
                    "variable `{}` needs a '{}' attribute",
                    &source[name_span.clone()],
                    name
                ),
                labels: vec![(
                    name_span.clone(),
                    format!("definition of `{}`", &source[name_span.clone()]).into(),
                )],
                notes: vec![],
            },
            Error::InvalidAtomicPointer(ref span) => ParseError {
                message: "atomic operation is done on a pointer to a non-atomic".to_string(),
                labels: vec![(span.clone(), "atomic pointer is invalid".into())],
                notes: vec![],
            },
            Error::InvalidAtomicOperandType(ref span) => ParseError {
                message: "atomic operand type is inconsistent with the operation".to_string(),
                labels: vec![(span.clone(), "atomic operand type is invalid".into())],
                notes: vec![],
            },
            Error::NotPointer(ref span) => ParseError {
                message: "the operand of the `*` operator must be a pointer".to_string(),
                labels: vec![(span.clone(), "expression is not a pointer".into())],
                notes: vec![],
            },
            Error::NotReference(what, ref span) => ParseError {
                message: format!("{} must be a reference", what),
                labels: vec![(span.clone(), "expression is not a reference".into())],
                notes: vec![],
            },
            Error::InvalidAssignment { ref span, ty } => ParseError {
                message: "invalid left-hand side of assignment".into(),
                labels: vec![(span.clone(), "cannot assign to this expression".into())],
                notes: match ty {
                    InvalidAssignmentType::Swizzle => vec![
                        "WGSL does not support assignments to swizzles".into(),
                        "consider assigning each component individually".into(),
                    ],
                    InvalidAssignmentType::ImmutableBinding => vec![
                        format!("'{}' is an immutable binding", &source[span.clone()]),
                        "consider declaring it with `var` instead of `let`".into(),
                    ],
                    InvalidAssignmentType::Other => vec![],
                },
            },
            Error::Pointer(what, ref span) => ParseError {
                message: format!("{} must not be a pointer", what),
                labels: vec![(span.clone(), "expression is a pointer".into())],
                notes: vec![],
            },
            Error::ReservedKeyword(ref name_span) => ParseError {
                message: format!(
                    "name `{}` is a reserved keyword",
                    &source[name_span.clone()]
                ),
                labels: vec![(
                    name_span.clone(),
                    format!("definition of `{}`", &source[name_span.clone()]).into(),
                )],
                notes: vec![],
            },
            Error::Redefinition {
                ref previous,
                ref current,
            } => ParseError {
                message: format!("redefinition of `{}`", &source[current.clone()]),
                labels: vec![
                    (
                        current.clone(),
                        format!("redefinition of `{}`", &source[current.clone()]).into(),
                    ),
                    (
                        previous.clone(),
                        format!("previous definition of `{}`", &source[previous.clone()]).into(),
                    ),
                ],
                notes: vec![],
            },
            Error::RecursiveDeclaration {
                ref ident,
                ref usage,
            } => ParseError {
                message: format!("declaration of `{}` is recursive", &source[ident.clone()]),
                labels: vec![
                    (ident.clone(), "".into()),
                    (usage.clone(), "uses itself here".into()),
                ],
                notes: vec![],
            },
            Error::CyclicDeclaration {
                ref ident,
                ref path,
            } => ParseError {
                message: format!("declaration of `{}` is cyclic", &source[ident.clone()]),
                labels: path
                    .iter()
                    .enumerate()
                    .flat_map(|(i, (ident, usage))| {
                        [
                            (ident.clone(), "".into()),
                            (
                                usage.clone(),
                                if i == path.len() - 1 {
                                    format!("ending the cycle").into()
                                } else {
                                    format!("uses `{}`", &source[ident.clone()]).into()
                                },
                            ),
                        ]
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                    })
                    .collect(),
                notes: vec![],
            },
            Error::ConstExprUnsupported(ref span) => ParseError {
                message: "this constant expression is not supported".to_string(),
                labels: vec![(span.clone(), "expression is not supported".into())],
                notes: vec!["this should be fixed in a future version of Naga".into()],
            },
            Error::InvalidSwitchValue { uint, ref span } => ParseError {
                message: "invalid switch value".to_string(),
                labels: vec![(
                    span.clone(),
                    if uint {
                        "expected unsigned integer"
                    } else {
                        "expected signed integer"
                    }
                    .into(),
                )],
                notes: vec![if uint {
                    format!(
                        "suffix the integer with a `u`: '{}u'",
                        &source[span.clone()]
                    )
                } else {
                    format!(
                        "remove the `u` suffix: '{}'",
                        &source[span.start..span.end - 1]
                    )
                }],
            },
            Error::CalledEntryPoint(ref span) => ParseError {
                message: "entry point cannot be called".to_string(),
                labels: vec![(span.clone(), "entry point cannot be called".into())],
                notes: vec![],
            },
            Error::Other => ParseError {
                message: "other error".to_string(),
                labels: vec![],
                notes: vec![],
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParseError {
    message: String,
    labels: Vec<(Span, Cow<'static, str>)>,
    notes: Vec<String>,
}

impl ParseError {
    pub fn labels(&self) -> impl Iterator<Item = (Span, &str)> + ExactSizeIterator + '_ {
        self.labels
            .iter()
            .map(|&(ref span, ref msg)| (span.clone(), msg.as_ref()))
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    fn diagnostic(&self) -> Diagnostic<()> {
        let diagnostic = Diagnostic::error()
            .with_message(self.message.to_string())
            .with_labels(
                self.labels
                    .iter()
                    .map(|label| {
                        Label::primary((), label.0.clone()).with_message(label.1.to_string())
                    })
                    .collect(),
            )
            .with_notes(
                self.notes
                    .iter()
                    .map(|note| format!("note: {}", note))
                    .collect(),
            );
        diagnostic
    }

    /// Emits a summary of the error to standard error stream.
    pub fn emit_to_stderr(&self, source: &str) {
        self.emit_to_stderr_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to standard error stream.
    pub fn emit_to_stderr_with_path(&self, source: &str, path: &str) {
        let files = SimpleFile::new(path, source);
        let config = codespan_reporting::term::Config::default();
        let writer = StandardStream::stderr(ColorChoice::Auto);
        term::emit(&mut writer.lock(), &config, &files, &self.diagnostic())
            .expect("cannot write error");
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string(&self, source: &str) -> String {
        self.emit_to_string_with_path(source, "wgsl")
    }

    /// Emits a summary of the error to a string.
    pub fn emit_to_string_with_path(&self, source: &str, path: &str) -> String {
        let files = SimpleFile::new(path, source);
        let config = codespan_reporting::term::Config::default();
        let mut writer = NoColor::new(Vec::new());
        term::emit(&mut writer, &config, &files, &self.diagnostic()).expect("cannot write error");
        String::from_utf8(writer.into_inner()).unwrap()
    }

    /// Returns a [`SourceLocation`] for the first label in the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        self.labels
            .get(0)
            .map(|label| NagaSpan::new(label.0.start as u32, label.0.end as u32).location(source))
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
