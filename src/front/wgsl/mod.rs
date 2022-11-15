/*!
Frontend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod ast;
mod construction;
mod conv;
mod index;
mod lexer;
mod number;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::ops::Range;
use thiserror::Error;

use self::number::Number;
use crate::front::wgsl::ast::{
    ConstructorType, GlobalDeclKind, TranslationUnit, TypeKind, VarDecl,
};
use crate::front::{Emitter, Typifier};
use crate::{
    arena::Handle,
    front::SymbolTable,
    proc::{
        ensure_block_returns, Alignment, Layouter, ResolveContext, ResolveError, TypeResolution,
    },
    Arena, FastHashMap, FastHashSet, NamedExpressions, SourceLocation, Span as NagaSpan,
    UniqueArena,
};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, NoColor, StandardStream},
    },
};

use crate::front::wgsl::index::Index;
use crate::front::wgsl::lexer::Lexer;

type Span = Range<usize>;
type TokenSpan<'a> = (Token<'a>, Span);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Token<'a> {
    Separator(char),
    Paren(char),
    Attribute,
    Number(Result<Number, NumberError>),
    Word(&'a str),
    Operation(char),
    LogicalOperation(char),
    ShiftOperation(char),
    AssignmentOperation(char),
    IncrementOperation,
    DecrementOperation,
    Arrow,
    Unknown(char),
    Trivia,
    End,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NumberType {
    I32,
    U32,
    F32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ExpectedToken<'a> {
    Token(Token<'a>),
    Identifier,
    Number(NumberType),
    Integer,
    /// Expected: constant, parenthesized expression, identifier
    PrimaryExpression,
    /// Expected: assignment, increment/decrement expression
    Assignment,
    /// Expected: '}', identifier
    FieldName,
    /// Expected: attribute for a type
    TypeAttribute,
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
    /// A compile-time constant expression.
    Constant,
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
    InvalidGatherComponent(Span),
    InvalidConstructorComponentType(Span, i32),
    InvalidIdentifierUnderscore(Span),
    ReservedIdentifierPrefix(Span),
    UnknownAddressSpace(Span),
    UnknownAttribute(Span),
    UnknownBuiltin(Span),
    UnknownAccess(Span),
    UnknownIdent(Span, &'a str),
    UnknownScalarType(Span),
    UnknownType(Span),
    UnknownStorageFormat(Span),
    UnknownConservativeDepth(Span),
    SizeAttributeTooLow(Span, u32),
    AlignAttributeTooLow(Span, Alignment),
    NonPowerOfTwoAlignAttribute(Span),
    InconsistentBinding(Span),
    TypeNotConstructible(Span),
    TypeNotInferrable(Span),
    InitializationTypeMismatch(Span, String, String),
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
    WrongArgumentCount {
        span: Span,
        expected: Range<u32>,
        found: u32,
    },
    FunctionReturnsVoid(Span),
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
                    ExpectedToken::PrimaryExpression => "expression".to_string(),
                    ExpectedToken::Assignment => "assignment or increment/decrement".to_string(),
                    ExpectedToken::FieldName => "field name or a closing curly bracket to signify the end of the struct".to_string(),
                    ExpectedToken::TypeAttribute => "type attribute".to_string(),
                    ExpectedToken::SwitchItem => "switch item ('case' or 'default') or a closing curly bracket to signify the end of the switch statement ('}')".to_string(),
                    ExpectedToken::WorkgroupSizeSeparator => "workgroup size separator (',') or a closing parenthesis".to_string(),
                    ExpectedToken::GlobalItem => "global item ('struct', 'const', 'var', 'type', ';', 'fn') or the end of the file".to_string(),
                    ExpectedToken::Type => "type".to_string(),
                    ExpectedToken::Variable => "variable access".to_string(),
                    ExpectedToken::Function => "function name".to_string(),
                    ExpectedToken::Constant => "compile-time constant".to_string(),
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
            Error::InvalidGatherComponent(ref bad_span) => ParseError {
                message: format!(
                    "textureGather component '{}' doesn't exist, must be 0, 1, 2, or 3",
                    &source[bad_span.clone()]
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
            Error::InitializationTypeMismatch(ref name_span, ref expected_ty, ref got_ty) => {
                ParseError {
                    message: format!(
                        "the type of `{}` is expected to be `{}`, but got `{}`",
                        &source[name_span.clone()],
                        expected_ty,
                        got_ty,
                    ),
                    labels: vec![(
                        name_span.clone(),
                        format!("definition of `{}`", &source[name_span.clone()]).into(),
                    )],
                    notes: vec![],
                }
            }
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
                    .flat_map(|(i, &(ref ident, ref usage))| {
                        [
                            (ident.clone(), "".into()),
                            (
                                usage.clone(),
                                if i == path.len() - 1 {
                                    "ending the cycle".into()
                                } else {
                                    format!("uses `{}`", &source[ident.clone()]).into()
                                },
                            ),
                        ]
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
            Error::WrongArgumentCount {
                ref span,
                ref expected,
                found,
            } => ParseError {
                message: format!(
                    "wrong number of arguments: expected {}, found {}",
                    if expected.len() < 2 {
                        format!("{}", expected.start)
                    } else {
                        format!("{}..{}", expected.start, expected.end)
                    },
                    found
                ),
                labels: vec![(span.clone(), "wrong number of arguments".into())],
                notes: vec![],
            },
            Error::FunctionReturnsVoid(ref span) => ParseError {
                message: "function does not return any value".to_string(),
                labels: vec![(span.clone(), "".into())],
                notes: vec![
                    "perhaps you meant to call the function in a separate statement?".into(),
                ],
            },
            Error::Other => ParseError {
                message: "other error".to_string(),
                labels: vec![],
                notes: vec![],
            },
        }
    }
}

impl crate::StorageFormat {
    const fn to_wgsl(self) -> &'static str {
        use crate::StorageFormat as Sf;
        match self {
            Sf::R8Unorm => "r8unorm",
            Sf::R8Snorm => "r8snorm",
            Sf::R8Uint => "r8uint",
            Sf::R8Sint => "r8sint",
            Sf::R16Uint => "r16uint",
            Sf::R16Sint => "r16sint",
            Sf::R16Float => "r16float",
            Sf::Rg8Unorm => "rg8unorm",
            Sf::Rg8Snorm => "rg8snorm",
            Sf::Rg8Uint => "rg8uint",
            Sf::Rg8Sint => "rg8sint",
            Sf::R32Uint => "r32uint",
            Sf::R32Sint => "r32sint",
            Sf::R32Float => "r32float",
            Sf::Rg16Uint => "rg16uint",
            Sf::Rg16Sint => "rg16sint",
            Sf::Rg16Float => "rg16float",
            Sf::Rgba8Unorm => "rgba8unorm",
            Sf::Rgba8Snorm => "rgba8snorm",
            Sf::Rgba8Uint => "rgba8uint",
            Sf::Rgba8Sint => "rgba8sint",
            Sf::Rgb10a2Unorm => "rgb10a2unorm",
            Sf::Rg11b10Float => "rg11b10float",
            Sf::Rg32Uint => "rg32uint",
            Sf::Rg32Sint => "rg32sint",
            Sf::Rg32Float => "rg32float",
            Sf::Rgba16Uint => "rgba16uint",
            Sf::Rgba16Sint => "rgba16sint",
            Sf::Rgba16Float => "rgba16float",
            Sf::Rgba32Uint => "rgba32uint",
            Sf::Rgba32Sint => "rgba32sint",
            Sf::Rgba32Float => "rgba32float",
        }
    }
}

impl crate::TypeInner {
    /// Formats the type as it is written in wgsl.
    ///
    /// For example `vec3<f32>`.
    ///
    /// Note: The names of a `TypeInner::Struct` is not known. Therefore this method will simply return "struct" for them.
    fn to_wgsl(
        &self,
        types: &UniqueArena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> String {
        use crate::TypeInner as Ti;

        match *self {
            Ti::Scalar { kind, width } => kind.to_wgsl(width),
            Ti::Vector { size, kind, width } => {
                format!("vec{}<{}>", size as u32, kind.to_wgsl(width))
            }
            Ti::Matrix {
                columns,
                rows,
                width,
            } => {
                format!(
                    "mat{}x{}<{}>",
                    columns as u32,
                    rows as u32,
                    crate::ScalarKind::Float.to_wgsl(width),
                )
            }
            Ti::Atomic { kind, width } => {
                format!("atomic<{}>", kind.to_wgsl(width))
            }
            Ti::Pointer { base, .. } => {
                let base = &types[base];
                let name = base.name.as_deref().unwrap_or("unknown");
                format!("ptr<{}>", name)
            }
            Ti::ValuePointer { kind, width, .. } => {
                format!("ptr<{}>", kind.to_wgsl(width))
            }
            Ti::Array { base, size, .. } => {
                let member_type = &types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => {
                        let constant = &constants[size];
                        let size = constant
                            .name
                            .clone()
                            .unwrap_or_else(|| match constant.inner {
                                crate::ConstantInner::Scalar {
                                    value: crate::ScalarValue::Uint(size),
                                    ..
                                } => size.to_string(),
                                crate::ConstantInner::Scalar {
                                    value: crate::ScalarValue::Sint(size),
                                    ..
                                } => size.to_string(),
                                _ => "?".to_string(),
                            });
                        format!("array<{}, {}>", base, size)
                    }
                    crate::ArraySize::Dynamic => format!("array<{}>", base),
                }
            }
            Ti::Struct { .. } => {
                // TODO: Actually output the struct?
                "struct".to_string()
            }
            Ti::Image {
                dim,
                arrayed,
                class,
            } => {
                let dim_suffix = match dim {
                    crate::ImageDimension::D1 => "_1d",
                    crate::ImageDimension::D2 => "_2d",
                    crate::ImageDimension::D3 => "_3d",
                    crate::ImageDimension::Cube => "_cube",
                };
                let array_suffix = if arrayed { "_array" } else { "" };

                let class_suffix = match class {
                    crate::ImageClass::Sampled { multi: true, .. } => "_multisampled",
                    crate::ImageClass::Depth { multi: false } => "_depth",
                    crate::ImageClass::Depth { multi: true } => "_depth_multisampled",
                    crate::ImageClass::Sampled { multi: false, .. }
                    | crate::ImageClass::Storage { .. } => "",
                };

                let type_in_brackets = match class {
                    crate::ImageClass::Sampled { kind, .. } => {
                        // Note: The only valid widths are 4 bytes wide.
                        // The lexer has already verified this, so we can safely assume it here.
                        // https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                        let element_type = kind.to_wgsl(4);
                        format!("<{}>", element_type)
                    }
                    crate::ImageClass::Depth { multi: _ } => String::new(),
                    crate::ImageClass::Storage { format, access } => {
                        if access.contains(crate::StorageAccess::STORE) {
                            format!("<{},write>", format.to_wgsl())
                        } else {
                            format!("<{}>", format.to_wgsl())
                        }
                    }
                };

                format!(
                    "texture{}{}{}{}",
                    class_suffix, dim_suffix, array_suffix, type_in_brackets
                )
            }
            Ti::Sampler { .. } => "sampler".to_string(),
            Ti::BindingArray { base, size, .. } => {
                let member_type = &types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => {
                        let size = constants[size].name.as_deref().unwrap_or("unknown");
                        format!("binding_array<{}, {}>", base, size)
                    }
                    crate::ArraySize::Dynamic => format!("binding_array<{}>", base),
                }
            }
        }
    }
}

mod type_inner_tests {
    #[test]
    fn to_wgsl() {
        let mut types = crate::UniqueArena::new();
        let mut constants = crate::Arena::new();
        let c = constants.append(
            crate::Constant {
                name: Some("C".to_string()),
                specialization: None,
                inner: crate::ConstantInner::Scalar {
                    width: 4,
                    value: crate::ScalarValue::Uint(32),
                },
            },
            Default::default(),
        );

        let mytype1 = types.insert(
            crate::Type {
                name: Some("MyType1".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );
        let mytype2 = types.insert(
            crate::Type {
                name: Some("MyType2".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );

        let array = crate::TypeInner::Array {
            base: mytype1,
            stride: 4,
            size: crate::ArraySize::Constant(c),
        };
        assert_eq!(array.to_wgsl(&types, &constants), "array<MyType1, C>");

        let mat = crate::TypeInner::Matrix {
            rows: crate::VectorSize::Quad,
            columns: crate::VectorSize::Bi,
            width: 8,
        };
        assert_eq!(mat.to_wgsl(&types, &constants), "mat2x4<f64>");

        let ptr = crate::TypeInner::Pointer {
            base: mytype2,
            space: crate::AddressSpace::Storage {
                access: crate::StorageAccess::default(),
            },
        };
        assert_eq!(ptr.to_wgsl(&types, &constants), "ptr<MyType2>");

        let img1 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Sampled {
                kind: crate::ScalarKind::Float,
                multi: true,
            },
        };
        assert_eq!(
            img1.to_wgsl(&types, &constants),
            "texture_multisampled_2d<f32>"
        );

        let img2 = crate::TypeInner::Image {
            dim: crate::ImageDimension::Cube,
            arrayed: true,
            class: crate::ImageClass::Depth { multi: false },
        };
        assert_eq!(img2.to_wgsl(&types, &constants), "texture_depth_cube_array");

        let img3 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Depth { multi: true },
        };
        assert_eq!(
            img3.to_wgsl(&types, &constants),
            "texture_depth_multisampled_2d"
        );

        let array = crate::TypeInner::BindingArray {
            base: mytype1,
            size: crate::ArraySize::Constant(c),
        };
        assert_eq!(
            array.to_wgsl(&types, &constants),
            "binding_array<MyType1, C>"
        );
    }
}

impl crate::ScalarKind {
    /// Format a scalar kind+width as a type is written in wgsl.
    ///
    /// Examples: `f32`, `u64`, `bool`.
    fn to_wgsl(self, width: u8) -> String {
        let prefix = match self {
            crate::ScalarKind::Sint => "i",
            crate::ScalarKind::Uint => "u",
            crate::ScalarKind::Float => "f",
            crate::ScalarKind::Bool => return "bool".to_string(),
        };
        format!("{}{}", prefix, width * 8)
    }
}

struct ParseExpressionContext<'input, 'temp, 'out> {
    expressions: &'out mut Arena<ast::Expression<'input>>,
    global_expressions: Option<&'out mut Arena<ast::Expression<'input>>>,
    local_table: &'temp mut SymbolTable<&'input str, Handle<ast::Local>>,
    locals: &'out mut Arena<ast::Local>,
    unresolved: &'out mut FastHashSet<ast::Dependency<'input>>,
}

impl<'a> ParseExpressionContext<'a, '_, '_> {
    fn reborrow(&mut self) -> ParseExpressionContext<'a, '_, '_> {
        ParseExpressionContext {
            expressions: self.expressions,
            global_expressions: self.global_expressions.as_deref_mut(),
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
            ParseExpressionContext<'a, '_, '_>,
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
            read_expressions: self.read_expressions,
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
                            NagaSpan::UNDEFINED,
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
        span: NagaSpan,
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
        span: NagaSpan,
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
        span: NagaSpan,
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
            NagaSpan::default(),
        );
        Some(constant)
    }

    /// Format type `ty`.
    fn fmt_ty(&self, ty: Handle<crate::Type>) -> &str {
        self.module.types[ty].name.as_deref().unwrap_or("unknown")
    }
}

struct ArgumentContext<'ctx, 'source> {
    args: std::slice::Iter<'ctx, Handle<ast::Expression<'source>>>,
    min_args: u32,
    args_used: u32,
    total_args: u32,
    span: Span,
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

    fn extract_impl(name: &str, name_span: NagaSpan) -> Result<u32, Error> {
        let ch = name
            .chars()
            .next()
            .ok_or_else(|| Error::BadAccessor(name_span.to_range().unwrap()))?;
        match Self::letter_component(ch) {
            Some(sc) => Ok(sc as u32),
            None => Err(Error::BadAccessor(name_span.to_range().unwrap())),
        }
    }

    fn make(name: &str, name_span: NagaSpan) -> Result<Self, Error> {
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

pub struct Parser {
    rules: Vec<(Rule, usize)>,
}

impl Parser {
    pub const fn new() -> Self {
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

    fn parse_switch_value<'a>(
        lexer: &mut Lexer<'a>,
    ) -> Result<(ast::SwitchValue, Span), Error<'a>> {
        let token_span = lexer.next();
        match token_span.0 {
            Token::Number(Ok(Number::U32(num))) => Ok((ast::SwitchValue::U32(num), token_span.1)),
            Token::Number(Ok(Number::I32(num))) => Ok((ast::SwitchValue::I32(num), token_span.1)),
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
        span: Span,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ConstructorType<'a>>, Error<'a>> {
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
            "atomic"
            | "binding_array"
            | "sampler"
            | "sampler_comparison"
            | "texture_1d"
            | "texture_1d_array"
            | "texture_2d"
            | "texture_2d_array"
            | "texture_3d"
            | "texture_cube"
            | "texture_cube_array"
            | "texture_multisampled_2d"
            | "texture_multisampled_2d_array"
            | "texture_depth_2d"
            | "texture_depth_2d_array"
            | "texture_depth_cube"
            | "texture_depth_cube_array"
            | "texture_depth_multisampled_2d"
            | "texture_storage_1d"
            | "texture_storage_1d_array"
            | "texture_storage_2d"
            | "texture_storage_2d_array"
            | "texture_storage_3d" => return Err(Error::TypeNotConstructible(span)),
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
                    let expr = self.parse_unary_expression(
                        lexer,
                        ParseExpressionContext {
                            expressions: match ctx.global_expressions {
                                Some(x) => x,
                                None => ctx.expressions,
                            },
                            global_expressions: None,
                            ..ctx
                        },
                    )?;
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
    fn parse_arguments<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<Vec<Handle<ast::Expression<'a>>>, Error<'a>> {
        lexer.open_arguments()?;
        let mut arguments = Vec::new();
        loop {
            if !arguments.is_empty() {
                if !lexer.next_argument()? {
                    break;
                }
            } else if lexer.skip(Token::Paren(')')) {
                break;
            }
            let arg = self.parse_general_expression(lexer, ctx.reborrow())?;
            arguments.push(arg);
        }

        Ok(arguments)
    }

    /// Expects [`Rule::PrimaryExpr`] or [`Rule::SingularExpr`] on top; does not pop it.
    /// Expects `name` to be consumed (not in lexer).
    fn parse_function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        assert!(self.rules.last().is_some());

        let expr = match name {
            // bitcast looks like a function call, but it's an operator and must be handled differently.
            "bitcast" => {
                lexer.expect_generic_paren('<')?;
                let to = self.parse_type_decl(lexer, ctx.reborrow())?;
                lexer.expect_generic_paren('>')?;

                lexer.open_arguments()?;
                let expr = self.parse_general_expression(lexer, ctx.reborrow())?;
                lexer.close_arguments()?;

                ast::Expression::Bitcast { expr, to }
            }
            // everything else must be handled later, since they can be hidden by user-defined functions.
            _ => {
                let arguments = self.parse_arguments(lexer, ctx.reborrow())?;
                ast::Expression::Call {
                    function: ast::Ident {
                        name,
                        span: name_span,
                    },
                    arguments,
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
        name_span: Span,
        ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> ast::IdentExpr<'a> {
        match ctx.local_table.lookup(name) {
            Some(&local) => ast::IdentExpr::Local(local),
            None => {
                ctx.unresolved.insert(ast::Dependency {
                    ident: name,
                    usage: name_span,
                });
                ast::IdentExpr::Unresolved(name)
            }
        }
    }

    fn parse_primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
                let start = lexer.start_byte_offset();
                let _ = lexer.next();

                if let Some(ty) =
                    self.parse_constructor_type(lexer, word, span.clone(), ctx.reborrow())?
                {
                    let end = lexer.end_byte_offset();
                    let components = self.parse_arguments(lexer, ctx.reborrow())?;
                    ast::Expression::Construct {
                        ty,
                        ty_span: start..end,
                        components,
                    }
                } else if let Token::Paren('(') = lexer.peek().0 {
                    self.pop_rule_span(lexer);
                    return self.parse_function_call(lexer, word, span, ctx);
                } else if word == "bitcast" {
                    self.pop_rule_span(lexer);
                    return self.parse_function_call(lexer, word, span, ctx);
                } else {
                    let ident = self.parse_ident_expr(word, span, ctx.reborrow());
                    ast::Expression::Ident(ident)
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut context: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.parse_general_expression_with_span(lexer, ctx.reborrow())
            .map(|(expr, _)| expr)
    }

    fn parse_general_expression_with_span<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut context: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<(ast::Ident<'a>, ast::Type<'a>), Error<'a>> {
        let (name, span) = lexer.next_ident_with_span()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.parse_type_decl(lexer, ctx.reborrow())?;
        Ok((ast::Ident { name, span }, ty))
    }

    fn parse_variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
    ) -> Result<TypeKind<'a>, Error<'a>> {
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
    ) -> Result<Option<TypeKind<'a>>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(TypeKind::Scalar { kind, width }));
        }

        Ok(Some(match word {
            "vec2" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                TypeKind::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            "vec3" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                TypeKind::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            "vec4" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                TypeKind::Vector {
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
                TypeKind::Atomic { kind, width }
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
                TypeKind::Pointer {
                    base: Box::new(base),
                    space,
                }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.parse_type_decl(lexer, ctx.reborrow())?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let size = self.parse_unary_expression(
                        lexer,
                        ParseExpressionContext {
                            expressions: match ctx.global_expressions {
                                Some(x) => x,
                                None => ctx.expressions,
                            },
                            global_expressions: None,
                            ..ctx
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
                    let size = self.parse_unary_expression(
                        lexer,
                        ParseExpressionContext {
                            expressions: match ctx.global_expressions {
                                Some(x) => x,
                                None => ctx.expressions,
                            },
                            global_expressions: None,
                            ..ctx
                        },
                    )?;
                    ast::ArraySize::Constant(size)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                TypeKind::BindingArray {
                    base: Box::new(base),
                    size,
                }
            }
            "sampler" => TypeKind::Sampler { comparison: false },
            "sampler_comparison" => TypeKind::Sampler { comparison: true },
            "texture_1d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_1d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_3d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_multisampled_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_multisampled_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_depth_2d" => TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_2d_array" => TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube" => TypeKind::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube_array" => TypeKind::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_multisampled_2d" => TypeKind::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: true },
            },
            "texture_storage_1d" => {
                let (format, access) = lexer.next_format_generic()?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_1d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d" => {
                let (format, access) = lexer.next_format_generic()?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                TypeKind::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_3d" => {
                let (format, access) = lexer.next_format_generic()?;
                TypeKind::Image {
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
        ctx: ParseExpressionContext<'a, '_, '_>,
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
        ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, 'out>,
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
                    Token::IncrementOperation => ast::StatementKind::Increment,
                    Token::DecrementOperation => ast::StatementKind::Decrement,
                    _ => unreachable!(),
                };

                let span_end = lexer.end_byte_offset();
                block.stmts.push(ast::Statement {
                    kind: op(target),
                    span: span_start..span_end,
                });
                return Ok(());
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
        mut ctx: ParseExpressionContext<'a, '_, 'out>,
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
        mut context: ParseExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        self.push_rule_span(Rule::SingularExpr, lexer);

        context.unresolved.insert(ast::Dependency {
            ident,
            usage: ident_span.clone(),
        });
        let arguments = self.parse_arguments(lexer, context.reborrow())?;
        let span_end = lexer.end_byte_offset();

        block.stmts.push(ast::Statement {
            kind: ast::StatementKind::Call {
                function: ast::Ident {
                    name: ident,
                    span: ident_span.clone(),
                },
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
        mut context: ParseExpressionContext<'a, '_, 'out>,
        block: &'out mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        match lexer.peek() {
            (Token::Word(name), span) => {
                // A little hack for 2 token lookahead.
                let cloned = lexer.clone();
                let _ = lexer.next();
                match lexer.peek() {
                    (Token::Paren('('), _) => {
                        self.parse_function_statement(lexer, name, span, context.reborrow(), block)
                    }
                    _ => {
                        *lexer = cloned;
                        self.parse_assignment_statement(lexer, context.reborrow(), block)
                    }
                }
            }
            _ => self.parse_assignment_statement(lexer, context.reborrow(), block),
        }
    }

    fn parse_switch_case_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, 'out>,
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
                                current: name_span,
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
                                current: name_span,
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
                                    let (value, value_span) = loop {
                                        let (value, value_span) = Self::parse_switch_value(lexer)?;
                                        if lexer.skip(Token::Separator(',')) {
                                            if lexer.skip(Token::Separator(':')) {
                                                break (value, value_span);
                                            }
                                        } else {
                                            lexer.skip(Token::Separator(':'));
                                            break (value, value_span);
                                        }
                                        cases.push(ast::SwitchCase {
                                            value,
                                            value_span,
                                            body: ast::Block::default(),
                                            fall_through: true,
                                        });
                                    };

                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, ctx.reborrow())?;

                                    cases.push(ast::SwitchCase {
                                        value,
                                        value_span,
                                        body,
                                        fall_through,
                                    });
                                }
                                (Token::Word("default"), value_span) => {
                                    lexer.skip(Token::Separator(':'));
                                    let (fall_through, body) =
                                        self.parse_switch_case_body(lexer, ctx.reborrow())?;
                                    cases.push(ast::SwitchCase {
                                        value: ast::SwitchValue::Default,
                                        value_span,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        mut ctx: ParseExpressionContext<'a, '_, '_>,
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
        out: &mut TranslationUnit<'a>,
        dependencies: &mut FastHashSet<ast::Dependency<'a>>,
    ) -> Result<ast::Function<'a>, Error<'a>> {
        self.push_rule_span(Rule::FunctionDecl, lexer);
        // read function name
        let (fun_name, span) = lexer.next_ident_with_span()?;
        if crate::keywords::wgsl::RESERVED.contains(&fun_name) {
            return Err(Error::ReservedKeyword(span));
        }

        let mut locals = Arena::new();
        let mut local_table = SymbolTable::default();

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
                ParseExpressionContext {
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

            let handle = locals.append(ast::Local, param_name.span.clone().into());
            local_table.add(param_name.name, handle);
            arguments.push(ast::FunctionArgument {
                name: param_name,
                ty: param_type,
                binding,
                handle,
            });
            ready = lexer.skip(Token::Separator(','));
        }
        // read return type
        let result = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let binding = self.parse_varying_binding(lexer)?;
            let ty = self.parse_type_decl(
                lexer,
                ParseExpressionContext {
                    expressions: &mut out.global_expressions,
                    global_expressions: None,
                    local_table: &mut local_table,
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
                ParseExpressionContext {
                    expressions: &mut expressions,
                    global_expressions: Some(&mut out.global_expressions),
                    local_table: &mut local_table,
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
        out: &mut TranslationUnit<'a>,
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
                    ParseExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                let type_span = lexer.span_from(start);
                Some((
                    ast::GlobalDecl {
                        kind: ast::GlobalDeclKind::Struct(ast::Struct {
                            name: ast::Ident { name, span },
                            members,
                        }),
                        dependencies,
                    },
                    type_span,
                ))
            }
            (Token::Word("type"), _) => {
                let (name, span) = lexer.next_ident_with_span()?;
                lexer.expect(Token::Operation('='))?;
                let ty = self.parse_type_decl(
                    lexer,
                    ParseExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                lexer.expect(Token::Separator(';'))?;
                let type_span = lexer.span_from(start);

                Some((
                    ast::GlobalDecl {
                        kind: ast::GlobalDeclKind::Type(ast::TypeAlias {
                            name: ast::Ident { name, span },
                            ty,
                        }),
                        dependencies,
                    },
                    type_span,
                ))
            }
            (Token::Word("const"), _) => {
                let (name, name_span) = lexer.next_ident_with_span()?;
                if crate::keywords::wgsl::RESERVED.contains(&name) {
                    return Err(Error::ReservedKeyword(name_span));
                }

                let ty = if lexer.skip(Token::Separator(':')) {
                    let ty = self.parse_type_decl(
                        lexer,
                        ParseExpressionContext {
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
                    ParseExpressionContext {
                        expressions: &mut out.global_expressions,
                        global_expressions: None,
                        local_table: &mut SymbolTable::default(),
                        locals: &mut Arena::new(),
                        unresolved: &mut dependencies,
                    },
                )?;
                lexer.expect(Token::Separator(';'))?;

                Some((
                    ast::GlobalDecl {
                        kind: ast::GlobalDeclKind::Const(ast::Const {
                            name: ast::Ident {
                                name,
                                span: name_span,
                            },
                            ty,
                            init,
                        }),
                        dependencies,
                    },
                    lexer.span_from(start),
                ))
            }
            (Token::Word("var"), _) => {
                let pvar = self.parse_variable_decl(
                    lexer,
                    ParseExpressionContext {
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

                Some((
                    ast::GlobalDecl {
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
                    },
                    lexer.span_from(start),
                ))
            }
            (Token::Word("fn"), _) => {
                let function = self.parse_function_decl(lexer, out, &mut dependencies)?;
                Some((
                    ast::GlobalDecl {
                        kind: ast::GlobalDeclKind::Fn(ast::Function {
                            entry_point: stage.map(|stage| ast::EntryPoint {
                                stage,
                                early_depth_test,
                                workgroup_size,
                            }),
                            ..function
                        }),
                        dependencies,
                    },
                    lexer.span_from(start),
                ))
            }
            (Token::End, _) => return Ok(false),
            other => return Err(Error::Unexpected(other.1, ExpectedToken::GlobalItem)),
        };

        if let Some((decl, span)) = decl {
            out.decls.append(decl, span.into());
        }

        match binding {
            None => Ok(true),
            // we had the attribute but no var?
            Some(_) => Err(Error::Other),
        }
    }

    pub fn parse<'a>(&mut self, source: &'a str) -> Result<TranslationUnit<'a>, ParseError> {
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

enum GlobalDecl {
    Function(Handle<crate::Function>),
    Var(Handle<crate::GlobalVariable>),
    Const(Handle<crate::Constant>),
    Type(Handle<crate::Type>),
    EntryPoint,
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
        span: NagaSpan,
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

                    let handle = ctx.as_expression(block, &mut emitter).interrupt_emitter(
                        crate::Expression::LocalVariable(var),
                        NagaSpan::UNDEFINED,
                    );
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

                let expr = self.lower_expression_for_reference(
                    target,
                    ctx.as_expression(block, &mut emitter),
                )?;
                let mut value =
                    self.lower_expression(value, ctx.as_expression(block, &mut emitter))?;

                if !expr.is_reference {
                    let ty = if ctx.named_expressions.contains_key(&expr.handle) {
                        InvalidAssignmentType::ImmutableBinding
                    } else {
                        match ctx.expressions[expr.handle] {
                            crate::Expression::Swizzle { .. } => InvalidAssignmentType::Swizzle,
                            _ => InvalidAssignmentType::Other,
                        }
                    };

                    return Err(Error::InvalidAssignment {
                        span: ctx.read_expressions.get_span(target).to_range().unwrap(),
                        ty,
                    });
                }

                let value = match op {
                    Some(op) => {
                        let mut ctx = ctx.as_expression(block, &mut emitter);
                        let mut left = ctx.apply_load_rule(expr);
                        ctx.binary_op_splat(op, &mut left, &mut value)?;
                        ctx.expressions.append(
                            crate::Expression::Binary {
                                op,
                                left,
                                right: value,
                            },
                            NagaSpan::from(stmt.span.clone()),
                        )
                    }
                    None => value,
                };
                block.extend(emitter.finish(ctx.expressions));

                crate::Statement::Store {
                    pointer: expr.handle,
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
                        _ => return Err(Error::BadIncrDecrReferenceType(value_span)),
                    },
                    _ => return Err(Error::BadIncrDecrReferenceType(value_span)),
                };
                let constant_inner = crate::ConstantInner::Scalar {
                    width,
                    value: match kind {
                        crate::ScalarKind::Sint => crate::ScalarValue::Sint(1),
                        crate::ScalarKind::Uint => crate::ScalarValue::Uint(1),
                        _ => return Err(Error::BadIncrDecrReferenceType(value_span)),
                    },
                };
                let constant = ectx.module.constants.append(
                    crate::Constant {
                        name: None,
                        specialization: None,
                        inner: constant_inner,
                    },
                    NagaSpan::UNDEFINED,
                );

                let left = ectx.expressions.append(
                    crate::Expression::Load {
                        pointer: reference.handle,
                    },
                    value_span.into(),
                );
                let right = ectx
                    .interrupt_emitter(crate::Expression::Constant(constant), NagaSpan::UNDEFINED);
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

        block.push(out, NagaSpan::from(stmt.span.clone()));

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
                let handle = ctx.module.constants.fetch_or_append(
                    crate::Constant {
                        name: None,
                        specialization: None,
                        inner,
                    },
                    NagaSpan::UNDEFINED,
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
                    .ok_or_else(|| Error::FunctionReturnsVoid(function.span.clone()))?;
                return Ok(TypedExpression {
                    handle,
                    is_reference: false,
                });
            }
            ast::Expression::Index { base, index } => {
                let expr = self.lower_expression_for_reference(base, ctx.reborrow())?;
                let index = self.lower_expression(index, ctx.reborrow())?;

                let ty = ctx.resolve_type(expr.handle)?;
                let wgsl_pointer =
                    ctx.module.types[ty].inner.pointer_space().is_some() && !expr.is_reference;

                if wgsl_pointer {
                    return Err(Error::Pointer(
                        "the value indexed by a `[]` subscripting expression",
                        ctx.read_expressions.get_span(base).to_range().unwrap(),
                    ));
                }

                if let crate::Expression::Constant(constant) = ctx.expressions[index] {
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
                            base: expr.handle,
                            index,
                        },
                        expr.is_reference,
                    )
                } else {
                    (
                        crate::Expression::Access {
                            base: expr.handle,
                            index,
                        },
                        expr.is_reference,
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
                        ctx.read_expressions.get_span(base).to_range().unwrap(),
                    ));
                }

                let access = match *composite {
                    crate::TypeInner::Struct { ref members, .. } => {
                        let index = members
                            .iter()
                            .position(|m| m.name.as_deref() == Some(field.name))
                            .ok_or_else(|| Error::BadAccessor(field.span.clone()))?
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
                        match Composition::make(field.name, NagaSpan::from(field.span.clone()))? {
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
                            from_type: ctx.fmt_ty(ty).to_string(),
                            span: to.span.clone(),
                            to_type: ctx.fmt_ty(to_resolved).to_string(),
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
        span: NagaSpan,
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

                            let result = ctx.interrupt_emitter(expression, span);
                            ctx.block.push(
                                crate::Statement::Atomic {
                                    pointer,
                                    fun: crate::AtomicFunction::Exchange {
                                        compare: Some(compare),
                                    },
                                    value,
                                    result,
                                },
                                span,
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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

                            let sc = ctx.prepare_sampling(image, image_span.into())?;
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
                        _ => return Err(Error::UnknownIdent(function.span.clone(), function.name)),
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
        span: NagaSpan,
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

        let result = ctx.interrupt_emitter(expression, span);
        ctx.block.push(
            crate::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            },
            span,
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
            .ok_or_else(|| Error::InvalidGatherComponent(span.to_range().unwrap()))
    }

    fn lower_struct(
        &mut self,
        s: &ast::Struct<'source>,
        span: NagaSpan,
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
                let base = self.resolve_type(base.as_ref(), ctx.reborrow())?;
                crate::TypeInner::Pointer { base, space }
            }
            TypeKind::Array { ref base, size } => {
                let base = self.resolve_type(base.as_ref(), ctx.reborrow())?;
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
                let base = self.resolve_type(base.as_ref(), ctx.reborrow())?;

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
        ctx.module.types.insert(
            crate::Type {
                inner,
                name: Some(name),
            },
            NagaSpan::UNDEFINED,
        )
    }

    fn constant(
        &mut self,
        expr: &ast::Expression<'source>,
        expr_span: NagaSpan,
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
            NagaSpan::UNDEFINED,
        );
        Ok(c)
    }

    fn constant_inner(
        &mut self,
        expr: &ast::Expression<'source>,
        expr_span: NagaSpan,
        mut ctx: OutputContext<'source, '_, '_>,
    ) -> Result<ConstantOrInner, Error<'source>> {
        let span = expr_span.to_range().unwrap();

        let inner = match *expr {
            ast::Expression::Literal(literal) => match literal {
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
            },
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
                None => return Err(Error::UnknownIdent(function.span.clone(), function.name)),
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

pub fn parse_str(source: &str) -> Result<crate::Module, ParseError> {
    let tu = Parser::new().parse(source)?;
    let index = Index::generate(&tu).map_err(|x| x.as_parse_error(source))?;
    Lowerer::new(&index)
        .lower(&tu)
        .map_err(|x| x.as_parse_error(source))
}
