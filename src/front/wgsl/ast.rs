use super::Number;
use crate::{Arena, FastHashSet, Handle, Span};
use std::hash::Hash;

#[derive(Debug, Default)]
pub struct TranslationUnit<'a> {
    pub decls: Arena<GlobalDecl<'a>>,
    pub global_expressions: Arena<Expression<'a>>,
    pub types: Arena<Type<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Ident<'a> {
    pub name: &'a str,
    pub span: Span,
}

#[derive(Debug)]
pub enum IdentExpr<'a> {
    Unresolved(&'a str),
    Local(Handle<Local>),
}

/// A reference to a module-scope definition or predeclared object.
///
/// Each [`GlobalDecl`] holds a set of these values, to be resolved to
/// specific definitions later. To support de-duplication, `Eq` and
/// `Hash` on a `Dependency` value consider only the name, not the
/// source location at which the reference occurs.
#[derive(Debug)]
pub struct Dependency<'a> {
    /// The name referred to.
    pub ident: &'a str,

    /// The location at which the reference to that name occurs.
    pub usage: Span,
}

impl Hash for Dependency<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ident.hash(state);
    }
}

impl PartialEq for Dependency<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ident == other.ident
    }
}

impl Eq for Dependency<'_> {}

/// A module-scope declaration.
#[derive(Debug)]
pub struct GlobalDecl<'a> {
    pub kind: GlobalDeclKind<'a>,

    /// Names of all module-scope or predeclared objects this
    /// declaration uses.
    pub dependencies: FastHashSet<Dependency<'a>>,
}

#[derive(Debug)]
pub enum GlobalDeclKind<'a> {
    Fn(Function<'a>),
    Var(GlobalVariable<'a>),
    Const(Const<'a>),
    Struct(Struct<'a>),
    Type(TypeAlias<'a>),
}

#[derive(Debug)]
pub struct FunctionArgument<'a> {
    pub name: Ident<'a>,
    pub ty: Handle<Type<'a>>,
    pub binding: Option<crate::Binding>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub struct FunctionResult<'a> {
    pub ty: Handle<Type<'a>>,
    pub binding: Option<crate::Binding>,
}

#[derive(Debug)]
pub struct EntryPoint {
    pub stage: crate::ShaderStage,
    pub early_depth_test: Option<crate::EarlyDepthTest>,
    pub workgroup_size: [u32; 3],
}

#[derive(Debug)]
pub struct Function<'a> {
    pub entry_point: Option<EntryPoint>,
    pub name: Ident<'a>,
    pub arguments: Vec<FunctionArgument<'a>>,
    pub result: Option<FunctionResult<'a>>,
    pub expressions: Arena<Expression<'a>>,
    pub locals: Arena<Local>,
    pub body: Block<'a>,
}

#[derive(Debug)]
pub struct GlobalVariable<'a> {
    pub name: Ident<'a>,
    pub space: crate::AddressSpace,
    pub binding: Option<crate::ResourceBinding>,
    pub ty: Handle<Type<'a>>,
    pub init: Option<Handle<Expression<'a>>>,
}

#[derive(Debug)]
pub struct StructMember<'a> {
    pub name: Ident<'a>,
    pub ty: Handle<Type<'a>>,
    pub binding: Option<crate::Binding>,
    pub align: Option<(u32, Span)>,
    pub size: Option<(u32, Span)>,
}

#[derive(Debug)]
pub struct Struct<'a> {
    pub name: Ident<'a>,
    pub members: Vec<StructMember<'a>>,
}

#[derive(Debug)]
pub struct TypeAlias<'a> {
    pub name: Ident<'a>,
    pub ty: Handle<Type<'a>>,
}

#[derive(Debug)]
pub struct Const<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Handle<Type<'a>>>,
    pub init: Handle<Expression<'a>>,
}

#[derive(Debug, Copy, Clone)]
pub enum ArraySize<'a> {
    Constant(Handle<Expression<'a>>),
    Dynamic,
}

#[derive(Debug)]
pub enum Type<'a> {
    Scalar {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Matrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: crate::Bytes,
    },
    Atomic {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Pointer {
        base: Handle<Type<'a>>,
        space: crate::AddressSpace,
    },
    Array {
        base: Handle<Type<'a>>,
        size: ArraySize<'a>,
    },
    Image {
        dim: crate::ImageDimension,
        arrayed: bool,
        class: crate::ImageClass,
    },
    Sampler {
        comparison: bool,
    },
    BindingArray {
        base: Handle<Type<'a>>,
        size: ArraySize<'a>,
    },
    User(Ident<'a>),
}

#[derive(Debug, Default)]
pub struct Block<'a> {
    pub stmts: Vec<Statement<'a>>,
}

#[derive(Debug)]
pub struct Statement<'a> {
    pub kind: StatementKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum StatementKind<'a> {
    LocalDecl(LocalDecl<'a>),
    Block(Block<'a>),
    If {
        condition: Handle<Expression<'a>>,
        accept: Block<'a>,
        reject: Block<'a>,
    },
    Switch {
        selector: Handle<Expression<'a>>,
        cases: Vec<SwitchCase<'a>>,
    },
    Loop {
        body: Block<'a>,
        continuing: Block<'a>,
        break_if: Option<Handle<Expression<'a>>>,
    },
    Break,
    Continue,
    Return {
        value: Option<Handle<Expression<'a>>>,
    },
    Kill,
    Call {
        function: Ident<'a>,
        arguments: Vec<Handle<Expression<'a>>>,
    },
    Assign {
        target: Handle<Expression<'a>>,
        op: Option<crate::BinaryOperator>,
        value: Handle<Expression<'a>>,
    },
    Increment(Handle<Expression<'a>>),
    Decrement(Handle<Expression<'a>>),
    Ignore(Handle<Expression<'a>>),
}

#[derive(Debug)]
pub enum SwitchValue {
    I32(i32),
    U32(u32),
    Default,
}

#[derive(Debug)]
pub struct SwitchCase<'a> {
    pub value: SwitchValue,
    pub value_span: Span,
    pub body: Block<'a>,
    pub fall_through: bool,
}

#[derive(Debug)]
pub enum ConstructorType<'a> {
    Scalar {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    PartialVector {
        size: crate::VectorSize,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    PartialMatrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    },
    Matrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: crate::Bytes,
    },
    PartialArray,
    Array {
        base: Handle<Type<'a>>,
        size: ArraySize<'a>,
    },
    Type(Handle<crate::Type>),
}

#[derive(Debug, Copy, Clone)]
pub enum Literal {
    Bool(bool),
    Number(Number),
}

#[derive(Debug)]
pub enum Expression<'a> {
    Literal(Literal),
    Ident(IdentExpr<'a>),
    Construct {
        ty: ConstructorType<'a>,
        ty_span: Span,
        components: Vec<Handle<Expression<'a>>>,
    },
    Unary {
        op: crate::UnaryOperator,
        expr: Handle<Expression<'a>>,
    },
    AddrOf(Handle<Expression<'a>>),
    Deref(Handle<Expression<'a>>),
    Binary {
        op: crate::BinaryOperator,
        left: Handle<Expression<'a>>,
        right: Handle<Expression<'a>>,
    },
    Call {
        function: Ident<'a>,
        arguments: Vec<Handle<Expression<'a>>>,
    },
    Index {
        base: Handle<Expression<'a>>,
        index: Handle<Expression<'a>>,
    },
    Member {
        base: Handle<Expression<'a>>,
        field: Ident<'a>,
    },
    Bitcast {
        expr: Handle<Expression<'a>>,
        to: Handle<Type<'a>>,
        ty_span: Span,
    },
}

#[derive(Debug)]
pub struct LocalVariable<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Handle<Type<'a>>>,
    pub init: Option<Handle<Expression<'a>>>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub struct Let<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Handle<Type<'a>>>,
    pub init: Handle<Expression<'a>>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub enum LocalDecl<'a> {
    Var(LocalVariable<'a>),
    Let(Let<'a>),
}

#[derive(Debug)]
pub struct Local;
