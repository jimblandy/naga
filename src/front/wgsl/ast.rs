use crate::front::wgsl::number::Number;
use crate::front::wgsl::Span;
use crate::{Arena, FastHashSet, Handle};
use std::hash::Hash;

#[derive(Debug, Default)]
pub struct TranslationUnit<'a> {
    pub decls: Arena<GlobalDecl<'a>>,
    pub global_expressions: Arena<Expression<'a>>,
}

#[derive(PartialEq, Debug, Clone)]
pub struct Ident<'a> {
    pub name: &'a str,
    pub span: Span,
}

#[derive(PartialEq, Debug)]
pub enum IdentExpr<'a> {
    Unresolved(&'a str),
    Local(Handle<Local>),
}

#[derive(Debug)]
pub struct Dependency<'a> {
    pub ident: &'a str,
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

#[derive(Debug)]
pub struct GlobalDecl<'a> {
    pub kind: GlobalDeclKind<'a>,
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
    pub ty: Type<'a>,
    pub binding: Option<crate::Binding>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub struct FunctionResult<'a> {
    pub ty: Type<'a>,
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
    pub ty: Type<'a>,
    pub init: Option<Handle<Expression<'a>>>,
}

#[derive(Debug)]
pub struct StructMember<'a> {
    pub name: Ident<'a>,
    pub ty: Type<'a>,
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
    pub ty: Type<'a>,
}

#[derive(Debug)]
pub struct Const<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Type<'a>>,
    pub init: Handle<Expression<'a>>,
}

#[derive(Debug)]
pub struct Type<'a> {
    pub kind: TypeKind<'a>,
    pub span: Span,
}

#[derive(Debug, Copy, Clone)]
pub enum ArraySize<'a> {
    Constant(Handle<Expression<'a>>),
    Dynamic,
}

#[derive(Debug)]
pub enum TypeKind<'a> {
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
        base: Box<Type<'a>>,
        space: crate::AddressSpace,
    },
    Array {
        base: Box<Type<'a>>,
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
        base: Box<Type<'a>>,
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
    VarDecl(Box<VarDecl<'a>>),
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
        base: Type<'a>,
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
        to: Type<'a>,
    },
}

#[derive(Debug)]
pub struct LocalVariable<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Type<'a>>,
    pub init: Option<Handle<Expression<'a>>>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub struct Let<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Type<'a>>,
    pub init: Handle<Expression<'a>>,
    pub handle: Handle<Local>,
}

#[derive(Debug)]
pub enum VarDecl<'a> {
    Var(LocalVariable<'a>),
    Const(Let<'a>),
    Let(Let<'a>),
}

#[derive(Debug)]
pub struct Local;