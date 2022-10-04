use crate::front::wgsl::number::Number;
use crate::front::wgsl::Span;
use crate::{Arena, Handle};

#[derive(Debug, Default)]
pub struct TranslationUnit<'a> {
    pub decls: Vec<GlobalDecl<'a>>,
    pub global_expressions: Arena<Expression<'a>>,
}

#[derive(PartialEq, Debug)]
pub struct Ident<'a> {
    pub name: &'a str,
    pub span: Span,
}

#[derive(Debug)]
pub struct GlobalDecl<'a> {
    pub kind: GlobalDeclKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum GlobalDeclKind<'a> {
    Fn(Function<'a>),
    Var(GlobalVariable<'a>),
    Let(Let<'a>),
    Const(Let<'a>),
    Struct(Struct<'a>),
    Type(TypeAlias<'a>),
}

#[derive(Debug)]
pub struct FunctionArgument<'a> {
    pub name: Ident<'a>,
    pub ty: Type<'a>,
    pub binding: Option<crate::Binding>,
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
    pub body: Block<'a>,
}

#[derive(Debug)]
pub struct GlobalVariable<'a> {
    pub name: Ident<'a>,
    pub space: crate::AddressSpace,
    pub binding: Option<crate::ResourceBinding>,
    pub ty: Option<Type<'a>>,
    pub init: Option<Expression<'a>>,
}

#[derive(Debug)]
pub struct StructMember<'a> {
    pub name: Ident<'a>,
    pub ty: Type<'a>,
    pub binding: Option<crate::Binding>,
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
pub struct Type<'a> {
    pub kind: TypeKind<'a>,
    pub span: Span,
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
        size: crate::ArraySize,
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
        size: crate::ArraySize,
    },
    User(Ident<'a>),
}

#[derive(Debug)]
pub struct Block<'a> {
    pub stmts: Vec<Stmt<'a>>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Stmt<'a> {
    pub kind: StmtKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum AssignTarget<'a> {
    Phony,
    Variable(Ident<'a>),
}

#[derive(Debug)]
pub enum StmtKind<'a> {
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
        target: AssignTarget<'a>,
        value: Handle<Expression<'a>>,
    },
}

#[derive(Debug)]
pub struct SwitchCase<'a> {
    pub value: SwitchValue<'a>,
    pub body: Block<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum SwitchValue<'a> {
    Expr(Handle<Expression<'a>>),
    Default,
}

#[derive(Debug)]
pub struct Expression<'a> {
    pub kind: ExprKind<'a>,
    pub span: Span,
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
        size: crate::ArraySize,
        stride: u32,
    },
}

#[derive(Debug)]
pub enum Literal {
    Bool(bool),
    Number(Number),
}

#[derive(Debug)]
pub enum ExprKind<'a> {
    Literal(Literal),
    Ident(Ident<'a>),
    Construct {
        ty: ConstructorType<'a>,
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
        result: Option<Handle<Expression<'a>>>,
    },
    Index {
        base: Handle<Expression<'a>>,
        index: Handle<Expression<'a>>,
    },
    Member {
        base: Handle<Expression<'a>>,
        field: Ident<'a>,
    },
    Bitcast {},
}

#[derive(Debug)]
pub struct LocalVariable<'a> {
    pub name: Ident<'a>,
    pub ty: Type<'a>,
    pub init: Option<Handle<Expression<'a>>>,
}

#[derive(Debug)]
pub enum VarDecl<'a> {
    Var(LocalVariable<'a>),
    Const(Let<'a>),
    Let(Let<'a>),
}

#[derive(Debug)]
pub struct Let<'a> {
    pub name: Ident<'a>,
    pub ty: Option<Type<'a>>,
    pub init: Handle<Expression<'a>>,
}
