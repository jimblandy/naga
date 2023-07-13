
struct Foo {
    float4 a;
    int b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

Foo ConstructFoo(float4 arg0, int arg1) {
    Foo ret = (Foo)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

typedef int ret_Constructarray4_int_[4];
ret_Constructarray4_int_ Constructarray4_int_(int arg0, int arg1, int arg2, int arg3) {
    int ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    Foo foo = (Foo)0;

    foo = ConstructFoo((1.0).xxxx, 1);
    float2x2 m0_ = float2x2(float2(1.0, 0.0), float2(0.0, 1.0));
    float4x4 m1_ = float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    uint2 cit0_ = (0u).xx;
    float2x2 cit1_ = float2x2((0.0).xx, (0.0).xx);
    int cit2_[4] = Constructarray4_int_(0, 1, 2, 3);
    bool ic0_ = bool((bool)0);
    int ic1_ = int((int)0);
    uint ic2_ = uint((uint)0);
    float ic3_ = float((float)0);
    uint2 ic4_ = uint2((uint2)0);
    float2x3 ic5_ = float2x3((float2x3)0);
    uint2 ic6_ = asuint((uint2)0);
    float2x3 ic7_ = asfloat((float2x3)0);
}
