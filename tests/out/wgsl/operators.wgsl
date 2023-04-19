const v_f32_one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
const v_f32_zero: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
const v_f32_half: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5);
const v_i32_one: vec4<i32> = vec4<i32>(1, 1, 1, 1);

fn builtins() -> vec4<f32> {
    let s1_ = select(0, 1, true);
    let s2_ = select(v_f32_zero, v_f32_one, true);
    let s3_ = select(v_f32_one, v_f32_zero, vec4<bool>(false, false, false, false));
    let m1_ = mix(v_f32_zero, v_f32_one, v_f32_half);
    let m2_ = mix(v_f32_zero, v_f32_one, 0.1);
    let b1_ = bitcast<f32>(1);
    let b2_ = bitcast<vec4<f32>>(v_i32_one);
    let v_i32_zero = vec4<i32>(0, 0, 0, 0);
    return (((((vec4<f32>((vec4(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4(b1_)) + b2_);
}

fn splat() -> vec4<f32> {
    let a_2 = (((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0));
    let b = (vec4(5) % vec4(2));
    return (a_2.xyxy + vec4<f32>(b));
}

fn splat_assignment() -> vec2<f32> {
    var a: vec2<f32>;

    a = vec2(2.0);
    let _e4 = a;
    a = (_e4 + vec2(1.0));
    let _e8 = a;
    a = (_e8 - vec2(3.0));
    let _e12 = a;
    a = (_e12 / vec2(4.0));
    let _e15 = a;
    return _e15;
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

fn logical() {
    let neg1_ = vec2<bool>(false, false);
    let bitwise_or0_ = (true | false);
    let bitwise_or1_ = (vec3(true) | vec3(false));
    let bitwise_and0_ = (true & false);
    let bitwise_and1_ = (vec4(true) & vec4(false));
}

fn arithmetic() {
    let neg1_1 = vec2<i32>(-1, -1);
    let neg2_ = vec2<f32>(-1.0, -1.0);
    let add3_ = (vec2(2) + vec2(1));
    let add4_ = (vec3(2u) + vec3(1u));
    let add5_ = (vec4(2.0) + vec4(1.0));
    let sub3_ = (vec2(2) - vec2(1));
    let sub4_ = (vec3(2u) - vec3(1u));
    let sub5_ = (vec4(2.0) - vec4(1.0));
    let mul3_ = (vec2(2) * vec2(1));
    let mul4_ = (vec3(2u) * vec3(1u));
    let mul5_ = (vec4(2.0) * vec4(1.0));
    let div3_ = (vec2(2) / vec2(1));
    let div4_ = (vec3(2u) / vec3(1u));
    let div5_ = (vec4(2.0) / vec4(1.0));
    let rem3_ = (vec2(2) % vec2(1));
    let rem4_ = (vec3(2u) % vec3(1u));
    let rem5_ = (vec4(2.0) % vec4(1.0));
    {
        let add0_ = (vec2(2) + vec2(1));
        let add1_ = (vec2(2) + vec2(1));
        let add2_ = (vec2(2u) + vec2(1u));
        let add3_1 = (vec2(2u) + vec2(1u));
        let add4_1 = (vec2(2.0) + vec2(1.0));
        let add5_1 = (vec2(2.0) + vec2(1.0));
        let sub0_ = (vec2(2) - vec2(1));
        let sub1_ = (vec2(2) - vec2(1));
        let sub2_ = (vec2(2u) - vec2(1u));
        let sub3_1 = (vec2(2u) - vec2(1u));
        let sub4_1 = (vec2(2.0) - vec2(1.0));
        let sub5_1 = (vec2(2.0) - vec2(1.0));
        let mul0_ = vec2<i32>(2, 2);
        let mul1_ = vec2<i32>(2, 2);
        let mul2_ = vec2<u32>(2u, 2u);
        let mul3_1 = vec2<u32>(2u, 2u);
        let mul4_1 = vec2<f32>(2.0, 2.0);
        let mul5_1 = vec2<f32>(2.0, 2.0);
        let div0_ = (vec2(2) / vec2(1));
        let div1_ = (vec2(2) / vec2(1));
        let div2_ = (vec2(2u) / vec2(1u));
        let div3_1 = (vec2(2u) / vec2(1u));
        let div4_1 = (vec2(2.0) / vec2(1.0));
        let div5_1 = (vec2(2.0) / vec2(1.0));
        let rem0_ = (vec2(2) % vec2(1));
        let rem1_ = (vec2(2) % vec2(1));
        let rem2_ = (vec2(2u) % vec2(1u));
        let rem3_1 = (vec2(2u) % vec2(1u));
        let rem4_1 = (vec2(2.0) % vec2(1.0));
        let rem5_1 = (vec2(2.0) % vec2(1.0));
    }
    let _e340 = vec3<f32>(0.0, 0.0, 0.0);
    let _e343 = vec3<f32>(0.0, 0.0, 0.0);
    let add = (mat3x3<f32>() + mat3x3<f32>());
    let _e349 = vec3<f32>(0.0, 0.0, 0.0);
    let _e352 = vec3<f32>(0.0, 0.0, 0.0);
    let sub = (mat3x3<f32>() - mat3x3<f32>());
    let _e356 = vec3<f32>(0.0, 0.0, 0.0);
    let mul_scalar0_ = mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
    let _e372 = vec3<f32>(0.0, 0.0, 0.0);
    let mul_scalar1_ = mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0));
    let _e391 = vec3<f32>(0.0, 0.0, 0.0);
    let mul_vector0_ = (mat4x3<f32>() * vec4(1.0));
    let _e400 = vec3<f32>(0.0, 0.0, 0.0);
    let mul_vector1_ = (vec3(2.0) * mat4x3<f32>());
    let _e406 = vec3<f32>(0.0, 0.0, 0.0);
    let _e409 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let mul = (mat4x3<f32>() * mat3x4<f32>());
}

fn bit() {
    let flip2_ = vec2<i32>(-2, -2);
    let flip3_ = vec3<u32>(4294967294u, 4294967294u, 4294967294u);
    let or2_ = (vec2(2) | vec2(1));
    let or3_ = (vec3(2u) | vec3(1u));
    let and2_ = (vec2(2) & vec2(1));
    let and3_ = (vec3(2u) & vec3(1u));
    let xor2_ = (vec2(2) ^ vec2(1));
    let xor3_ = (vec3(2u) ^ vec3(1u));
    let shl2_ = (vec2(2) << vec2(1u));
    let shl3_ = (vec3(2u) << vec3(1u));
    let shr2_ = (vec2(2) >> vec2(1u));
    let shr3_ = (vec3(2u) >> vec3(1u));
}

fn comparison() {
    let eq3_ = (vec2(2) == vec2(1));
    let eq4_ = (vec3(2u) == vec3(1u));
    let eq5_ = (vec4(2.0) == vec4(1.0));
    let neq3_ = (vec2(2) != vec2(1));
    let neq4_ = (vec3(2u) != vec3(1u));
    let neq5_ = (vec4(2.0) != vec4(1.0));
    let lt3_ = (vec2(2) < vec2(1));
    let lt4_ = (vec3(2u) < vec3(1u));
    let lt5_ = (vec4(2.0) < vec4(1.0));
    let lte3_ = (vec2(2) <= vec2(1));
    let lte4_ = (vec3(2u) <= vec3(1u));
    let lte5_ = (vec4(2.0) <= vec4(1.0));
    let gt3_ = (vec2(2) > vec2(1));
    let gt4_ = (vec3(2u) > vec3(1u));
    let gt5_ = (vec4(2.0) > vec4(1.0));
    let gte3_ = (vec2(2) >= vec2(1));
    let gte4_ = (vec3(2u) >= vec3(1u));
    let gte5_ = (vec4(2.0) >= vec4(1.0));
}

fn assignment() {
    var a_1: i32;
    var vec0_: vec3<i32>;

    a_1 = 1;
    let _e3 = a_1;
    a_1 = (_e3 + 1);
    let _e6 = a_1;
    a_1 = (_e6 - 1);
    let _e8 = a_1;
    let _e9 = a_1;
    a_1 = (_e9 * _e8);
    let _e11 = a_1;
    let _e12 = a_1;
    a_1 = (_e12 / _e11);
    let _e15 = a_1;
    a_1 = (_e15 % 1);
    let _e18 = a_1;
    a_1 = (_e18 & 0);
    let _e21 = a_1;
    a_1 = (_e21 | 0);
    let _e24 = a_1;
    a_1 = (_e24 ^ 0);
    let _e27 = a_1;
    a_1 = (_e27 << 2u);
    let _e30 = a_1;
    a_1 = (_e30 >> 1u);
    let _e33 = a_1;
    a_1 = (_e33 + 1);
    let _e36 = a_1;
    a_1 = (_e36 - 1);
    vec0_ = vec3<i32>();
    let _e42 = vec0_.y;
    vec0_.y = (_e42 + 1);
    let _e46 = vec0_.y;
    vec0_.y = (_e46 - 1);
    return;
}

fn negation_avoids_prefix_decrement() {
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = builtins();
    let _e1 = splat();
    let _e8 = bool_cast(vec3<f32>(1.0, 1.0, 1.0));
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
