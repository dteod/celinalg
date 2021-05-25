#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <linalg/vector.hpp>
#include <traits.hpp>

using namespace linalg;

#define TYPE_PARAMETER_LIST \
    uint8_t, uint16_t, uint32_t, uint64_t, __uint128_t, int8_t, int16_t, int32_t, int64_t, __int128_t, float, double,          \
    std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>, std::complex<uint64_t>,  \
    std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>, std::complex<int64_t>,    \
    std::complex<float>, std::complex<double>

// Notice that these tests won't work with std::complex<__int128_t> and std::complex<__uint128_t>.
// The reason is that there is no "int256_t" that we can bitcast to a complex of 128-bit type.
// Where are you gonna use it, anyways?


TEMPLATE_TEST_CASE("vector instantiation", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    if constexpr(std::same_as<TestType, int>) {
        Vector<int, 3> v {1, 2, 3};
    } else if constexpr(std::same_as<TestType, double>) {
        Vector<double, 3> v {1., 2., 3.};
    } else if constexpr(std::same_as<TestType, std::complex<int>>) {
        using cint = std::complex<int>;
        Vector<std::complex<int>, 3> v { cint{1, 0}, cint{2, 0}, cint{3, 0}};
    }
}

TEMPLATE_TEST_CASE("ctad-assisted vector instantiation", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    if constexpr(std::same_as<TestType, int>) {
        Vector v {1, 2, 3};
        static_assert(std::same_as<decltype(v), Vector<int, 3>>);
    } else if constexpr(std::same_as<TestType, double>) {
        Vector v {1., 2., 3.};
        static_assert(std::same_as<decltype(v), Vector<double, 3>>);
    } else if constexpr(std::same_as<TestType, std::complex<int>>) {
        Vector v { std::complex<int>{1, 0}, std::complex<int>{2, 0}, std::complex<int>{3, 0}};
        static_assert(std::same_as<decltype(v), Vector<std::complex<int>, 3>>);
    }
}

TEMPLATE_TEST_CASE("vector concatenation", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto cat_vector = concat(v, v);
    REQUIRE(cat_vector[0] == v[0]);
    REQUIRE(cat_vector[1] == v[1]);
    REQUIRE(cat_vector[2] == v[2]);
    REQUIRE(cat_vector[3] == v[0]);
    REQUIRE(cat_vector[4] == v[1]);
    REQUIRE(cat_vector[5] == v[2]);

    REQUIRE(&cat_vector[0] == &v[0]);
    REQUIRE(&cat_vector[1] == &v[1]);
    REQUIRE(&cat_vector[2] == &v[2]);
    REQUIRE(&cat_vector[3] == &v[0]);
    REQUIRE(&cat_vector[4] == &v[1]);
    REQUIRE(&cat_vector[5] == &v[2]);

    static_assert(decltype(cat_vector)::static_size == 2*Vector<TestType, 3>::static_size);
}

TEMPLATE_TEST_CASE("subvector instantiation", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto s = v.subvector(1);
    REQUIRE(s[0] == v[1]);
    REQUIRE(s[1] == v[2]);
}

TEMPLATE_TEST_CASE("subvector mismatched size", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    DynamicVector<TestType> dv {TestType(1), TestType(2), TestType(3), TestType(1), TestType(2), TestType(3)};
    // REQUIRE_THROWS_AS(dv.subvector(1, 8), std::out_of_range); // this should pass but is marked as failed on VSCode for some reasons
}

TEMPLATE_TEST_CASE("subvector extension", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    DynamicVector<TestType> dv {TestType(1), TestType(2), TestType(3), TestType(1), TestType(2), TestType(3)};
    auto dv_s = dv.subvector(1, 3);
    REQUIRE(dv_s.size() == 2);
    dv_s.resize(4, TestType(0));
    REQUIRE(dv_s.size() == 4);
    REQUIRE(dv.size() == 8);

    REQUIRE(dv[0] == TestType(1));
    REQUIRE(dv[1] == TestType(2));
    REQUIRE(dv[2] == TestType(3));
    REQUIRE(dv[3] == TestType(0));
    REQUIRE(dv[4] == TestType(0));
    REQUIRE(dv[5] == TestType(1));
    REQUIRE(dv[6] == TestType(2));
    REQUIRE(dv[7] == TestType(3));
}

TEMPLATE_TEST_CASE("vector element-wise sum", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};

    auto vv = v+v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == v[i] + v[i]);
    }
}

TEMPLATE_TEST_CASE("vector element-wise difference", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v-v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == v[i] - v[i]);
    }
}

TEMPLATE_TEST_CASE("vector element-wise product", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v*v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == v[i] * v[i]);
    }
}

TEMPLATE_TEST_CASE("vector element-wise division", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v/v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == v[i] / v[i]);
    }
}

TEMPLATE_TEST_CASE("vector element-wise modulo", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v%v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == detail::expression_operator<detail::Operation::MODULO>::call(v[i], v[i]));
    }
}

TEMPLATE_TEST_CASE("vector element-wise and", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector v1 {TestType(1), TestType(2), TestType(3)};
    Vector v2 {TestType(0), TestType(0), TestType(0)};
    auto vv = v1 && v2;

    for(size_t i = 0; i != v1.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vv[i] == (std::norm(v1[i]) && std::norm(v2[i])));
        } else {
            REQUIRE(vv[i] == (static_cast<bool>(v1[i]) && static_cast<bool>(v2[i])));
        }
    }
}

TEMPLATE_TEST_CASE("vector element-wise or", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector v1 {TestType(1), TestType(2), TestType(3)};
    Vector v2 {TestType(0), TestType(0), TestType(0)};
    auto vv = v1 || v2;

    for(size_t i = 0; i != v1.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vv[i] == (std::norm(v1[i]) || std::norm(v2[i])));
        } else {
            REQUIRE(vv[i] == (static_cast<bool>(v1[i]) || static_cast<bool>(v2[i])));
        }
    }
}

TEMPLATE_TEST_CASE("vector element-wise bitand", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v&v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i]) &
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i])
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector element-wise bitor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v|v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i]) |
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i])
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector element-wise bitxor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v {TestType(1), TestType(2), TestType(3)};
    auto vv = v^v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vv[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i]) ^
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(v[i])
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector-scalar sum", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v + TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == v[i] + TestType(10));
    }
}

TEMPLATE_TEST_CASE("vector-scalar difference", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v - TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == v[i] - TestType(10));
    }
}

TEMPLATE_TEST_CASE("vector-scalar product", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v * TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == v[i] * TestType(10));
    }
}

TEMPLATE_TEST_CASE("vector-scalar division", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v / TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == v[i] / TestType(10));
    }
}

TEMPLATE_TEST_CASE("vector-scalar modulo", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v % TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == detail::expression_operator<detail::Operation::MODULO>::call(v[i], TestType(10)));
    }
}

TEMPLATE_TEST_CASE("vector-scalar and", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v && TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vs[i] == ( std::norm(v[i]) && std::norm(TestType(10)) ));
        } else {
            REQUIRE(vs[i] == ( static_cast<bool>(v[i]) && static_cast<bool>(TestType(10)) ));
        }
    }
}

TEMPLATE_TEST_CASE("vector-scalar or", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v || TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vs[i] == ( std::norm(v[i]) || std::norm(TestType(10)) ));
        } else {
            REQUIRE(vs[i] == ( static_cast<bool>(v[i]) || static_cast<bool>(TestType(10)) ));
        }
    }
}

TEMPLATE_TEST_CASE("vector-scalar bitand", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v & TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) &
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector-scalar bitor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v | TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) |
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector-scalar bitxor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = v ^ TestType(10);

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) ^
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector cross product", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v1{TestType(1), TestType(6), TestType(1)};
    Vector<TestType, 3> v2{TestType(2), TestType(6), TestType(2)};
    auto vp = cprod(v1, v2);

    // this fails with uint8_t and uint16_t because of sign underflows
    if constexpr(!(std::same_as<TestType, uint8_t> || std::same_as<TestType, uint16_t>)) {
        REQUIRE(vp[0] == (v1[1]*v2[2] - v1[2]*v2[1]));
        REQUIRE(vp[1] == (v1[2]*v2[0] - v1[0]*v2[2]));
        REQUIRE(vp[2] == (v1[0]*v2[1] - v1[1]*v2[0]));
    }
}

TEMPLATE_TEST_CASE("vector scalar product", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v1{TestType(1), TestType(2), TestType(3)};
    Vector<TestType, 3> v2{TestType(2), TestType(6), TestType(1)};
    auto vs = sprod(v1, v2);

    REQUIRE(vs.get() == (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]));
}

namespace {
    template<typename TestType>
    constexpr auto vector_scalar_product_passed() {
        // With a modified version of libstdc++ (marking every call of std::vector with constexpr)
        // this function works also with doubles (since 3 doubles are 24 bytes, the Vector will switch to a std::vector that allocates the memory on the heap)
        // std::vector allocation and destruction and std::allocate are core constant expressions, and may be used in a constant expression.
        // For now, even with the libstdc++ hack this cannot be handled though:
        //     constexpr std::vector<int> v{1,2,3};
        // because even if operator new is a core constant expression it is not a constant expression per-se.
        // Read this https://www.quora.com/How-do-I-use-the-C-20-constexpr-std-vector?share=1
        Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
        return std::accumulate(v.begin(), v.end(), TestType(0));
    }

    template<auto Value>
    class Getter {
    public:
        inline constexpr static auto value = Value;
    };

} 

#if LINALG_USE_CONSTEXPR_VECTOR_COMPILER
// automatically set by CMake if using the hacked libstdc++
TEMPLATE_TEST_CASE("constexpr Vector for non-complex types", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    if constexpr(req::complex<TestType>) {
        // std::complex won't work, it cannot be used as a template non-type parameter since "it is not structural" 
        // even if it has all the characteristics to be used as a structural one.
        // Evaluating it at runtime is ok though
        Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
        [[maybe_unused]] auto b = std::accumulate(v.begin(), v.end(), TestType(0));
    } else {
        [[maybe_unused]] constexpr auto b = Getter<vector_scalar_product_passed<TestType>()>::value;
    }
    REQUIRE( true );
}
#else
// 
TEMPLATE_TEST_CASE("constexpr Vector for stack-allocated objects", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    if constexpr(req::complex<TestType> || std::same_as<TestType, double> /*allocating 3 doubles requires 24 bytes that go on the heap*/ ) {
        Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
        [[maybe_unused]] auto b = std::accumulate(v.begin(), v.end(), TestType(0));
        std::cout << b << std::endl;
    } else {
        [[maybe_unused]] constexpr auto b = Getter<vector_scalar_product_passed<TestType>()>::value;
        std::cout << b << std::endl;
    }
    REQUIRE( true );
}
#endif


TEMPLATE_TEST_CASE("scalar-vector sum", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) + v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == TestType(10) + v[i] );
    }
}

TEMPLATE_TEST_CASE("scalar-vector difference", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) - v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == TestType(10) - v[i] );
    }
}

TEMPLATE_TEST_CASE("scalar-vector product", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) * v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == TestType(10) * v[i] );
    }
}

TEMPLATE_TEST_CASE("scalar-vector division", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) / v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == TestType(10) / v[i] );
    }
}

TEMPLATE_TEST_CASE("scalar-vector modulo", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) % v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == detail::expression_operator<detail::Operation::MODULO>::call(TestType(10), v[i]));
    }
}

TEMPLATE_TEST_CASE("scalar-vector and", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) && v;

    for(size_t i = 0; i != v.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vs[i] == ( std::norm(v[i]) && std::norm(TestType(10)) ));
        } else {
            REQUIRE(vs[i] == ( static_cast<bool>(v[i]) && static_cast<bool>(TestType(10)) ));
        }
    }
}

TEMPLATE_TEST_CASE("scalar-vector or", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) || v;

    for(size_t i = 0; i != v.size(); ++i) {
        if constexpr(req::complex<TestType>) {
            REQUIRE(vs[i] == ( std::norm(v[i]) || std::norm(TestType(10)) ));
        } else {
            REQUIRE(vs[i] == ( static_cast<bool>(v[i]) || static_cast<bool>(TestType(10)) ));
        }
    }
}

TEMPLATE_TEST_CASE("scalar-vector bitand", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) & v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) &
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("scalar-vector bitor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) | v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) |
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("scalar-vector bitxor", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 3> v{TestType(1), TestType(2), TestType(3)};
    auto vs = TestType(10) ^ v;

    for(size_t i = 0; i != v.size(); ++i) {
        REQUIRE(vs[i] == (
            std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<decltype(v[i])>)>>(v[i]) ^
            std::bit_cast<traits::unsigned_of_size_t<sizeof(TestType)>>(TestType(10))
            )
        );
    }
}

TEMPLATE_TEST_CASE("vector zero-initialized by default", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 10> v;
    REQUIRE(std::find_if(v.begin(), v.end(), [](auto& x){ return x != TestType(0); }) == v.end());
}

TEMPLATE_TEST_CASE("vector generation", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 10> v;
    if constexpr(req::complex<TestType>) {
        // no operator++ for std::complex
        TestType tt(0);
        std::generate(v.begin(), v.end(), [&](){ return tt += 1; });
    } else {
        std::iota(v.begin(), v.end(), TestType(1));
    }
    for(TestType i {0}; auto& x : v) {
        REQUIRE(x == (i+=1));        
    }
}

TEMPLATE_TEST_CASE("iterator has direct access", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 10> v;
    for(size_t i = 0; auto& x : v) {
        REQUIRE(std::addressof(x) == std::addressof(v[i++]));
    }
}

TEMPLATE_TEST_CASE("vector accumulate", "[linalg][vector]", TYPE_PARAMETER_LIST) {
    Vector<TestType, 10> v;
    if constexpr(req::complex<TestType>) {
        // no operator++ for std::complex
        TestType tt(0);
        std::generate(v.begin(), v.end(), [&](){ return tt += 1; });
    } else {
        std::iota(v.begin(), v.end(), TestType(1));
    }
    REQUIRE(std::accumulate(v.begin(), v.end(), TestType(0)) == TestType(55));
}

TEST_CASE("std mathematical functions", "[linalg][vector]") {
    Vector<int, 10> v;
    std::iota(v.begin(), v.end(), -5);

    auto abs_v = abs(v);
    REQUIRE(abs_v[0] == 5);
    REQUIRE(abs_v[1] == 4);
    REQUIRE(abs_v[2] == 3);
    REQUIRE(abs_v[3] == 2);
    REQUIRE(abs_v[4] == 1);
    REQUIRE(abs_v[5] == 0);
    REQUIRE(abs_v[6] == 1);
    REQUIRE(abs_v[7] == 2);
    REQUIRE(abs_v[8] == 3);
    REQUIRE(abs_v[9] == 4);
}