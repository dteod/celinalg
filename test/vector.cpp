#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <linalg/vector.hpp>
#include <traits.hpp>

using namespace linalg;

#define TYPE_PARAMETER_LIST \
    uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,        \
    std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>, std::complex<uint64_t>,  \
    std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>, std::complex<int64_t>,      \
    std::complex<float>, std::complex<double>

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
        auto b = std::accumulate(v.begin(), v.end(), TestType(0));
        REQUIRE(b == TestType(1+2+3));
    } else {
        [[maybe_unused]] constexpr auto b = Getter<vector_scalar_product_passed<TestType>()>::value;
        REQUIRE(b == TestType(1+2+3));
    }
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

TEST_CASE("vector math functions - abs/fabs", "[linalg][vector]") {
    Vector v{1, 2, 5, -6, -1, 0, 0, 4, -4, 8};
    {
        auto fcn = abs(v);
        REQUIRE(fcn[0] == 1);
        REQUIRE(fcn[1] == 2);
        REQUIRE(fcn[2] == 5);
        REQUIRE(fcn[3] == 6);
        REQUIRE(fcn[4] == 1);
        REQUIRE(fcn[5] == 0);
        REQUIRE(fcn[6] == 0);
        REQUIRE(fcn[7] == 4);
        REQUIRE(fcn[8] == 4);
        REQUIRE(fcn[9] == 8);
    }
    {
        auto fcn = fabs(v);
        REQUIRE(fcn[0] == 1);
        REQUIRE(fcn[1] == 2);
        REQUIRE(fcn[2] == 5);
        REQUIRE(fcn[3] == 6);
        REQUIRE(fcn[4] == 1);
        REQUIRE(fcn[5] == 0);
        REQUIRE(fcn[6] == 0);
        REQUIRE(fcn[7] == 4);
        REQUIRE(fcn[8] == 4);
        REQUIRE(fcn[9] == 8);
    }
    
}

TEST_CASE("vector math functions - fmod", "[linalg][vector]") {
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.};

    auto fcn = fmod(v1, v2);
    REQUIRE(fcn[0] == fmod(v1[0], v2[0]));
    REQUIRE(fcn[1] == fmod(v1[1], v2[1]));
    REQUIRE(fcn[2] == fmod(v1[2], v2[2]));
    REQUIRE(fcn[3] == fmod(v1[3], v2[3]));
    REQUIRE(fcn[4] == fmod(v1[4], v2[4]));
}

TEST_CASE("vector math functions - remainder", "[linalg][vector]") {
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.};

    auto fcn = remainder(v1, v2);
    REQUIRE(fcn[0] == remainder(v1[0], v2[0]));
    REQUIRE(fcn[1] == remainder(v1[1], v2[1]));
    REQUIRE(fcn[2] == remainder(v1[2], v2[2]));
    REQUIRE(fcn[3] == remainder(v1[3], v2[3]));
    REQUIRE(fcn[4] == remainder(v1[4], v2[4]));
}

TEST_CASE("vector math functions - remquo", "[linalg][vector]") {
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.};
    std::array<int, 5> arr{};
    std::array<int*, 5> pArr;
    std::transform(arr.begin(), arr.end(), pArr.begin(), [](int& v) { return &v; }); 

    auto fcn = remquo(v1, v2, pArr);

    int buffer = 13094948;
    auto check = [&](size_t i) {
        REQUIRE(fcn[i] == remquo(v1[i], v2[i], &buffer));
        REQUIRE(arr[i] == buffer); 
    };
    for(size_t i = 0; i != pArr.size(); ++i) {
        check(i);
    }
}


TEST_CASE("vector math functions - fma", "[linalg][vector]") {
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.}, v3{10., 7., -0.5, 33.6, std::numbers::pi};
    auto out = fma(v1, v2, v3);
    for(size_t i = 0; i != v1.size(); ++i) {
        REQUIRE(out[i] == v1[i]*v2[i] + v3[i]); 
    }
}

#define DEFINE_FUNCTION_CALL_TEST_1(FUNCTION)                           \
TEST_CASE("vector math functions - " #FUNCTION, "[linalg][vector]") {   \
    Vector v{1., 2., 5., -6., -1.};                  \
    auto out = FUNCTION(v);                                     \
    for(size_t i = 0; i != v.size(); ++i) {                     \
        if(std::isnan(out[i])) {                                \
            REQUIRE(std::isnan(std::FUNCTION(v[i])));           \
        } else {                                                \
            REQUIRE(out[i] == std::FUNCTION(v[i]));             \
        }                                                       \
    }                                                           \
}

#define DEFINE_FUNCTION_CALL_TEST_2(FUNCTION)                                \
TEST_CASE("vector math functions - " #FUNCTION, "[linalg][vector]") {               \
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.};   \
    auto out = FUNCTION(v1, v2);                                             \
    for(size_t i = 0; i != v1.size(); ++i) {                                 \
        if(std::isnan(out[i])) {                                             \
            REQUIRE(std::isnan(std::FUNCTION(v1[i], v2[i])));                \
        } else {                                                             \
            REQUIRE(out[i] == std::FUNCTION(v1[i], v2[i]));                  \
        }                                                                    \
    }                                                                        \
}

#define DEFINE_FUNCTION_CALL_TEST_3(FUNCTION)                                                                           \
TEST_CASE("vector math functions - " #FUNCTION, "[linalg][vector]") {                                                          \
    Vector v1{1., 2., 5., -6., -1.}, v2{0.1, 0.5, 4., -4., 8.}, v3{10., 7., -0.5, 33.6, std::numbers::pi};   \
    {\
        auto out = FUNCTION(v1, v2, v3);                                                                                    \
        for(size_t i = 0; i != v1.size(); ++i) {                                                                            \
            if(std::isnan(out[i])) {                                                                                        \
                REQUIRE(std::isnan(std::FUNCTION(v1[i], v2[i], v3[i])));                                                    \
            } else {                                                                                                        \
                REQUIRE(out[i] == std::FUNCTION(v1[i], v2[i], v3[i]));                                                      \
            }                                                                                                               \
        }                                                                                                                   \
    }\
    {\
        auto out = FUNCTION(v1, v2, 10.0);                                                                                    \
        for(size_t i = 0; i != v1.size(); ++i) {                                                                            \
            if(std::isnan(out[i])) {                                                                                        \
                REQUIRE(std::isnan(std::FUNCTION(v1[i], v2[i], 10)));                                                    \
            } else {                                                                                                        \
                REQUIRE(out[i] == std::FUNCTION(v1[i], v2[i], 10));                                                      \
            }                                                                                                               \
        }                                                                                                                   \
    }\
}


DEFINE_FUNCTION_CALL_TEST_2(fmax);
DEFINE_FUNCTION_CALL_TEST_2(fmaxf);
DEFINE_FUNCTION_CALL_TEST_2(fmaxl);
DEFINE_FUNCTION_CALL_TEST_2(fmin);
DEFINE_FUNCTION_CALL_TEST_2(fminf);
DEFINE_FUNCTION_CALL_TEST_2(fminl);
DEFINE_FUNCTION_CALL_TEST_2(fdim);
DEFINE_FUNCTION_CALL_TEST_2(fdimf);
DEFINE_FUNCTION_CALL_TEST_2(fdiml);
DEFINE_FUNCTION_CALL_TEST_3(lerp);
DEFINE_FUNCTION_CALL_TEST_1(exp);
DEFINE_FUNCTION_CALL_TEST_1(exp2);
DEFINE_FUNCTION_CALL_TEST_1(exp2f);
DEFINE_FUNCTION_CALL_TEST_1(exp2l);
DEFINE_FUNCTION_CALL_TEST_1(expm1);
DEFINE_FUNCTION_CALL_TEST_1(expm1f);
DEFINE_FUNCTION_CALL_TEST_1(expm1l);
DEFINE_FUNCTION_CALL_TEST_1(log);
DEFINE_FUNCTION_CALL_TEST_1(log10);
DEFINE_FUNCTION_CALL_TEST_1(log1p);
DEFINE_FUNCTION_CALL_TEST_1(log1pf);
DEFINE_FUNCTION_CALL_TEST_1(log1pl);
DEFINE_FUNCTION_CALL_TEST_2(pow);
DEFINE_FUNCTION_CALL_TEST_1(sqrt);
DEFINE_FUNCTION_CALL_TEST_1(cbrt);
DEFINE_FUNCTION_CALL_TEST_1(cbrtf);
DEFINE_FUNCTION_CALL_TEST_1(cbrtl);
DEFINE_FUNCTION_CALL_TEST_2(hypot);
DEFINE_FUNCTION_CALL_TEST_2(hypotf);
DEFINE_FUNCTION_CALL_TEST_2(hypotl);
DEFINE_FUNCTION_CALL_TEST_1(sin);
DEFINE_FUNCTION_CALL_TEST_1(cos);
DEFINE_FUNCTION_CALL_TEST_1(tan);
DEFINE_FUNCTION_CALL_TEST_1(asin);
DEFINE_FUNCTION_CALL_TEST_1(acos);
DEFINE_FUNCTION_CALL_TEST_1(atan);
DEFINE_FUNCTION_CALL_TEST_2(atan2);
DEFINE_FUNCTION_CALL_TEST_1(sinh);
DEFINE_FUNCTION_CALL_TEST_1(cosh);
DEFINE_FUNCTION_CALL_TEST_1(tanh);
DEFINE_FUNCTION_CALL_TEST_1(asinh);
DEFINE_FUNCTION_CALL_TEST_1(asinhf);
DEFINE_FUNCTION_CALL_TEST_1(asinhl);
DEFINE_FUNCTION_CALL_TEST_1(acosh);
DEFINE_FUNCTION_CALL_TEST_1(acoshf);
DEFINE_FUNCTION_CALL_TEST_1(acoshl);
DEFINE_FUNCTION_CALL_TEST_1(atanh);
DEFINE_FUNCTION_CALL_TEST_1(atanhf);
DEFINE_FUNCTION_CALL_TEST_1(atanhl);
DEFINE_FUNCTION_CALL_TEST_1(erf);
DEFINE_FUNCTION_CALL_TEST_1(erff);
DEFINE_FUNCTION_CALL_TEST_1(erfl);
DEFINE_FUNCTION_CALL_TEST_1(erfc);
DEFINE_FUNCTION_CALL_TEST_1(erfcf);
DEFINE_FUNCTION_CALL_TEST_1(erfcl);
DEFINE_FUNCTION_CALL_TEST_1(tgamma);
DEFINE_FUNCTION_CALL_TEST_1(tgammaf);
DEFINE_FUNCTION_CALL_TEST_1(tgammal);
DEFINE_FUNCTION_CALL_TEST_1(lgamma);
DEFINE_FUNCTION_CALL_TEST_1(lgammaf);
DEFINE_FUNCTION_CALL_TEST_1(lgammal);
DEFINE_FUNCTION_CALL_TEST_1(ceil);
DEFINE_FUNCTION_CALL_TEST_1(floor);
DEFINE_FUNCTION_CALL_TEST_1(trunc);
DEFINE_FUNCTION_CALL_TEST_1(truncf);
DEFINE_FUNCTION_CALL_TEST_1(truncl);
DEFINE_FUNCTION_CALL_TEST_1(round);
DEFINE_FUNCTION_CALL_TEST_1(roundf);
DEFINE_FUNCTION_CALL_TEST_1(roundl);
DEFINE_FUNCTION_CALL_TEST_1(lround);
DEFINE_FUNCTION_CALL_TEST_1(lroundf);
DEFINE_FUNCTION_CALL_TEST_1(lroundl);
DEFINE_FUNCTION_CALL_TEST_1(llround);
DEFINE_FUNCTION_CALL_TEST_1(llroundf);
DEFINE_FUNCTION_CALL_TEST_1(llroundl);
DEFINE_FUNCTION_CALL_TEST_1(nearbyint);
DEFINE_FUNCTION_CALL_TEST_1(nearbyintf);
DEFINE_FUNCTION_CALL_TEST_1(nearbyintl);
DEFINE_FUNCTION_CALL_TEST_1(rint);
DEFINE_FUNCTION_CALL_TEST_1(rintf);
DEFINE_FUNCTION_CALL_TEST_1(rintl);
DEFINE_FUNCTION_CALL_TEST_1(lrint);
DEFINE_FUNCTION_CALL_TEST_1(lrintf);
DEFINE_FUNCTION_CALL_TEST_1(lrintl);
DEFINE_FUNCTION_CALL_TEST_1(llrint);
DEFINE_FUNCTION_CALL_TEST_1(llrintf);
DEFINE_FUNCTION_CALL_TEST_1(llrintl);


TEST_CASE("vector math functions - frexp", "[linalg][vector]") {
    Vector v{1., 2., 5., -6., -1.};
    std::array<int, 5> arr{};
    std::array<int*, 5> pArr;
    std::transform(arr.begin(), arr.end(), pArr.begin(), [](int& v) { return &v; }); 

    auto fcn = frexp(v, pArr);

    int buffer = 13094948;
    auto check = [&](size_t i) {
        REQUIRE(fcn[i] == frexp(v[i], &buffer));
        REQUIRE(arr[i] == buffer); 
    };
    for(size_t i = 0; i != pArr.size(); ++i) {
        check(i);
    }
}

TEST_CASE("vector math functions - ldexp", "[linalg][vector]") {
    Vector v{1., 2., 5., -6., -1.};
    std::array<int, 5> arr{1, 2, 3, 4, 5};
    auto fcn = ldexp(v, arr);
    auto check = [&](size_t i) {
        REQUIRE(fcn[i] == ldexp(v[i], arr[i]));
    };
    for(size_t i = 0; i != arr.size(); ++i) {
        check(i);
    }
}

TEST_CASE("vector math functions - modf", "[linalg][vector]") {
    Vector v{1., 2., 5., -6., -1.};
    std::array<double, 5> arr{};
    std::array<double*, 5> pArr;
    std::transform(arr.begin(), arr.end(), pArr.begin(), [](double& v) { return &v; }); 

    auto fcn = modf(v, pArr);

    double buffer = 13094948;
    auto check = [&](size_t i) {
        REQUIRE(fcn[i] == modf(v[i], &buffer));
        REQUIRE(arr[i] == buffer); 
    };
    for(size_t i = 0; i != pArr.size(); ++i) {
        check(i);
    }
}

DEFINE_FUNCTION_CALL_TEST_2(scalbn);
DEFINE_FUNCTION_CALL_TEST_2(scalbnf);
DEFINE_FUNCTION_CALL_TEST_2(scalbnl);
DEFINE_FUNCTION_CALL_TEST_2(scalbln);
DEFINE_FUNCTION_CALL_TEST_2(scalblnf);
DEFINE_FUNCTION_CALL_TEST_2(scalblnl);
DEFINE_FUNCTION_CALL_TEST_1(ilogb);
DEFINE_FUNCTION_CALL_TEST_1(ilogbf);
DEFINE_FUNCTION_CALL_TEST_1(ilogbl);
DEFINE_FUNCTION_CALL_TEST_1(logb);
DEFINE_FUNCTION_CALL_TEST_1(logbf);
DEFINE_FUNCTION_CALL_TEST_1(logbl);
DEFINE_FUNCTION_CALL_TEST_2(nextafter);
DEFINE_FUNCTION_CALL_TEST_2(nextafterf);
DEFINE_FUNCTION_CALL_TEST_2(nextafterl);
DEFINE_FUNCTION_CALL_TEST_2(nexttoward);
DEFINE_FUNCTION_CALL_TEST_2(nexttowardf);
DEFINE_FUNCTION_CALL_TEST_2(nexttowardl);
DEFINE_FUNCTION_CALL_TEST_2(copysign);
DEFINE_FUNCTION_CALL_TEST_2(copysignf);
DEFINE_FUNCTION_CALL_TEST_2(copysignl);
DEFINE_FUNCTION_CALL_TEST_1(fpclassify);
DEFINE_FUNCTION_CALL_TEST_1(isfinite);
DEFINE_FUNCTION_CALL_TEST_1(isinf);
DEFINE_FUNCTION_CALL_TEST_1(isnan);
DEFINE_FUNCTION_CALL_TEST_1(isnormal);
DEFINE_FUNCTION_CALL_TEST_1(signbit);
DEFINE_FUNCTION_CALL_TEST_2(isgreater);
DEFINE_FUNCTION_CALL_TEST_2(isgreaterequal);
DEFINE_FUNCTION_CALL_TEST_2(isless);
DEFINE_FUNCTION_CALL_TEST_2(islessequal);
DEFINE_FUNCTION_CALL_TEST_2(islessgreater);
DEFINE_FUNCTION_CALL_TEST_2(isunordered);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_laguerre);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_laguerref);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_laguerrel);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_legendre);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_legendref);
// DEFINE_FUNCTION_CALL_TEST_3(assoc_legendrel);
// DEFINE_FUNCTION_CALL_TEST_2(beta);
// DEFINE_FUNCTION_CALL_TEST_2(betaf);
// DEFINE_FUNCTION_CALL_TEST_2(betal);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_1);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_1f);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_1l);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_2);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_2f);
// DEFINE_FUNCTION_CALL_TEST_1(comp_ellint_2l);
// DEFINE_FUNCTION_CALL_TEST_2(comp_ellint_3);
// DEFINE_FUNCTION_CALL_TEST_2(comp_ellint_3f);
// DEFINE_FUNCTION_CALL_TEST_2(comp_ellint_3l);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_i);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_if);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_il);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_j);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_jf);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_jl);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_k);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_kf);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_bessel_kl);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_neumann);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_neumannf);
// DEFINE_FUNCTION_CALL_TEST_2(cyl_neumannl);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_1);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_1f);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_1l);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_2);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_2f);
// DEFINE_FUNCTION_CALL_TEST_2(ellint_2l);
// DEFINE_FUNCTION_CALL_TEST_3(ellint_3);
// DEFINE_FUNCTION_CALL_TEST_3(ellint_3f);
// DEFINE_FUNCTION_CALL_TEST_3(ellint_3l);
// DEFINE_FUNCTION_CALL_TEST_1(expint);
// DEFINE_FUNCTION_CALL_TEST_1(expintf);
// DEFINE_FUNCTION_CALL_TEST_1(expintl);
// DEFINE_FUNCTION_CALL_TEST_2(hermite);
// DEFINE_FUNCTION_CALL_TEST_2(hermitef);
// DEFINE_FUNCTION_CALL_TEST_2(hermitel);
// DEFINE_FUNCTION_CALL_TEST_2(legendre);
// DEFINE_FUNCTION_CALL_TEST_2(legendref);
// DEFINE_FUNCTION_CALL_TEST_2(legendrel);
// DEFINE_FUNCTION_CALL_TEST_2(laguerre);
// DEFINE_FUNCTION_CALL_TEST_2(laguerref);
// DEFINE_FUNCTION_CALL_TEST_2(laguerrel);
// DEFINE_FUNCTION_CALL_TEST_1(riemann_zeta);
// DEFINE_FUNCTION_CALL_TEST_1(riemann_zetaf);
// DEFINE_FUNCTION_CALL_TEST_1(riemann_zetal);
// DEFINE_FUNCTION_CALL_TEST_2(sph_bessel);
// DEFINE_FUNCTION_CALL_TEST_2(sph_besself);
// DEFINE_FUNCTION_CALL_TEST_2(sph_bessell);
// DEFINE_FUNCTION_CALL_TEST_3(sph_legendre);
// DEFINE_FUNCTION_CALL_TEST_3(sph_legendref);
// DEFINE_FUNCTION_CALL_TEST_3(sph_legendrel);
// DEFINE_FUNCTION_CALL_TEST_2(sph_neumann);
// DEFINE_FUNCTION_CALL_TEST_2(sph_neumannf);
// DEFINE_FUNCTION_CALL_TEST_2(sph_neumannl);