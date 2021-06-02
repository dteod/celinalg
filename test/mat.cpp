#include <iostream>
#include <ranges>

#include "linalg/matrix.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace linalg;

#define TYPE_PARAMETER_LIST \
    uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double,        \
    std::complex<uint8_t>, std::complex<uint16_t>, std::complex<uint32_t>, std::complex<uint64_t>,  \
    std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>, std::complex<int64_t>,      \
    std::complex<float>, std::complex<double>

TEMPLATE_TEST_CASE("matrix instantiation", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix<TestType, 3, 3> m {{
        { TestType(1), TestType(2), TestType(3)},
        { TestType(5), TestType(0), TestType(2)},
        { TestType(7), TestType(1), TestType(1)},
    }};

    REQUIRE( m[0][0] == TestType(1) );
    REQUIRE( m[0][1] == TestType(2) );
    REQUIRE( m[0][2] == TestType(3) );
    REQUIRE( m[1][0] == TestType(5) );
    REQUIRE( m[1][1] == TestType(0) );
    REQUIRE( m[1][2] == TestType(2) );
    REQUIRE( m[2][0] == TestType(7) );
    REQUIRE( m[2][1] == TestType(1) );
    REQUIRE( m[2][2] == TestType(1) );
}

TEMPLATE_TEST_CASE("CTAD-assisted matrix instantiation", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix m {{
        { TestType(1), TestType(2), TestType(3)},
        { TestType(5), TestType(0), TestType(2)},
        { TestType(7), TestType(1), TestType(1)},
    }};

    static_assert(std::same_as<decltype(m), Matrix<TestType, 3, 3>>);

    REQUIRE( m[0][0] == TestType(1) );
    REQUIRE( m[0][1] == TestType(2) );
    REQUIRE( m[0][2] == TestType(3) );
    REQUIRE( m[1][0] == TestType(5) );
    REQUIRE( m[1][1] == TestType(0) );
    REQUIRE( m[1][2] == TestType(2) );
    REQUIRE( m[2][0] == TestType(7) );
    REQUIRE( m[2][1] == TestType(1) );
    REQUIRE( m[2][2] == TestType(1) ); 
}

TEMPLATE_TEST_CASE("matrix sum", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix m {{
        {TestType(1), TestType(2), TestType(3)},
        {TestType(1), TestType(2), TestType(3)},
        {TestType(1), TestType(2), TestType(3)}
    }};

    auto sum = m + m;
    for(auto i : std::views::iota(0, 3))
        for(auto j : std::views::iota(0, 3))
            REQUIRE(sum[i][j] == m[i][j] + m[i][j]);
}

TEMPLATE_TEST_CASE("matrix cprod", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix m{{
        {TestType(1), TestType(2), TestType(3)},
        {TestType(1), TestType(2), TestType(3)},
        {TestType(1), TestType(2), TestType(3)}
    }};

    auto cp = cprod(m, m);
    REQUIRE( cp[0][0] == TestType(6)  );
    REQUIRE( cp[0][1] == TestType(12) );
    REQUIRE( cp[0][2] == TestType(18) );
    REQUIRE( cp[1][0] == TestType(6)  );
    REQUIRE( cp[1][1] == TestType(12) );
    REQUIRE( cp[1][2] == TestType(18) );
    REQUIRE( cp[2][0] == TestType(6)  );
    REQUIRE( cp[2][1] == TestType(12) );
    REQUIRE( cp[2][2] == TestType(18) );

    Matrix store = cp;
    m = cprod(m, m);

    REQUIRE( m[0][0] == store[0][0] );
    REQUIRE( m[0][1] == store[0][1] );
    REQUIRE( m[0][2] == store[0][2] );
    REQUIRE( m[1][0] == store[1][0] );
    REQUIRE( m[1][1] == store[1][1] );
    REQUIRE( m[1][2] == store[1][2] );
    REQUIRE( m[2][0] == store[2][0] );
    REQUIRE( m[2][1] == store[2][1] );
    REQUIRE( m[2][2] == store[2][2] );

}


TEMPLATE_TEST_CASE("matrix random access", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix m { { 
        {TestType(1), TestType(2), TestType(3)}, 
        {TestType(4), TestType(5), TestType(6)}, 
        {TestType(7), TestType(8), TestType(9)}
    } };

    // call syntax
    REQUIRE( m(0, 0) == TestType(1) );
    REQUIRE( m(0, 1) == TestType(2) );
    REQUIRE( m(0, 2) == TestType(3) );
    REQUIRE( m(1, 0) == TestType(4) );
    REQUIRE( m(1, 1) == TestType(5) );
    REQUIRE( m(1, 2) == TestType(6) );
    REQUIRE( m(2, 0) == TestType(7) );
    REQUIRE( m(2, 1) == TestType(8) );
    REQUIRE( m(2, 2) == TestType(9) );
    
    // subscript syntax
    REQUIRE( m[0][0] == TestType(1) );
    REQUIRE( m[0][1] == TestType(2) );
    REQUIRE( m[0][2] == TestType(3) );
    REQUIRE( m[1][0] == TestType(4) );
    REQUIRE( m[1][1] == TestType(5) );
    REQUIRE( m[1][2] == TestType(6) );
    REQUIRE( m[2][0] == TestType(7) );
    REQUIRE( m[2][1] == TestType(8) );
    REQUIRE( m[2][2] == TestType(9) );
}

TEMPLATE_TEST_CASE("matrix transposition", "[linalg][matrix]", TYPE_PARAMETER_LIST) {
    Matrix m { { 
        {TestType(1), TestType(2), TestType(3)}, 
        {TestType(4), TestType(5), TestType(6)}, 
        {TestType(7), TestType(8), TestType(9)}
    } };


    REQUIRE( m[0][0] == TestType(1) );
    REQUIRE( m[0][1] == TestType(2) );
    REQUIRE( m[0][2] == TestType(3) );
    REQUIRE( m[1][0] == TestType(4) );
    REQUIRE( m[1][1] == TestType(5) );
    REQUIRE( m[1][2] == TestType(6) );
    REQUIRE( m[2][0] == TestType(7) );
    REQUIRE( m[2][1] == TestType(8) );
    REQUIRE( m[2][2] == TestType(9) );

    auto tm = transpose(m);
    REQUIRE( tm[0][0] == TestType(1) );
    REQUIRE( tm[0][1] == TestType(4) );
    REQUIRE( tm[0][2] == TestType(7) );
    REQUIRE( tm[1][0] == TestType(2) );
    REQUIRE( tm[1][1] == TestType(5) );
    REQUIRE( tm[1][2] == TestType(8) );
    REQUIRE( tm[2][0] == TestType(3) );
    REQUIRE( tm[2][1] == TestType(6) );
    REQUIRE( tm[2][2] == TestType(9) );
}

