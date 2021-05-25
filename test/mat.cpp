#include <iostream>
#include "linalg/matrix.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

using namespace linalg;

TEST_CASE("matrix sum", "[matrix]") {
    Matrix m {{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3}
    }};

    auto sum = m + m;
    REQUIRE(sum[0][0] == 2);

    auto prod = cprod(m, m);

    static_assert(contains_fixed_state_operation<decltype(sum)> == false);
    static_assert(contains_fixed_state_operation<decltype(prod)>);
    static_assert(contains_fixed_state_operation<decltype(prod + sum)>);

    [[maybe_unused]] Matrix m2 = prod;

    std::cout << "follows a state invalidation: ";
    m = prod;
}

TEST_CASE("matrix cprod", "[matrix]") {
    Matrix m{{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3}
    }};

    auto cp = cprod(m, m);
    REQUIRE(cp[0][0] == 6);
    REQUIRE(cp[0][1] == 12);
    REQUIRE(cp[0][2] == 18);
    REQUIRE(cp[1][0] == 6);
    REQUIRE(cp[1][1] == 12);
    REQUIRE(cp[1][2] == 18);
    REQUIRE(cp[2][0] == 6);
    REQUIRE(cp[2][1] == 12);
    REQUIRE(cp[2][2] == 18);
}

TEST_CASE("matrix transposition", "[matrix]") {
    Matrix m{{
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3}
    }};

    REQUIRE(m[0][0] == 1);
    REQUIRE(m[0][1] == 2);
    REQUIRE(m[0][2] == 3);
    REQUIRE(m[1][0] == 1);
    REQUIRE(m[1][1] == 2);
    REQUIRE(m[1][2] == 3);
    REQUIRE(m[2][0] == 1);
    REQUIRE(m[2][1] == 2);
    REQUIRE(m[2][2] == 3);

    auto tm = transpose(m);
    REQUIRE(tm[0][0] == 1);
    REQUIRE(tm[0][1] == 1);
    REQUIRE(tm[0][2] == 1);
    REQUIRE(tm[1][0] == 2);
    REQUIRE(tm[1][1] == 2);
    REQUIRE(tm[1][2] == 2);
    REQUIRE(tm[2][0] == 3);
    REQUIRE(tm[2][1] == 3);
    REQUIRE(tm[2][2] == 3);
}