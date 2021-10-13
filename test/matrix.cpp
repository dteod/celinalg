#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <celinalg/matrix.hpp>

using namespace celinalg;


SCENARIO("matrix iterators") {
    Matrix m { { // first one is for brace initialization, second one is list-initialization of a std::array
        {1, 2, 3}, 
        {4, 5, 6}, 
        {7, 8, 9}
    } };

    WHEN("element-wise") {
        THEN("row-first traversal") {
            for(auto counter = 1; auto element : m.elements_view()) {
                REQUIRE( element == counter++ );
            };
        }
        THEN("column-first traversal") {
            int sequence[] {1, 4, 7, 2, 5, 8, 3, 6, 9 };
            for(uint counter = 0; auto element : m.elements_view<MatrixDimension::BY_COLS>()) {
                REQUIRE( element == sequence[counter++]);
            }
        }
    }

    WHEN("dimension-wise") {
        THEN("row-wise access") {
            auto counter = 1;
            for(auto row : m.rows_view()) {
                for(auto element : row) {
                    REQUIRE( element == counter++ );
                }
            }
        }
        THEN("column-wise access") {
            auto counter = 0;
            int sequence[] { 1, 4, 7, 2, 5, 8, 3, 6, 9 };
            for(auto col : m.cols_view()) {
                for(auto element : col) {
                    REQUIRE( element == sequence[counter++] );
                }
            }
        }
    }
}

SCENARIO("matrix subscript") {
    Matrix m { { // first one is for brace initialization, second one is list-initialization of a std::array
        {1, 2, 3}, 
        {4, 5, 6}, 
        {7, 8, 9}
    } };

    WHEN("element-wise") {
        THEN("row-first traversal") {
            auto view = m.elements_view();
            for(size_t counter = 0; counter != m.numel(); ++counter) {
                REQUIRE( view[counter] == static_cast<int>(1 + counter) );
            };
        }
        THEN("column-first traversal") {
            auto view = m.elements_view<MatrixDimension::BY_COLS>();
            int sequence[] {1, 4, 7, 2, 5, 8, 3, 6, 9 };
            for(size_t counter = 0; counter != m.numel(); ++counter) {
                REQUIRE( view[counter] == sequence[counter] );
            };
        }
    }

    WHEN("dimension-wise") {
        THEN("row-wise") {
            auto counter = 1;
            for(auto row : m.rows_view()) {
                for(auto element : row) {
                    REQUIRE( element == counter++ );
                }
            }
        }
        THEN("column-wise") {
            auto counter = 0;
            int sequence[] {1, 4, 7, 2, 5, 8, 3, 6, 9 };
            for(auto col : m.cols_view()) {
                for(auto element : col) {
                    REQUIRE( element == sequence[counter++] );
                }
            }
        }
    }
}

template<class> class TD;

SCENARIO("matrix-matrix operations") {
    Matrix m1 { { // first one is for brace initialization, second one is list-initialization of a std::array
        {1, 2, 3}, 
        {4, 5, 6}, 
        {7, 8, 9}
    } };

    Matrix m2 { { // first one is for brace initialization, second one is list-initialization of a std::array
        {1, 2, 3}, 
        {4, 5, 6}, 
        {7, 8, 9}
    } };
    WHEN("adding matrices") {
        THEN("") {
            auto sum = m1 + m2;
            REQUIRE( sum[0][0] == 2   );
            REQUIRE( sum[0][1] == 4   );
            REQUIRE( sum[0][2] == 6   );
            REQUIRE( sum[1][0] == 8   );
            REQUIRE( sum[1][1] == 10  );
            REQUIRE( sum[1][2] == 12  );
            REQUIRE( sum[2][0] == 14  );
            REQUIRE( sum[2][1] == 16  );
            REQUIRE( sum[2][2] == 18  );

            Matrix mat = sum;
            
            REQUIRE( mat[0][0] == 2   );
            REQUIRE( mat[0][1] == 4   );
            REQUIRE( mat[0][2] == 6   );
            REQUIRE( mat[1][0] == 8   );
            REQUIRE( mat[1][1] == 10  );
            REQUIRE( mat[1][2] == 12  );
            REQUIRE( mat[2][0] == 14  );
            REQUIRE( mat[2][1] == 16  );
            REQUIRE( mat[2][2] == 18  );
        }
    }
}


SCENARIO("submatrix") {
    Matrix m { { // first one is for brace initialization, second one is list-initialization of a std::array
        {1, 2, 3}, 
        {4, 5, 6}, 
        {7, 8, 9}
    } };


    // TODO move this to a detached scenario
    // Matrices and submatrices are entirely usable in constant expressions
    constexpr int ce_val = [](){
        Matrix m { { // first one is for brace initialization, second one is list-initialization of a std::array
            {1, 2, 3}, 
            {4, 5, 6}, 
            {7, 8, 9}
        } };
        auto submatrix = m.submatrix(1, 3, 1, 2);
        submatrix = Matrix<int, 2, 1>();
        return m[1][1];
    }();

    WHEN("MATLAB-style matrix replacement") {
        THEN("") {
            auto submatrix = m.submatrix(1, 3, 1, 2); 
            // submatrix is a matrix view, dynamic by default. There is more freedom in using dynamic matrices,
            // operations on them will be rendered correct as long as dimensions agree.
            // No more static check, though.
            submatrix = Matrix<int, 2, 1>();

            // this is usable as well but it goes beyond the range (no runtime bound-check for speed)
            // submatrix = Matrix<int, 2, 2>();

            // this compiles correctly, values are evaluated in a constexpr context. 
            // Matrices will be fully constexpr as soon as compilers enable constexpr std::vector
            if constexpr(ce_val == 0) {
                REQUIRE(true);
            } else {
                REQUIRE(false);
            }
            REQUIRE(m[0][0] == 1);
            REQUIRE(m[0][1] == 2);
            REQUIRE(m[0][2] == 3);
            REQUIRE(m[1][0] == 4);
            REQUIRE(m[1][1] == 0);
            REQUIRE(m[1][2] == 0);
            REQUIRE(m[2][0] == 7);
            REQUIRE(m[2][1] == 8);
            REQUIRE(m[2][2] == 9);
        }
    }
}