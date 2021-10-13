#pragma once

#include "celinalg/forward.hpp"
#include "celinalg/utils.hpp"
#include "celinalg/matrix_views.hpp"
#include "celinalg/operation.hpp"
#include "celinalg/matrix.hxx"

namespace celinalg {

namespace traits {

template<typename T>
struct fixed_state_operation {
    inline constexpr static bool value = []() constexpr {
        if constexpr(T::is_expression) {
            if constexpr(T::is_view) {
                return fixed_state_operation<typename T::matrix_type>::value;
            } else {
                return fixed_state_operation<typename T::operand_type_1>::value || fixed_state_operation<typename T::operand_type_2>::value;
            }
        } else {
            return false;
        }
    }();
};

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
struct fixed_state_operation<celinalg::detail::MatrixCrossProductExpression<M1, M2>>: std::true_type {};

template<typename T>
using fixed_state_operation_v = fixed_state_operation<T>::value;

}

template<typename T>
concept contains_fixed_state_operation = celinalg::traits::fixed_state_operation<T>::value;

}