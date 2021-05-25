#pragma once

namespace linalg::utils {

template<typename T1, typename T2>
constexpr bool expression_reference_check_state_invalidation(const T1* self, const T2* p) {
    if constexpr(std::same_as<T1, T2>) {
        return self == p;
    } else {
        if constexpr(T2::is_view) {
            return expression_reference_check_state_invalidation(self, &p->m);
        } else if constexpr(T2::is_expression) {
            return expression_reference_check_state_invalidation(self, &p->m1) || expression_reference_check_state_invalidation(self, &p->m2);
        } else {
            return false;
        }
    }
}

}