#pragma once

#include <cmath>
#include <bit>

#include <traits.hpp>

namespace linalg {

namespace detail {
    
enum class Operation {
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    MODULO,
    AND,
    OR,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR
};

template<Operation op>
struct expression_operator {
    template<typename T1, typename T2>
    inline constexpr static auto call(T1&& x1, T2&& x2) { 
        if constexpr(op==Operation::ADDITION) {
            return std::forward<T1>(x1) + std::forward<T2>(x2);
        } else if constexpr(op==Operation::SUBTRACTION) {
            return std::forward<T1>(x1) - std::forward<T2>(x2);
        } else if constexpr(op==Operation::MULTIPLICATION) {
            return std::forward<T1>(x1) * std::forward<T2>(x2);
        } else if constexpr(op==Operation::DIVISION) {
            return std::forward<T1>(x1) / std::forward<T2>(x2);
        } else if constexpr(op==Operation::MODULO) {
            if constexpr(req::floating_point<std::decay_t<T1>> || req::floating_point<std::decay_t<T2>>) {
                return std::fmod(std::forward<T1>(x1), std::forward<T2>(x2));
            } else if constexpr(req::complex<std::decay_t<T1>> || req::complex<std::decay_t<T2>>) {
                return std::forward<T1>(x1) - (std::forward<T1>(x1)/std::forward<T2>(x2))*std::forward<T2>(x2);
            } else {
                return std::forward<T1>(x1) % std::forward<T2>(x2);
            }
        } else if constexpr(op==Operation::OR) {
            if constexpr(req::complex<std::decay_t<T1>> || req::complex<std::decay_t<T2>>) {
                return static_cast<bool>(std::norm(std::forward<T1>(x1))) || static_cast<bool>(std::norm(std::forward<T2>(x2)));
            } else {
                return static_cast<bool>(std::forward<T1>(x1)) || static_cast<bool>(std::forward<T2>(x2));
            }
        } else if constexpr(op==Operation::AND) {
            if constexpr(req::complex<std::decay_t<T1>> || req::complex<std::decay_t<T2>>) {
                return static_cast<bool>(std::norm(std::forward<T1>(x1))) && static_cast<bool>(std::norm(std::forward<T2>(x2)));
            } else {
                return static_cast<bool>(std::forward<T1>(x1)) && static_cast<bool>(std::forward<T2>(x2));
            }
        } else if constexpr(op==Operation::BITWISE_OR) {
            return 
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T1>)>>(std::forward<T1>(x1)) |
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T2>)>>(std::forward<T2>(x2)) ;
        } else if constexpr(op==Operation::BITWISE_AND) {
            return 
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T1>)>>(std::forward<T1>(x1)) &
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T2>)>>(std::forward<T2>(x2)) ;
        } else if constexpr(op==Operation::BITWISE_XOR) {
            return 
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T1>)>>(std::forward<T1>(x1)) ^
                std::bit_cast<traits::unsigned_of_size_t<sizeof(std::decay_t<T2>)>>(std::forward<T2>(x2)) ;
        } 
    }
};


}

}