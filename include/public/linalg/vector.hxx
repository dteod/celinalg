#pragma once

#include <cassert>

#include <stdexcept>
#include <algorithm>
#include <numeric>

#include "linalg/vector.hpp"

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
    inline constexpr static auto call(auto&& v1, auto&& v2) { 
        if constexpr(op==Operation::ADDITION) {
            return std::forward<decltype(v1)>(v1) + std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::SUBTRACTION) {
            return std::forward<decltype(v1)>(v1) - std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::MULTIPLICATION) {
            return std::forward<decltype(v1)>(v1) * std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::DIVISION) {
            return std::forward<decltype(v1)>(v1) / std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::MODULO) {
            return std::forward<decltype(v1)>(v1) % std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::OR) {
            return std::forward<decltype(v1)>(v1) || std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::AND) {
            return std::forward<decltype(v1)>(v1) && std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::BITWISE_OR) {
            return std::forward<decltype(v1)>(v1) | std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::BITWISE_AND) {
            return std::forward<decltype(v1)>(v1) & std::forward<decltype(v2)>(v2);
        } else if constexpr(op==Operation::BITWISE_XOR) {
            return std::forward<decltype(v1)>(v1) ^ std::forward<decltype(v2)>(v2);
        } 
    }
};

template<Operation op, vector V1, vector V2> requires suitable_vec_expression<V1, V2>
class VecExpression {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
public:
    inline constexpr static bool is_temporary { true }; 
    inline constexpr static size_t static_size { ((dynamic_vector<V1> || dynamic_vector<V2>) ? 0 : V1::static_size) };
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
    using iterator = linear_element_iterator<VecExpression<op, V1, V2>>;
    constexpr VecExpression(const V1& v1, const V2& v2): v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept(static_vector<V1> && static_vector<V2>) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size()) {
                throw std::length_error("size mismatch");
            }
        }
        return v1.size();
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        return expression_operator<op>::call(v1[index], v2[index]);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        return expression_operator<op>::call(v1.at(index), v2.at(index));
    }

    inline constexpr auto begin() const { return iterator(*this, 0); }
    inline constexpr auto cbegin() const { return iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() const { return iterator(*this, size()); }
    inline constexpr auto cend() const { return iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
};

template<Operation op, vector Vec, req::number Scalar>
class VecScalarExpression {
    std::conditional_t<Vec::is_temporary, Vec, const Vec&> vec;
    Scalar s;
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { Vec::static_size };
    using value_type = traits::common_type_t<typename Vec::value_type, Scalar>;
    using iterator = linear_element_iterator<VecScalarExpression<op, Vec, Scalar>>;

    constexpr VecScalarExpression(const Vec& vec, const Scalar& s) noexcept : vec{vec}, s{s} {}

    inline constexpr size_t size() const noexcept {
        auto sz = vec.size();
        return sz;
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        return expression_operator<op>::call(vec[index], s);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        return expression_operator<op>::call(vec.at(index), s);
    }

    inline constexpr auto begin() const { return iterator(*this, 0); }
    inline constexpr auto cbegin() const { return iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() const { return iterator(*this, size()); }
    inline constexpr auto cend() const { return iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
};


template<Operation op, req::number Scalar, vector Vec>
class ScalarVecExpression {
    Scalar s;
    std::conditional_t<Vec::is_temporary, Vec, const Vec&> vec;
public:
    inline constexpr static bool is_temporary { true }; 
    inline constexpr static size_t static_size { Vec::static_size };
    using value_type = traits::common_type_t<typename Vec::value_type, Scalar>;
    using iterator = linear_element_iterator<ScalarVecExpression<op, Scalar, Vec>>;
    constexpr ScalarVecExpression(const Scalar& s, const Vec& vec) noexcept: s{s}, vec{vec} {}

    inline constexpr size_t size() const noexcept {
        return vec.size();
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { 
        return expression_operator<op>::call(s, vec[index]);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        return expression_operator<op>::call(s, vec.at(index));
    }

    inline constexpr auto begin() const { return iterator(*this, 0); }
    inline constexpr auto cbegin() const { return iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() const { return iterator(*this, size()); }
    inline constexpr auto cend() const { return iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
};

template<vector V1, vector V2> requires suitable_vec_expression<V1, V2>
class VectorScalarProduct {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
public:
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
    
    constexpr VectorScalarProduct(const V1& v1, const V2& v2) noexcept: v1{v1}, v2{v2} {}
    
    inline constexpr operator value_type() const noexcept {
        return get();
    }

    inline constexpr value_type get() const noexcept {
        return std::inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<value_type>(0));
    }
};

template<vector V1, vector V2> requires suitable_cross_product_expression<V1, V2>
class CrossProductExpression {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { 3 };
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
    using iterator = linear_element_iterator<CrossProductExpression<V1, V2>>;

    constexpr CrossProductExpression(const V1& v1, const V2& v2) noexcept : v1{v1}, v2{v2} {
        assert(v1.size() == v2.size() && v1.size() == 3);
    }

    inline constexpr size_t size() const noexcept { return v1.size() == v2.size() && v1.size() == 3 ? 3 : 0; }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        size_t idx0 = index, idx1 = (1 + index)%3;
        value_type out = v1[idx0]*v2[idx1] - v1[idx1] * v2[idx0];
        return index == 0 || index == 2 ? out : -out ;
    }

    inline constexpr decltype(auto) at(size_t index) const {
        if(index > 2)
            throw std::out_of_range();
        size_t idx0 = index, idx1 = (1 + index)%3;
        value_type out = v1.at(idx0) * v2.at(idx1) - v1.at(idx1) * v2.at(idx0);
        return index == 0 || index == 2 ? out : -out ;
    }
};

}

template<vector V1, vector V2> inline auto operator+ (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::ADDITION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator- (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::SUBTRACTION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator* (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::MULTIPLICATION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator/ (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::DIVISION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator% (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::MODULO, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator&&(const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::AND, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator||(const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::OR, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator& (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::BITWISE_AND, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator| (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::BITWISE_OR, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator^ (const V1& v1, const V2& v2) { return detail::VecExpression<detail::Operation::BITWISE_XOR, V1, V2>(v1, v2); }

template<vector V, req::number S> inline auto operator+ (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::ADDITION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator- (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::SUBTRACTION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator* (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::MULTIPLICATION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator/ (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::DIVISION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator% (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::MODULO, V, S>(v, s); }
template<vector V, req::number S> inline auto operator&&(const V& v, S s) { return detail::VecScalarExpression<detail::Operation::AND, V, S>(v, s); }
template<vector V, req::number S> inline auto operator||(const V& v, S s) { return detail::VecScalarExpression<detail::Operation::OR, V, S>(v, s); }
template<vector V, req::number S> inline auto operator& (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::BITWISE_AND, V, S>(v, s); }
template<vector V, req::number S> inline auto operator| (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::BITWISE_OR, V, S>(v, s); }
template<vector V, req::number S> inline auto operator^ (const V& v, S s) { return detail::VecScalarExpression<detail::Operation::BITWISE_XOR, V, S>(v, s); }

template<req::number S, vector V> inline auto operator+ (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::ADDITION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator- (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::SUBTRACTION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator* (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::MULTIPLICATION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator/ (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::DIVISION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator% (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::MODULO, S, V>(s, v); }
template<req::number S, vector V> inline auto operator&&(S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::AND, S, V>(s, v); }
template<req::number S, vector V> inline auto operator||(S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::OR, S, V>(s, v); }
template<req::number S, vector V> inline auto operator& (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::BITWISE_AND, S, V>(s, v); }
template<req::number S, vector V> inline auto operator| (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::BITWISE_OR, S, V>(s, v); }
template<req::number S, vector V> inline auto operator^ (S s, const V& v) { return detail::ScalarVecExpression<detail::Operation::BITWISE_XOR, S, V>(s, v); }

template<vector V1, vector V2> requires suitable_cross_product_expression<V1, V2>
inline constexpr auto cprod(const V1& v1, const V2& v2) noexcept { return detail::CrossProductExpression(v1, v2); }

template<vector V1, vector V2> requires suitable_vec_expression<V1, V2>
inline constexpr auto sprod(const V1& v1, const V2& v2) noexcept { return detail::VectorScalarProduct(v1, v2); }

}