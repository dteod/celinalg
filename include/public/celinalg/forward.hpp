#pragma once

#include <concepts>

#include "celinalg/number.hpp"
#include "celinalg/math.hpp"


namespace std {
// TODO remove when it will be implemented

template<typename B>
concept __boolean_testable_impl = std::convertible_to<B, bool>;

template<typename B>
concept boolean_testable =                       // exposition only
    __boolean_testable_impl<B> &&
    requires (B&& b) {
        { !std::forward<B>(b) } -> __boolean_testable_impl;
    };

}

namespace celinalg {

template<req::number, size_t>
class Vector;

template<req::number T>
using DynamicVector = Vector<T, 0>;

template<req::number, size_t, size_t>
class Matrix;

template<req::number>
class Quaternion;

template<typename T> concept expression_participant = requires { { T::is_temporary } -> std::boolean_testable; };

template<typename T> concept container = req::number<typename T::value_type>;

template<typename T>
concept matrix = expression_participant<T> && container<T> && requires {
    { T::static_size } -> std::convertible_to<size_t>;
    { T::static_rows } -> std::convertible_to<size_t>;
    { T::static_cols } -> std::convertible_to<size_t>;
};

template<typename T>
concept static_matrix = matrix<T> && ( T::static_size != 0 );

template<typename T>
concept dynamic_rows_matrix = matrix<T> && (T::static_rows == 0);

template<typename T>
concept dynamic_cols_matrix = matrix<T> && (T::static_cols == 0);

template<typename T>
concept dynamic_matrix = dynamic_rows_matrix<T> && dynamic_cols_matrix<T>;

template<typename T>
concept square_matrix = static_matrix<T> && (T::static_cols == T::static_rows);

template<typename M1, typename M2>
concept suitable_matrix_same_size_expression = matrix<M1> && matrix<M2> && (dynamic_matrix<M1> || dynamic_matrix<M2> || (M1::static_rows == M2::static_rows && M1::static_cols == M2::static_cols));

template<typename M1, typename M2>
concept suitable_matrix_cross_size_expression = matrix<M1> && matrix<M2> && (dynamic_matrix<M1> || dynamic_matrix<M2> || (M1::static_cols == M2::static_rows));

template<typename T>
concept vector = expression_participant<T> && container<T> && requires {
    { T::static_size } -> std::convertible_to<size_t>;
} && !matrix<T>;
// } && (!matrix<T> || (matrix<T> && (T::static_rows == 1 || T::static_cols == 1))); 
// in theory, Nx1 matrices are N-sized vectors. 
// This greatly increases metaprogramming effort though
// To be implemented someday

template<typename T>
concept static_vector = vector<T> && ( T::static_size != 0 );

template<typename T>
concept dynamic_vector = vector<T> && ( T::static_size == 0 );

template<typename V1, typename V2>
concept vectors_same_value_type = vector<V1> && vector<V2> && std::same_as<typename V1::value_type, typename V2::value_type>;

template<typename V1, typename V2>
concept suitable_vector_expression = vector<V1> && vector<V2> && (dynamic_vector<V1> || dynamic_vector<V2> || (V1::static_size == V2::static_size));

template<typename V1, typename V2>
concept suitable_vector_cross_product_expression = suitable_vector_expression<V1, V2> && (dynamic_vector<V1> || V1::static_size == 3) && (dynamic_vector<V2> || V2::static_size == 3);

template<typename M, typename V>
concept suitable_matrix_vector_multiplication = matrix<M> && vector<V> && (dynamic_matrix<M> || dynamic_vector<V> || (M::static_cols == V::static_size) );

template<typename V, typename M>
concept suitable_vector_matrix_multiplication = vector<M> && matrix<V> && (dynamic_vector<V> || dynamic_matrix<M> || (V::static_size == M::static_rows) );

template<typename X, typename Y>
concept suitable_cross_size_expression = suitable_matrix_cross_size_expression<X, Y> || suitable_matrix_vector_multiplication<X, Y> || suitable_vector_matrix_multiplication<X, Y>;

template<typename X, typename Y>
concept suitable_same_size_expression = suitable_matrix_same_size_expression<X, Y> || suitable_vector_expression<X, Y>;

namespace detail {

struct default_index_picker {
    inline constexpr static size_t pick(const auto&, size_t index) noexcept {
        return index;
    } 
};

template<typename Element, typename IndexPicker = default_index_picker> requires(container<Element> && expression_participant<Element>)
class linear_element_iterator {
    std::conditional_t<Element::is_temporary, Element, Element&> e;
    ssize_t m_index;
    template<container Tmp> friend linear_element_iterator<Tmp> operator+(ssize_t, linear_element_iterator<Tmp>&);
    template<container Tmp> friend const linear_element_iterator<Tmp> operator+(ssize_t, const linear_element_iterator<Tmp>&);
public:
    constexpr linear_element_iterator(Element& e, ssize_t index) noexcept: e{e}, m_index{index} {}
    inline constexpr decltype(auto) operator*() const noexcept { return e.pick(IndexPicker::pick(e, m_index)); }
    inline constexpr decltype(auto) operator[](ssize_t index) const noexcept { return e.pick(IndexPicker::pick(e, m_index + index)); }

    constexpr auto& operator++() noexcept { m_index++; return *this; }
    constexpr auto operator++(int) noexcept { linear_element_iterator it(e, m_index++); return it; }
    constexpr auto& operator--() noexcept { m_index--; return *this; }
    constexpr auto operator--(int) noexcept { linear_element_iterator it(e, m_index--); return it; }

    inline constexpr auto operator+(ssize_t index) const noexcept { linear_element_iterator it(e, m_index + index); return it; }
    inline constexpr auto operator-(ssize_t index) const noexcept { linear_element_iterator it(e, m_index - index); return it; }
    inline constexpr auto& operator+=(ssize_t index) noexcept { m_index += index; return *this; }
    inline constexpr auto& operator-=(ssize_t index) noexcept { m_index -= index; return *this; }

    inline constexpr auto operator-(linear_element_iterator it) const noexcept { return m_index - it.m_index; }

    inline constexpr bool operator==(linear_element_iterator it) const noexcept { return (&e == &it.e) && (m_index == it.m_index);}
    inline constexpr bool operator!=(linear_element_iterator it) const noexcept { return (&e != &it.e) || (m_index != it.m_index);}

    inline constexpr std::partial_ordering operator<=>(linear_element_iterator it) const noexcept { 
        return (&e != &it.e) ? std::partial_ordering::unordered : m_index <=> it.m_index
        ; 
    }

    using difference_type = ptrdiff_t;
    using value_type = typename std::decay_t<Element>::value_type;
    // using pointer = value_type*;
    using reference = decltype(std::declval<linear_element_iterator>()[std::declval<size_t>()]);
    using iterator_category = std::random_access_iterator_tag;
};

template<typename Element> constexpr linear_element_iterator<Element> operator+(ssize_t index, const linear_element_iterator<Element>& it) { return linear_element_iterator<Element>(it.e, index + it.m_index); }
template<typename Element> constexpr linear_element_iterator<const Element> operator+(ssize_t index, const linear_element_iterator<const Element>& it) { return linear_element_iterator<const Element>(it.e, index + it.m_index); }

}

}