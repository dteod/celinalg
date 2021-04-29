#pragma once

#include <requirements.hpp>
#include <concepts>

namespace linalg {

template<req::number, size_t>
class Elementtor;

template<req::number T>
using DynamicElementtor = Elementtor<T, 0>;

template<req::number, size_t, size_t>
class Matrix;

template<req::number>
class Quaternion;

template<typename T> concept expression_participant = requires { { T::is_temporary } -> std::convertible_to<bool>; };

template<typename T> concept container = requires { typename T::value_type; } && req::number<typename T::value_type>;

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

template<typename M1, typename M2>
concept suitable_matrix_addition_expression = matrix<M1> && matrix<M2> && (dynamic_matrix<M1> || dynamic_matrix<M2> || (M1::static_rows == M2::static_rows && M1::static_cols == M2::static_cols));

template<typename M1, typename M2>
concept suitable_matrix_multiplication_expression = matrix<M1> && matrix<M2> && (dynamic_matrix<M1> || dynamic_matrix<M2> || (M1::static_cols == M2::static_rows));


template<typename T>
concept vector = expression_participant<T> && container<T> && requires {
    { T::static_size } -> std::convertible_to<size_t>;
} && !matrix<T>;

template<typename T>
concept static_vector = vector<T> && ( T::static_size != 0 );

template<typename T>
concept dynamic_vector = vector<T> && ( T::static_size == 0 );

template<typename V1, typename V2>
concept suitable_vec_expression = vector<V1> && vector<V2> && (dynamic_vector<V1> || dynamic_vector<V2> || (V1::static_size == V2::static_size));

template<typename V1, typename V2>
concept suitable_cross_product_expression = suitable_vec_expression<V1, V2> && (dynamic_vector<V1> || V1::static_size == 3) && (dynamic_vector<V2> || V2::static_size == 3);

struct default_index_picker {
    inline constexpr static size_t pick(const auto& element, size_t index) {
        return index;
    } 
};


template<typename Element, typename IndexPicker = default_index_picker> requires(container<Element> && expression_participant<Element>)
class linear_element_iterator {
    std::conditional_t<Element::is_temporary, Element, Element&> v;
    ssize_t m_index;
    template<container Tmp> friend linear_element_iterator<Tmp> operator+(ssize_t, linear_element_iterator<Tmp>&);
    template<container Tmp> friend const linear_element_iterator<Tmp> operator+(ssize_t, const linear_element_iterator<Tmp>&);
public:
    constexpr linear_element_iterator(Element& v, ssize_t index): v{v}, m_index{index} {}
    inline constexpr decltype(auto) operator*() const { return v[IndexPicker::pick(v, m_index)]; }
    inline constexpr decltype(auto) operator[](ssize_t index) const { return v[IndexPicker::pick(v, m_index + index)]; }

    constexpr auto& operator++() noexcept { m_index++; return *this; }
    constexpr auto operator++(int) noexcept { linear_element_iterator it(v, m_index++); return it; }
    constexpr auto& operator--() noexcept { m_index--; return *this; }
    constexpr auto operator--(int) noexcept { linear_element_iterator it(v, m_index--); return it; }

    inline constexpr auto operator+(ssize_t index) const noexcept { linear_element_iterator it(v, m_index + index); return it; }
    inline constexpr auto operator-(ssize_t index) const noexcept { linear_element_iterator it(v, m_index - index); return it; }
    inline constexpr auto& operator+=(ssize_t index) noexcept { m_index += index; return *this; }
    inline constexpr auto& operator-=(ssize_t index) noexcept { m_index -= index; return *this; }

    inline constexpr auto operator-(linear_element_iterator it) const noexcept { return m_index - it.m_index; }

    inline constexpr bool operator==(linear_element_iterator it) const noexcept { return (&v == &it.v) && (m_index == it.m_index);}
    inline constexpr bool operator!=(linear_element_iterator it) const noexcept { return (&v != &it.v) || (m_index != it.m_index);}

    inline constexpr std::partial_ordering operator<=>(linear_element_iterator it) const noexcept { 
        return (&v != &it.v) ? std::partial_ordering::unordered : m_index <=> it.m_index
        ; 
    }

    using difference_type = ptrdiff_t;
    using value_type = typename std::decay_t<Element>::value_type;
    // using pointer = value_type*;
    using reference = decltype(std::declval<linear_element_iterator>()[std::declval<size_t>()]);
    using iterator_category = std::random_access_iterator_tag;
};

template<typename Element> constexpr linear_element_iterator<Element> operator+(ssize_t index, linear_element_iterator<Element>& it) { return linear_element_iterator<Element>(it.v, index + it.m_index); }
template<typename Element> constexpr const linear_element_iterator<Element> operator+(ssize_t index, const linear_element_iterator<Element>& it) { return linear_element_iterator<Element>(it.v, index + it.m_index); }


}