#pragma once

#include "linalg/forward.hpp"
#include "traits/common_type.hpp"

#include <array>
#include <vector>
#include <span>

#include "linalg/vector_view.hpp"

namespace linalg {

template<req::number T, size_t Size>
class Vector {
public:
    using value_type = T;
    inline constexpr static size_t static_size = Size;
private:
    inline constexpr static bool dynamic = (static_size == 0);
    inline constexpr static bool heap_allocated = dynamic || sizeof(value_type)*static_size > 16 ;
    using implementation_type = std::conditional_t<heap_allocated, std::vector<value_type>, std::array<value_type, static_size>>;
    implementation_type m_data;
    using iterator = typename implementation_type::iterator;
    using const_iterator = typename implementation_type::const_iterator;

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    inline constexpr static Vector ones() requires(!dynamic) {
        Vector v;
        if constexpr(heap_allocated) {
            v.m_data = std::vector<value_type>(static_size, static_cast<value_type>(1));
        } else {
            std::fill(v.begin(), v.end(), static_cast<value_type>(1));
        }
        return v;
    }

    inline constexpr static Vector ones(size_t size) requires(dynamic) {
        Vector v;
        if constexpr(dynamic) {
            v.m_data = std::vector<value_type>(size, static_cast<value_type>(1));
        }
        return v;
    }

    inline constexpr static Vector zeros(size_t size) requires(dynamic) {
        Vector v;
        if constexpr(dynamic) {
            v.m_data = std::vector<value_type>(size, static_cast<value_type>(0));
        }
        return v;
    }

    inline constexpr static bool is_temporary { false };
    constexpr Vector() noexcept {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        std::fill(m_data.begin(), m_data.end(), static_cast<value_type>(0));
    }


    template<size_t ArrSize> requires((ArrSize == static_size) && (!dynamic))
    constexpr Vector(const value_type(&array)[ArrSize]) noexcept(!heap_allocated) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic) {
            std::copy(std::begin(array), std::end(array), m_data.begin());
        }
    }

    constexpr Vector(const std::array<value_type, static_size>& array) noexcept(!heap_allocated) requires(!dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic) {
            std::copy(std::begin(array), std::end(array), m_data.begin());
        }
    }

    constexpr Vector(const std::span<value_type, static_size>& span) noexcept(!heap_allocated) requires(!dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic) {
            std::copy(std::begin(span), std::end(span), m_data.begin());
        }
    }

    constexpr Vector(const std::span<value_type>& span) requires(dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(span.size());
        }
        std::copy(std::begin(span), std::end(span), m_data.begin());
    }

    constexpr Vector(const std::vector<value_type>& vector) requires(dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(vector.size());
        }
        std::copy(std::begin(vector), std::end(vector), m_data.begin());
    }

    constexpr Vector(std::vector<value_type>&& vector) noexcept requires(dynamic) {
        if constexpr(heap_allocated)
            m_data.swap(vector);
    }

    constexpr Vector(std::convertible_to<value_type> auto&&... args) noexcept(!heap_allocated)
    requires ( sizeof...(args) != 0 && !(sizeof...(args) != 1 && (vector<std::remove_reference_t<decltype(args)>> && ...)) )
        : m_data{std::forward<decltype(args)>(args)...}
    {}

    template<vector V>
    constexpr Vector(const V& vec) noexcept(!heap_allocated) requires suitable_vector_expression<Vector, V>{
        if constexpr(heap_allocated) {
            m_data.resize(vec.size());
        }
        std::copy(vec.begin(), vec.end(), m_data.begin());
    }

    constexpr Vector(Vector&& vec) noexcept {
        if constexpr(heap_allocated) {
            m_data.swap(vec.m_data);
        } else {
            std::copy(vec.begin(), vec.end(), m_data.begin());
        }
    }

    template<vector V>
    inline constexpr decltype(auto) operator=(const V& vec) noexcept(!heap_allocated) requires suitable_vector_expression<Vector, V>{
        if constexpr(heap_allocated) {
            m_data.resize(vec.size());
        }
        std::copy(vec.begin(), vec.end(), m_data.begin());
        return *this;
    }

    inline constexpr decltype(auto) operator=(Vector&& vec) noexcept {
        if constexpr(heap_allocated) {
            m_data.swap(vec.m_data);
        } else {
            std::copy(vec.begin(), vec.end(), m_data.begin());
        }
        return *this;
    }

    template<size_t ArrSize> requires(ArrSize == static_size)
    inline constexpr decltype(auto) operator=(const value_type(&array)[ArrSize]) noexcept(!heap_allocated) requires(!dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic)
            std::copy(std::begin(array), std::end(array), m_data.begin());
        return *this;
    }

    inline constexpr decltype(auto) operator=(const std::array<value_type, static_size>& array) noexcept(!heap_allocated) requires(!dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic)
            std::copy(std::begin(array), std::end(array), m_data.begin());
        return *this;
    }

    inline constexpr decltype(auto) operator=(const std::span<value_type, static_size>& span) noexcept(!heap_allocated) requires(!dynamic) {
        if constexpr(heap_allocated) {
            m_data.resize(static_size);
        }
        if constexpr(!dynamic)
            std::copy(std::begin(span), std::end(span), m_data.begin());
        return *this;
    }

    inline constexpr decltype(auto) operator=(const std::span<value_type>& span) noexcept(!heap_allocated) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.resize(span.size());
        }
        std::copy(std::begin(span), std::end(span), m_data.begin());
        return *this;
    }

    inline constexpr size_t size() const noexcept { 
        return m_data.size();
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        return (m_data[index]);
    }

    inline constexpr decltype(auto) operator[](size_t index) noexcept {
        return (m_data[index]);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        return m_data.at(index);
    }

    inline constexpr decltype(auto) at(size_t index) {
        return m_data.at(index);
    }

    inline constexpr decltype(auto) front() const noexcept { return *begin(); }
    inline constexpr decltype(auto) front() noexcept { return *begin(); }
    inline constexpr decltype(auto) back() const noexcept { return *rbegin(); }
    inline constexpr decltype(auto) back() noexcept { return *rbegin(); }

    inline constexpr auto begin() const noexcept { return m_data.begin(); }
    inline constexpr auto rbegin() const noexcept { return m_data.rbegin(); }
    inline constexpr auto cbegin() const noexcept { return m_data.cbegin(); }
    inline constexpr auto crbegin() const noexcept { return m_data.crbegin(); }
    inline constexpr auto end() const noexcept { return m_data.end(); }
    inline constexpr auto rend() const noexcept { return m_data.rend(); }
    inline constexpr auto cend() const noexcept { return m_data.cend(); }
    inline constexpr auto crend() const noexcept { return m_data.crend(); }

    inline constexpr auto begin() noexcept { return m_data.begin(); }
    inline constexpr auto rbegin() noexcept { return m_data.rbegin(); }
    inline constexpr auto end() noexcept { return m_data.end(); }
    inline constexpr auto rend() noexcept { return m_data.rend(); }

    inline constexpr void reserve(size_t size) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.reserve(size);
        }
    }

    inline constexpr void resize(size_t size) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.resize(size);
        }
    }

    inline constexpr void resize(size_t size, const value_type& v) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.resize(size, v);
        }
    }

    inline constexpr void push_back(value_type v) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.push_back(v);
        }
    }

    inline constexpr void emplace(size_t pos, auto&&... args) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.emplace(pos, std::forward<decltype(args)>(args)...);
        }
    }

    inline constexpr void emplace_back(auto&&... args) requires(dynamic) {
        if constexpr(dynamic) {
            m_data.emplace_back(std::forward<decltype(args)>(args)...);
        }
    }

    inline constexpr void pop_back() noexcept requires(dynamic) {
        if constexpr(dynamic) {
            m_data.pop_back();
        }
    }

    inline constexpr void swap(Vector& vec) noexcept requires(heap_allocated) {
        if constexpr(dynamic) {
            m_data.swap(vec.m_data);
        }
    }

    inline constexpr bool empty() const noexcept requires(dynamic) {
        if constexpr(dynamic) {
            return m_data.empty();
        } else {
            // Code written here is unreachable, it is only written for the type-system
            return false;
        }
    }

    inline constexpr auto erase(auto first, auto last) requires(dynamic) {
        if constexpr(dynamic) {
            return m_data.erase(first, last);
        } else {
            return begin();
        }
    }

    inline constexpr auto erase(auto position) requires(dynamic) {
        if constexpr(dynamic) {
            return m_data.erase(position);
        } else {
            return begin();
        }
    }

    inline constexpr auto data() const& noexcept {
        if constexpr(heap_allocated) {
            return std::span<const value_type>(m_data.begin(), m_data.end());
        } else {
            return std::span(m_data);
        }
    }

    inline constexpr auto subvector(size_t begin = 0) const& noexcept {
        return detail::VectorView(*this, begin);
    }

    inline constexpr auto subvector(size_t begin, size_t end) const& noexcept {
        return detail::VectorView(*this, begin, end);
    }

    inline constexpr auto subvector(size_t begin = 0)& noexcept {
        return detail::VectorView(*this, begin);
    }

    inline constexpr auto subvector(size_t begin, size_t end)& noexcept {
        return detail::VectorView(*this, begin, end);
    }

    inline constexpr auto data() & noexcept {
        if constexpr(dynamic) {
            return std::span<value_type>(m_data.begin(), m_data.end());
        } else {
            return std::span<value_type, static_size>(m_data.begin(), m_data.end());
        }
    }

    template<vector V> inline constexpr decltype(auto) operator+=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this + v; 
        return *this; 
    }

    template<vector V> inline constexpr decltype(auto) operator-=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this - v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator*=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this * v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator/=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this / v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator%=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this % v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator&=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this & v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator|=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this | v;
        return *this;
    } 

    template<vector V> inline constexpr decltype(auto) operator^=(const V& v) requires suitable_vector_expression<Vector, V> {
        *this = *this ^ v;
        return *this;
    }

    inline constexpr decltype(auto) operator+=(value_type v) {
        *this = *this + v;
        return *this;
    } 

    inline constexpr decltype(auto) operator-=(value_type v) {
        *this = *this - v;
        return *this;
    } 

    inline constexpr decltype(auto) operator*=(value_type v) {
        *this = *this * v;
        return *this;
    } 

    inline constexpr decltype(auto) operator/=(value_type v) {
        *this = *this / v;
        return *this;
    } 

    inline constexpr decltype(auto) operator%=(value_type v) {
        *this = *this % v;
        return *this;
    } 

    inline constexpr decltype(auto) operator&=(value_type v) {
        *this = *this & v;
        return *this;
    } 

    inline constexpr decltype(auto) operator|=(value_type v) {
        *this = *this | v;
        return *this;
    } 

    inline constexpr decltype(auto) operator^=(value_type v) {
        *this = *this ^ v;
        return *this;
    }
};

namespace detail {
    inline constexpr auto begin(vector auto&& v) noexcept { return std::forward<decltype(v)>(v).begin(); }
    inline constexpr auto end(vector auto&& v) noexcept { return std::forward<decltype(v)>(v).end(); }
    inline constexpr auto rbegin(vector auto&& v) noexcept { return std::forward<decltype(v)>(v).rbegin(); }
    inline constexpr auto rend(vector auto&& v) noexcept { return std::forward<decltype(v)>(v).rend(); }
    inline constexpr auto cbegin(const vector auto& v) noexcept { return std::forward<decltype(v)>(v).cbegin(); }
    inline constexpr auto cend(const vector auto& v) noexcept { return std::forward<decltype(v)>(v).cend(); }
    inline constexpr auto crbegin(const vector auto& v) noexcept { return std::forward<decltype(v)>(v).crbegin(); }
    inline constexpr auto crend(const vector auto& v) noexcept { return std::forward<decltype(v)>(v).crend(); }
}

inline constexpr auto begin(vector auto&& v) noexcept { return detail::begin(std::forward<decltype(v)>(v)); }
inline constexpr auto end(vector auto&& v) noexcept { return detail::end(std::forward<decltype(v)>(v)); }
inline constexpr auto rbegin(vector auto&& v) noexcept { return detail::rbegin(std::forward<decltype(v)>(v)); }
inline constexpr auto rend(vector auto&& v) noexcept { return detail::rend(std::forward<decltype(v)>(v)); }
inline constexpr auto cbegin(const vector auto& v) noexcept { return detail::cbegin(std::forward<decltype(v)>(v)); }
inline constexpr auto cend(const vector auto& v) noexcept { return detail::cend(std::forward<decltype(v)>(v)); }
inline constexpr auto crbegin(const vector auto& v) noexcept { return detail::crbegin(std::forward<decltype(v)>(v)); }
inline constexpr auto crend(const vector auto& v) noexcept { return detail::crend(std::forward<decltype(v)>(v)); }

template<req::number T>
using DynamicVector = Vector<T, 0>;

Vector(req::number auto&&... args)
    -> Vector< traits::common_type_t<std::remove_reference_t<decltype(args)>...>, sizeof...(args) >;

Vector(vector auto&& vec) 
    -> Vector<typename std::remove_reference_t<decltype(vec)>::value_type, std::remove_reference_t<decltype(vec)>::static_size>;

template<typename T, size_t N> Vector(const T(&)[N]) -> Vector<T, N>;
template<typename T, size_t N> Vector(std::array<T, N>) -> Vector<T, N>;
template<typename T, size_t N> Vector(std::span<T, N>) -> Vector<T, N>;
template<typename T, size_t N> Vector(std::span<const T, N>) -> Vector<T, N>;
template<typename T>           Vector(std::span<T>) -> DynamicVector<T>;

template<vector V1, vector V2> requires suitable_vector_cross_product_expression<V1, V2>
inline constexpr auto cprod(const V1&, const V2&) noexcept;

template<vector V1, vector V2> requires suitable_vector_expression<V1, V2>
inline constexpr auto sprod(const V1&, const V2&) noexcept;

template<vector V1, vector V2> requires vectors_same_value_type<V1, V2>
inline constexpr auto concat(V1& v1, V2& v2) noexcept;

template<typename T> using Vec2 = Vector<T, 2>;
template<typename T> using Vec3 = Vector<T, 3>;
template<typename T> using Vec4 = Vector<T, 4>;
template<size_t N> using IntVec = Vector<int, N>;
template<size_t N> using FloatVec = Vector<float, N>;
template<size_t N> using DoubleVec = Vector<double, N>;
using IntVec2 = IntVec<2>;
using IntVec3 = IntVec<3>;
using IntVec4 = IntVec<4>;
using FloatVec2 = FloatVec<2>;
using FloatVec3 = FloatVec<3>;
using FloatVec4 = FloatVec<4>;
using DoubleVec2 = DoubleVec<2>;
using DoubleVec3 = DoubleVec<3>;
using DoubleVec4 = DoubleVec<4>;

}


#include "linalg/vector.hxx"