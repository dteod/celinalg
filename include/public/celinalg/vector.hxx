#pragma once

#include <cassert>

#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>

#include "celinalg/vector.hpp"
#include "celinalg/operation.hpp"

namespace celinalg {

namespace detail {

template<Operation op, vector V1, vector V2> requires suitable_vector_expression<V1, V2>
class VectorExpression {
public:
    inline constexpr static bool is_temporary { true }; 
    inline constexpr static size_t static_size { ((dynamic_vector<V1> || dynamic_vector<V2>) ? 0 : V1::static_size) };
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    using iterator = detail::linear_element_iterator<VectorExpression<op, V1, V2>>;
    constexpr VectorExpression(const V1& v1, const V2& v2) noexcept: v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept(static_vector<V1> && static_vector<V2> && noexcept(v1.size())) {
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

    inline constexpr auto begin() const noexcept { return iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<Operation op, vector Vec, req::number Scalar>
class VectorScalarExpression {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { Vec::static_size };
    using value_type = traits::common_type_t<typename Vec::value_type, Scalar>;
private:
    std::conditional_t<Vec::is_temporary, Vec, const Vec&> vec;
    Scalar s;

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    using iterator = detail::linear_element_iterator<VectorScalarExpression<op, Vec, Scalar>>;
    using const_iterator = detail::linear_element_iterator<const VectorScalarExpression<op, Vec, Scalar>>;

    constexpr VectorScalarExpression(const Vec& vec, const Scalar& s) noexcept : vec{vec}, s{s} {}

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

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<Operation op, req::number Scalar, vector Vec>
class ScalarVectorExpression {
    Scalar s;
    std::conditional_t<Vec::is_temporary, Vec, const Vec&> vec;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    inline constexpr static bool is_temporary { true }; 
    inline constexpr static size_t static_size { Vec::static_size };
    using value_type = traits::common_type_t<typename Vec::value_type, Scalar>;
    using iterator = detail::linear_element_iterator<ScalarVectorExpression<op, Scalar, Vec>>;
    using const_iterator = detail::linear_element_iterator<const VectorScalarExpression<op, Vec, Scalar>>;
    constexpr ScalarVectorExpression(const Scalar& s, const Vec& vec) noexcept: s{s}, vec{vec} {}

    inline constexpr size_t size() const noexcept {
        return vec.size();
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { 
        return expression_operator<op>::call(s, vec[index]);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        return expression_operator<op>::call(s, vec.at(index));
    }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, vector V2> requires suitable_vector_expression<V1, V2>
class VectorScalarProduct {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
public:
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
    
    constexpr VectorScalarProduct(const V1& v1, const V2& v2) noexcept: v1{v1}, v2{v2} {}
    
    inline constexpr operator value_type() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) {
        return get();
    }

    inline constexpr value_type get() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size()) {
                throw std::runtime_error("size mismatch");
            }
        }
        return std::inner_product(v1.begin(), v1.end(), v2.begin(), static_cast<value_type>(0));
    }
};

template<vector V1, vector V2> requires suitable_vector_cross_product_expression<V1, V2>
class VectorCrossProductExpression {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { 3 };
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    using iterator = detail::linear_element_iterator<VectorCrossProductExpression<V1, V2>>;
    using const_iterator = detail::linear_element_iterator<const VectorCrossProductExpression<V1, V2>>;

    constexpr VectorCrossProductExpression(const V1& v1, const V2& v2) noexcept : v1{v1}, v2{v2} {
        assert(v1.size() == v2.size() && v1.size() == 3);
    }

    inline constexpr size_t size() const noexcept { return v1.size() == v2.size() && v1.size() == 3 ? 3 : 0; }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        size_t idx0 = (index + 1)%3, idx1 = (2 + index)%3;
        value_type out = v1[idx0]*v2[idx1] - v1[idx1]*v2[idx0];
        return out;
    }

    inline constexpr decltype(auto) at(size_t index) const {
        if(index > 2)
            throw std::out_of_range();
        size_t idx0 = (index + 1)%3, idx1 = (2 + index)%3;
        value_type out = v1.at(idx0) * v2.at(idx1) - v1.at(idx1) * v2.at(idx0);
        return out;
    }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }
    
    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, vector V2> requires vectors_same_value_type<V1, V2>
class VectorConcatenation {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return index < v1.size() ? v1[index] : v2[index - v1.size()]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return index < v1.size() ? v1[index] : v2[index - v1.size()]; }
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size + V2::static_size };
    using value_type = typename V1::value_type;
    using iterator = detail::linear_element_iterator<VectorConcatenation<V1, V2>>;

    constexpr VectorConcatenation(V1& v1, V2& v2) noexcept: v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept { return v1.size() + v2.size(); }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { return pick(index); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V, std::invocable<typename V::value_type> F>
class UnaryVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, typename V::value_type>;
private:
    std::conditional_t<V::is_temporary, V, V&> v;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v[index]); }
public:
    using iterator = detail::linear_element_iterator<UnaryVectorFunction>;
    constexpr UnaryVectorFunction(V& v) noexcept: v{v} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) { return v.size(); }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("UnaryVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, vector V2, std::invocable<typename V1::value_type, typename V2::value_type> F> requires suitable_vector_expression<V1, V2>
class BinaryVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, typename V1::value_type, typename V2::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], v2[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], v2[index]); }
public:
    using iterator = detail::linear_element_iterator<BinaryVectorFunction>;
    constexpr BinaryVectorFunction(const V1& v1, const V2& v2) noexcept: v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) { 
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size()) 
                throw std::runtime_error("BinaryVectorFunction: size mismatch"); 
        }
        return v1.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("BinaryVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V, req::number Scalar, std::invocable<typename V::value_type, Scalar> F>
class VectorScalarFunction {
public:
    inline constexpr static bool is_temporary  { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, typename V::value_type, Scalar>;
private:
    std::conditional_t<V::is_temporary, V, V&> v;
    Scalar s;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v[index], s); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v[index], s); }
public:
    using iterator = detail::linear_element_iterator<VectorScalarFunction>;
    constexpr VectorScalarFunction(const V& v1, Scalar s) noexcept: v{v}, s{s} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) { return v.size(); }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("VectorScalarFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<req::number Scalar, vector V, std::invocable<Scalar, typename V::value_type> F>
class ScalarVectorFunction {
public:
    inline constexpr static bool is_temporary  { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, Scalar, typename V::value_type>;
private:
    Scalar s;
    std::conditional_t<V::is_temporary, V, V&> v;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v[index], s); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v[index], s); }
public:
    using iterator = detail::linear_element_iterator<ScalarVectorFunction>;
    constexpr ScalarVectorFunction(const V& v1, Scalar s) noexcept: v{v}, s{s} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) { return v.size(); }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("ScalarVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, vector V2, vector V3, std::invocable<typename V1::value_type, typename V2::value_type, typename V3::value_type> F> requires(suitable_vector_expression<V1, V2> && suitable_vector_expression<V1, V3> && suitable_vector_expression<V2, V3>)
class TernaryVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> || dynamic_vector<V3> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, typename V1::value_type, typename V2::value_type, typename V3::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    std::conditional_t<V3::is_temporary, V3, V3&> v3;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], v2[index], v3[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], v2[index], v3[index]); }
public:
    using iterator = detail::linear_element_iterator<TernaryVectorFunction>;
    constexpr TernaryVectorFunction(V1& v1, V2& v2, V3& v3) noexcept: v1{v1}, v2{v2}, v3{v3} {}

    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2> || dynamic_vector<V3>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2> || dynamic_vector<V3>) {
            if(v1.size() != v2.size() || v1.size() != v3.size())
                throw std::runtime_error("TernaryVectorFunction: size mismatch");
        }
        return v1.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("TernaryVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, vector V2, typename Scalar, std::invocable<typename V1::value_type, typename V2::value_type, Scalar> F> requires(suitable_vector_expression<V1, V2> && !vector<Scalar>)
class VectorVectorScalarFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, typename V1::value_type, typename V2::value_type, Scalar>;
private:
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    Scalar s;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], v2[index], s); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], v2[index], s); }
public:
    using iterator = detail::linear_element_iterator<VectorVectorScalarFunction>;
    constexpr VectorVectorScalarFunction(V1& v1, V2& v2, Scalar s) noexcept: v1{v1}, v2{v2}, s{s} {}

    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size())
                throw std::runtime_error("VectorVectorScalarFunction: size mismatch");
        }
        return v1.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("VectorVectorScalarFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V1, typename Scalar, vector V2, std::invocable<typename V1::value_type, Scalar, typename V2::value_type> F> requires(suitable_vector_expression<V1, V2> && !vector<Scalar>)
class VectorScalarVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, typename V1::value_type, Scalar, typename V2::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    Scalar s;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], s, v2[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], s, v2[index]); }
public:
    using iterator = detail::linear_element_iterator<VectorScalarVectorFunction>;
    constexpr VectorScalarVectorFunction(V1& v1, Scalar s, V2& v2) noexcept: v1{v1}, s{s}, v2{v2} {}

    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size())
                throw std::runtime_error("VectorScalarVectorFunction: size mismatch");
        }
        return v1.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("VectorScalarVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<typename Scalar, vector V1, vector V2, std::invocable<Scalar, typename V1::value_type, typename V2::value_type> F> requires(suitable_vector_expression<V1, V2> && !vector<Scalar>)
class ScalarVectorVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, Scalar, typename V1::value_type, typename V2::value_type>;
private:
    Scalar s;
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(s, v1[index], v2[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(s, v1[index], v2[index]); }
public:
    using iterator = detail::linear_element_iterator<ScalarVectorVectorFunction>;
    constexpr ScalarVectorVectorFunction(Scalar s, V1& v1, V2& v2) noexcept: s{s}, v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2>) {
            if(v1.size() != v2.size())
                throw std::runtime_error("ScalarVectorVectorFunction: size mismatch");
        }
        return v1.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("ScalarVectorVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<typename Scalar1, typename Scalar2, vector V, std::invocable<Scalar1, Scalar2, typename V::value_type> F> requires(!(vector<Scalar1> || vector<Scalar2>) )
class ScalarScalarVectorFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, Scalar1, Scalar2, typename V::value_type>;
private:
    Scalar1 s1;
    Scalar2 s2;
    std::conditional_t<V::is_temporary, V, V&> v;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(s1, s2, v[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(s1, s2, v[index]); }
public:
    using iterator = detail::linear_element_iterator<ScalarScalarVectorFunction>;
    constexpr ScalarScalarVectorFunction(Scalar1 s1, Scalar2 s2, V& v) noexcept: s1{s1}, s2{s2}, v{v} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) {
        return v.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("ScalarScalarVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<typename Scalar1, vector V, typename Scalar2, std::invocable<Scalar1, typename V::value_type, Scalar2> F> requires(!(vector<Scalar1> || vector<Scalar2>) )
class ScalarVectorScalarFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, Scalar1, typename V::value_type, Scalar2>;
private:
    Scalar1 s1;
    std::conditional_t<V::is_temporary, V, V&> v;
    Scalar2 s2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(s1, v[index], s2); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(s1, v[index], s2); }
public:
    using iterator = detail::linear_element_iterator<ScalarVectorScalarFunction>;
    constexpr ScalarVectorScalarFunction(Scalar1 s1, V& v, Scalar2 s2) noexcept: s1{s1}, v{v}, s2{s2} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) {
        return v.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("ScalarVectorScalarFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

template<vector V, typename Scalar1, typename Scalar2, std::invocable<typename V::value_type, Scalar1, Scalar2> F> requires(!(vector<Scalar1> || vector<Scalar2>) )
class VectorScalarScalarFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, typename V::value_type, Scalar1, Scalar2>;
private:
    std::conditional_t<V::is_temporary, V, V&> v;
    Scalar1 s1;
    Scalar2 s2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v[index], s1, s2); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v[index], s1, s2); }
public:
    using iterator = detail::linear_element_iterator<VectorScalarScalarFunction>;
    constexpr VectorScalarScalarFunction(V& v, Scalar1 s1, Scalar2 s2) noexcept: v{v}, s1{s1}, s2{s2} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) {
        return v.size(); 
    }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("VectorScalarScalarFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};


template<vector V, typename Container, std::invocable<typename V::value_type, typename Container::value_type> F>
class VectorContainerFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { V::static_size };
    using value_type = std::invoke_result_t<F, typename V::value_type, typename Container::value_type>;
private:
    std::conditional_t<V::is_temporary, V, V&> v1;
    Container& v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], v2[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], v2[index]); }

    template<typename T> struct is_container_array: std::false_type {};
    template<typename T, size_t N> struct is_container_array<std::array<T, N>>: std::true_type {};
    template<typename T> inline constexpr static bool is_container_array_v { is_container_array<T>::value };
public:
    constexpr VectorContainerFunction(const V& v1, Container& v2) noexcept: v1{v1}, v2{v2} {}
    
    inline constexpr size_t size() const noexcept(!dynamic_vector<V> && is_container_array_v<Container>) {
        if constexpr(dynamic_vector<V> || !is_container_array_v<Container>) {
            if(v1.size() != v2.size())
                throw std::runtime_error("VectorContainerFunction: size mismatch");
        }
        return v1.size(); 
    }
    
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("TernaryVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};


template<vector V1, vector V2, typename Container, std::invocable<typename V1::value_type, typename V2::value_type, typename Container::value_type> F> requires suitable_vector_expression<V1, V2>
class VectorVectorContainerFunction {
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { dynamic_vector<V1> || dynamic_vector<V2> ? 0 : V1::static_size };
    using value_type = std::invoke_result_t<F, typename V1::value_type, typename V2::value_type, typename Container::value_type>;
private:
    std::conditional_t<V1::is_temporary, V1, V1&> v1;
    std::conditional_t<V2::is_temporary, V2, V2&> v2;
    Container& v3;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return F{}(v1[index], v2[index], v3[index]); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return F{}(v1[index], v2[index], v3[index]); }

    template<typename T> struct is_container_array: std::false_type {};
    template<typename T, size_t N> struct is_container_array<std::array<T, N>>: std::true_type {};
    template<typename T> inline constexpr static bool is_container_array_v { is_container_array<T>::value };
public:
    constexpr VectorVectorContainerFunction(const V1& v1, const V2& v2, Container& v3) noexcept: v1{v1}, v2{v2}, v3{v3} {}
    
    inline constexpr size_t size() const noexcept(!(dynamic_vector<V1> || dynamic_vector<V2> || !is_container_array_v<Container>)) {
        if constexpr(dynamic_vector<V1> || dynamic_vector<V2> || !is_container_array_v<Container>) {
            if(v1.size() != v2.size() || v1.size() != v3.size())
                throw std::runtime_error("VectorVectorContainerFunction: size mismatch");
        }
        return v1.size(); 
    }
    
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("TernaryVectorFunction"); }

    inline constexpr auto begin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const noexcept { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto cend() const noexcept { return const_iterator(*this, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() noexcept { return iterator(*this, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() noexcept { return iterator(*this, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(*this, size())); }

    [[nodiscard]] inline constexpr auto subvector(size_t begin = 0) const noexcept {
        return detail::VectorView(*this, begin);
    }

    [[nodiscard]] inline constexpr auto subvector(size_t begin, size_t end) const noexcept {
        return detail::VectorView(*this, begin, end);
    }
};

}

template<vector V1, vector V2> inline auto operator+ (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::ADDITION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator- (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::SUBTRACTION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator* (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::MULTIPLICATION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator/ (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::DIVISION, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator% (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::MODULO, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator&&(const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::AND, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator||(const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::OR, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator& (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::BITWISE_AND, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator| (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::BITWISE_OR, V1, V2>(v1, v2); }
template<vector V1, vector V2> inline auto operator^ (const V1& v1, const V2& v2) { return detail::VectorExpression<detail::Operation::BITWISE_XOR, V1, V2>(v1, v2); }

template<vector V, req::number S> inline auto operator+ (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::ADDITION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator- (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::SUBTRACTION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator* (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::MULTIPLICATION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator/ (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::DIVISION, V, S>(v, s); }
template<vector V, req::number S> inline auto operator% (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::MODULO, V, S>(v, s); }
template<vector V, req::number S> inline auto operator&&(const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::AND, V, S>(v, s); }
template<vector V, req::number S> inline auto operator||(const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::OR, V, S>(v, s); }
template<vector V, req::number S> inline auto operator& (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::BITWISE_AND, V, S>(v, s); }
template<vector V, req::number S> inline auto operator| (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::BITWISE_OR, V, S>(v, s); }
template<vector V, req::number S> inline auto operator^ (const V& v, S s) { return detail::VectorScalarExpression<detail::Operation::BITWISE_XOR, V, S>(v, s); }

template<req::number S, vector V> inline auto operator+ (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::ADDITION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator- (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::SUBTRACTION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator* (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::MULTIPLICATION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator/ (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::DIVISION, S, V>(s, v); }
template<req::number S, vector V> inline auto operator% (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::MODULO, S, V>(s, v); }
template<req::number S, vector V> inline auto operator&&(S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::AND, S, V>(s, v); }
template<req::number S, vector V> inline auto operator||(S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::OR, S, V>(s, v); }
template<req::number S, vector V> inline auto operator& (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::BITWISE_AND, S, V>(s, v); }
template<req::number S, vector V> inline auto operator| (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::BITWISE_OR, S, V>(s, v); }
template<req::number S, vector V> inline auto operator^ (S s, const V& v) { return detail::ScalarVectorExpression<detail::Operation::BITWISE_XOR, S, V>(s, v); }

template<vector V1, vector V2> requires suitable_vector_cross_product_expression<V1, V2>
inline constexpr auto cprod(const V1& v1, const V2& v2) noexcept { return detail::VectorCrossProductExpression(v1, v2); }

template<vector V1, vector V2> requires suitable_vector_expression<V1, V2>
inline constexpr auto sprod(const V1& v1, const V2& v2) noexcept { return detail::VectorScalarProduct(v1, v2); }

template<vector V1, vector V2> requires vectors_same_value_type<V1, V2>
inline constexpr auto concat(V1& v1, V2& v2) noexcept { return detail::VectorConcatenation(v1, v2); }


#define celinalg_DECLARE_FUNCTION(NAME)                                                                                           \
namespace detail {                                                                                                       \
    inline constexpr auto NAME = [](auto&& x) constexpr noexcept { return math::NAME(std::forward<decltype(x)>(x)); };  \
}                                                                                                                        \
template<vector V> inline constexpr auto NAME(const V& v) noexcept {                                                     \
    return detail::UnaryVectorFunction<const V, decltype(detail::NAME)>(v);                                              \
}

#define celinalg_DECLARE_FUNCTION_2(NAME)                                                                                                                                                                                                                    \
namespace detail {                                                                                                      \
    inline constexpr auto NAME = [](auto&& x, auto&& y) constexpr noexcept {                                            \
        return math::NAME(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y));                                 \
    };                                                                                                                  \
}                                                                                                                       \
template<vector V1, vector V2>                                                                                          \
inline constexpr auto NAME(const V1& v1, const V2& v2) noexcept {                                                       \
    return detail::BinaryVectorFunction<const V1, const V2, decltype(detail::NAME)>(v1, v2);                            \
}                                                                                                                       \
template<vector V, std::convertible_to<typename V::value_type> Scalar>                                                  \
inline constexpr auto NAME(const V& v, Scalar scalar) noexcept {                                                        \
    return detail::VectorScalarFunction<const V, Scalar, decltype(detail::NAME)>{ v, scalar };                          \
}                                                                                                                       \
template<vector V, std::convertible_to<typename V::value_type> Scalar>                                                  \
inline constexpr auto NAME(Scalar scalar, const V& v) noexcept {                                                        \
    return detail::ScalarVectorFunction<Scalar, const V, decltype(detail::NAME)>{ scalar, v };                          \
}

#define celinalg_DECLARE_FUNCTION_3(NAME)                                                                                             \
namespace detail {                                                                                                                  \
    inline constexpr auto NAME = [](auto&& x, auto&& y, auto&& z) constexpr noexcept {                                              \
        return math::NAME(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y), std::forward<decltype(z)>(z));               \
    };                                                                                                                              \
}                                                                                                                                   \
template<vector V1, vector V2, vector V3> inline constexpr auto NAME(const V1& v1, const V2& v2, const V3& v3) {                    \
    return detail::TernaryVectorFunction<const V1, const V2, const V3, decltype(detail::NAME)>(v1, v2, v3);                         \
}                                                                                                                                   \
template<vector V1, vector V2, typename Scalar> requires(!vector<Scalar>) inline constexpr auto NAME(const V1& v1, const V2& v2, Scalar s) {                  \
    return detail::VectorVectorScalarFunction<const V1, const V2, Scalar, decltype(detail::NAME)>(v1, v2, s);                       \
}                                                                                                                                   \
template<vector V1, typename Scalar, vector V2> requires(!vector<Scalar>) inline constexpr auto NAME(const V1& v1, Scalar s, const V2& v2) {                  \
    return detail::VectorScalarVectorFunction<const V1, Scalar, const V2, decltype(detail::NAME)>(v1, s, v2);                       \
}                                                                                                                                   \
template<typename Scalar, vector V1, vector V2> requires(!vector<Scalar>) inline constexpr auto NAME(Scalar s, const V1& v1, const V2& v2) {                  \
    return detail::ScalarVectorVectorFunction<Scalar, const V1, const V2, decltype(detail::NAME)>(s, v1, v2);                       \
}                                                                                                                                   \
template<vector V, typename Scalar1, typename Scalar2> requires(!(vector<Scalar1> || vector<Scalar2>)) inline constexpr auto NAME(const V& v, Scalar1 s1, Scalar2 s2) {             \
    return detail::VectorScalarScalarFunction<const V, Scalar1, Scalar2, decltype(detail::NAME)>(v, s1, s2);                        \
}                                                                                                                                   \
template<typename Scalar1, vector V, typename Scalar2> requires(!(vector<Scalar1> || vector<Scalar2>)) inline constexpr auto NAME(Scalar1 s1, const V& v, Scalar2 s2) {             \
    return detail::ScalarVectorScalarFunction<Scalar1, const V, Scalar2, decltype(detail::NAME)>(s1, v, s2);                        \
}                                                                                                                                   \
template<typename Scalar1, typename Scalar2, vector V> requires(!(vector<Scalar1> || vector<Scalar2>)) inline constexpr auto NAME(Scalar1 s1, Scalar2 s2, const V& v) {             \
    return detail::ScalarScalarVectorFunction<Scalar1, Scalar2, const V, decltype(detail::NAME)>(s1, s2, v);                        \
}

#define celinalg_DECLARE_FUNCTION_2_CONTAINER(NAME)                                                       \
namespace detail {                                                                                      \
    inline constexpr auto NAME = [](auto&& x, auto&& y) constexpr noexcept {                            \
        return math::NAME(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y));                 \
    };                                                                                                  \
}                                                                                                       \
template<vector V, typename T, size_t N> requires(                                                      \
    (dynamic_vector<V> || (static_vector<V> && V::static_size == N))                                    \
)                                                                                                       \
inline constexpr auto NAME(const V& v1, std::array<T, N>& v2) noexcept {                                \
    return detail::VectorContainerFunction<const V, std::array<T, N>, decltype(detail::NAME)>(v1, v2);  \
}                                                                                                       \
template<vector V, typename T>                                                                          \
inline constexpr auto NAME(const V& v1, std::vector<T>& v2) noexcept {                                  \
    return detail::VectorContainerFunction<const V, std::vector<T>, decltype(detail::NAME)>(v1, v2);    \
}

#define celinalg_DECLARE_FUNCTION_3_CONTAINER(NAME)                                                                           \
namespace detail {                                                                                                          \
    inline constexpr auto NAME = [](auto&& x, auto&& y, auto&& z) constexpr noexcept {                                      \
        return math::NAME(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y), std::forward<decltype(z)>(z));       \
    };                                                                                                                      \
}                                                                                                                           \
template<vector V1, vector V2, typename T, size_t N> requires(                                                              \
    suitable_vector_expression<V1, V2> &&                                                                                   \
    (dynamic_vector<V1> || (static_vector<V1> && V1::static_size == N)) &&                                                  \
    (dynamic_vector<V2> || (static_vector<V2> && V2::static_size == N))                                                     \
)                                                                                                                           \
inline constexpr auto NAME(const V1& v1, const V2& v2, std::array<T, N>& v3) noexcept {                                     \
    return detail::VectorVectorContainerFunction<const V1, const V2, std::array<T, N>, decltype(detail::NAME)>(v1, v2, v3); \
}                                                                                                                           \
template<vector V1, vector V2, typename T> requires suitable_vector_expression<V1, V2>                                      \
inline constexpr auto NAME(const V1& v1, const V2& v2, std::vector<T>& v3) noexcept {                                       \
    return detail::VectorVectorContainerFunction<const V1, const V2, std::vector<T>, decltype(detail::NAME)>(v1, v2, v3);   \
}

celinalg_DECLARE_FUNCTION(abs); 
celinalg_DECLARE_FUNCTION(fabs); 
celinalg_DECLARE_FUNCTION_2(fmod);
celinalg_DECLARE_FUNCTION_2(remainder);
celinalg_DECLARE_FUNCTION_2(remainderf);
celinalg_DECLARE_FUNCTION_2(remainderl);
celinalg_DECLARE_FUNCTION_3_CONTAINER(remquo);
celinalg_DECLARE_FUNCTION_3_CONTAINER(remquof);
celinalg_DECLARE_FUNCTION_3_CONTAINER(remquol);
celinalg_DECLARE_FUNCTION_3(fma);
celinalg_DECLARE_FUNCTION_3(fmaf);
celinalg_DECLARE_FUNCTION_3(fmal);
celinalg_DECLARE_FUNCTION_2(fmax);
celinalg_DECLARE_FUNCTION_2(fmaxf);
celinalg_DECLARE_FUNCTION_2(fmaxl);
celinalg_DECLARE_FUNCTION_2(fmin);
celinalg_DECLARE_FUNCTION_2(fminf);
celinalg_DECLARE_FUNCTION_2(fminl);
celinalg_DECLARE_FUNCTION_2(fdim);
celinalg_DECLARE_FUNCTION_2(fdimf);
celinalg_DECLARE_FUNCTION_2(fdiml);
celinalg_DECLARE_FUNCTION_3(lerp);
celinalg_DECLARE_FUNCTION(exp);
celinalg_DECLARE_FUNCTION(exp2);
celinalg_DECLARE_FUNCTION(exp2f);
celinalg_DECLARE_FUNCTION(exp2l);
celinalg_DECLARE_FUNCTION(expm1);
celinalg_DECLARE_FUNCTION(expm1f);
celinalg_DECLARE_FUNCTION(expm1l);
celinalg_DECLARE_FUNCTION(log);
celinalg_DECLARE_FUNCTION(log10);
celinalg_DECLARE_FUNCTION(log1p);
celinalg_DECLARE_FUNCTION(log1pf);
celinalg_DECLARE_FUNCTION(log1pl);
celinalg_DECLARE_FUNCTION_2(pow);
celinalg_DECLARE_FUNCTION(sqrt);
celinalg_DECLARE_FUNCTION(cbrt);
celinalg_DECLARE_FUNCTION(cbrtf);
celinalg_DECLARE_FUNCTION(cbrtl);
celinalg_DECLARE_FUNCTION_2(hypot);
celinalg_DECLARE_FUNCTION_2(hypotf);
celinalg_DECLARE_FUNCTION_2(hypotl);
celinalg_DECLARE_FUNCTION(sin);
celinalg_DECLARE_FUNCTION(cos);
celinalg_DECLARE_FUNCTION(tan);
celinalg_DECLARE_FUNCTION(asin);
celinalg_DECLARE_FUNCTION(acos);
celinalg_DECLARE_FUNCTION(atan);
celinalg_DECLARE_FUNCTION_2(atan2);
celinalg_DECLARE_FUNCTION(sinh);
celinalg_DECLARE_FUNCTION(cosh);
celinalg_DECLARE_FUNCTION(tanh);
celinalg_DECLARE_FUNCTION(asinh);
celinalg_DECLARE_FUNCTION(asinhf);
celinalg_DECLARE_FUNCTION(asinhl);
celinalg_DECLARE_FUNCTION(acosh);
celinalg_DECLARE_FUNCTION(acoshf);
celinalg_DECLARE_FUNCTION(acoshl);
celinalg_DECLARE_FUNCTION(atanh);
celinalg_DECLARE_FUNCTION(atanhf);
celinalg_DECLARE_FUNCTION(atanhl);
celinalg_DECLARE_FUNCTION(erf);
celinalg_DECLARE_FUNCTION(erff);
celinalg_DECLARE_FUNCTION(erfl);
celinalg_DECLARE_FUNCTION(erfc);
celinalg_DECLARE_FUNCTION(erfcf);
celinalg_DECLARE_FUNCTION(erfcl);
celinalg_DECLARE_FUNCTION(tgamma);
celinalg_DECLARE_FUNCTION(tgammaf);
celinalg_DECLARE_FUNCTION(tgammal);
celinalg_DECLARE_FUNCTION(lgamma);
celinalg_DECLARE_FUNCTION(lgammaf);
celinalg_DECLARE_FUNCTION(lgammal);
celinalg_DECLARE_FUNCTION(ceil);
celinalg_DECLARE_FUNCTION(floor);
celinalg_DECLARE_FUNCTION(trunc);
celinalg_DECLARE_FUNCTION(truncf);
celinalg_DECLARE_FUNCTION(truncl);
celinalg_DECLARE_FUNCTION(round);
celinalg_DECLARE_FUNCTION(roundf);
celinalg_DECLARE_FUNCTION(roundl);
celinalg_DECLARE_FUNCTION(lround);
celinalg_DECLARE_FUNCTION(lroundf);
celinalg_DECLARE_FUNCTION(lroundl);
celinalg_DECLARE_FUNCTION(llround);
celinalg_DECLARE_FUNCTION(llroundf);
celinalg_DECLARE_FUNCTION(llroundl);
celinalg_DECLARE_FUNCTION(nearbyint);
celinalg_DECLARE_FUNCTION(nearbyintf);
celinalg_DECLARE_FUNCTION(nearbyintl);
celinalg_DECLARE_FUNCTION(rint);
celinalg_DECLARE_FUNCTION(rintf);
celinalg_DECLARE_FUNCTION(rintl);
celinalg_DECLARE_FUNCTION(lrint);
celinalg_DECLARE_FUNCTION(lrintf);
celinalg_DECLARE_FUNCTION(lrintl);
celinalg_DECLARE_FUNCTION(llrint);
celinalg_DECLARE_FUNCTION(llrintf);
celinalg_DECLARE_FUNCTION(llrintl);
celinalg_DECLARE_FUNCTION_2_CONTAINER(frexp);
celinalg_DECLARE_FUNCTION_2_CONTAINER(ldexp);
celinalg_DECLARE_FUNCTION_2_CONTAINER(modf);
celinalg_DECLARE_FUNCTION_2(scalbn);
celinalg_DECLARE_FUNCTION_2(scalbnf);
celinalg_DECLARE_FUNCTION_2(scalbnl);
celinalg_DECLARE_FUNCTION_2(scalbln);
celinalg_DECLARE_FUNCTION_2(scalblnf);
celinalg_DECLARE_FUNCTION_2(scalblnl);
celinalg_DECLARE_FUNCTION(ilogb);
celinalg_DECLARE_FUNCTION(ilogbf);
celinalg_DECLARE_FUNCTION(ilogbl);
celinalg_DECLARE_FUNCTION(logb);
celinalg_DECLARE_FUNCTION(logbf);
celinalg_DECLARE_FUNCTION(logbl);
celinalg_DECLARE_FUNCTION_2(nextafter);
celinalg_DECLARE_FUNCTION_2(nextafterf);
celinalg_DECLARE_FUNCTION_2(nextafterl);
celinalg_DECLARE_FUNCTION_2(nexttoward);
celinalg_DECLARE_FUNCTION_2(nexttowardf);
celinalg_DECLARE_FUNCTION_2(nexttowardl);
celinalg_DECLARE_FUNCTION_2(copysign);
celinalg_DECLARE_FUNCTION_2(copysignf);
celinalg_DECLARE_FUNCTION_2(copysignl);
celinalg_DECLARE_FUNCTION(fpclassify);
celinalg_DECLARE_FUNCTION(isfinite);
celinalg_DECLARE_FUNCTION(isinf);
celinalg_DECLARE_FUNCTION(isnan);
celinalg_DECLARE_FUNCTION(isnormal);
celinalg_DECLARE_FUNCTION(signbit);
celinalg_DECLARE_FUNCTION_2(isgreater);
celinalg_DECLARE_FUNCTION_2(isgreaterequal);
celinalg_DECLARE_FUNCTION_2(isless);
celinalg_DECLARE_FUNCTION_2(islessequal);
celinalg_DECLARE_FUNCTION_2(islessgreater);
celinalg_DECLARE_FUNCTION_2(isunordered);
celinalg_DECLARE_FUNCTION_3(assoc_laguerre);
celinalg_DECLARE_FUNCTION_3(assoc_laguerref);
celinalg_DECLARE_FUNCTION_3(assoc_laguerrel);
celinalg_DECLARE_FUNCTION_3(assoc_legendre);
celinalg_DECLARE_FUNCTION_3(assoc_legendref);
celinalg_DECLARE_FUNCTION_3(assoc_legendrel);
celinalg_DECLARE_FUNCTION_2(beta);
celinalg_DECLARE_FUNCTION_2(betaf);
celinalg_DECLARE_FUNCTION_2(betal);
celinalg_DECLARE_FUNCTION(comp_ellint_1);
celinalg_DECLARE_FUNCTION(comp_ellint_1f);
celinalg_DECLARE_FUNCTION(comp_ellint_1l);
celinalg_DECLARE_FUNCTION(comp_ellint_2);
celinalg_DECLARE_FUNCTION(comp_ellint_2f);
celinalg_DECLARE_FUNCTION(comp_ellint_2l);
celinalg_DECLARE_FUNCTION_2(comp_ellint_3);
celinalg_DECLARE_FUNCTION_2(comp_ellint_3f);
celinalg_DECLARE_FUNCTION_2(comp_ellint_3l);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_i);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_if);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_il);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_j);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_jf);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_jl);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_k);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_kf);
celinalg_DECLARE_FUNCTION_2(cyl_bessel_kl);
celinalg_DECLARE_FUNCTION_2(cyl_neumann);
celinalg_DECLARE_FUNCTION_2(cyl_neumannf);
celinalg_DECLARE_FUNCTION_2(cyl_neumannl);
celinalg_DECLARE_FUNCTION_2(ellint_1);
celinalg_DECLARE_FUNCTION_2(ellint_1f);
celinalg_DECLARE_FUNCTION_2(ellint_1l);
celinalg_DECLARE_FUNCTION_2(ellint_2);
celinalg_DECLARE_FUNCTION_2(ellint_2f);
celinalg_DECLARE_FUNCTION_2(ellint_2l);
celinalg_DECLARE_FUNCTION_3(ellint_3);
celinalg_DECLARE_FUNCTION_3(ellint_3f);
celinalg_DECLARE_FUNCTION_3(ellint_3l);
celinalg_DECLARE_FUNCTION(expint);
celinalg_DECLARE_FUNCTION(expintf);
celinalg_DECLARE_FUNCTION(expintl);
celinalg_DECLARE_FUNCTION_2(hermite);
celinalg_DECLARE_FUNCTION_2(hermitef);
celinalg_DECLARE_FUNCTION_2(hermitel);
celinalg_DECLARE_FUNCTION_2(legendre);
celinalg_DECLARE_FUNCTION_2(legendref);
celinalg_DECLARE_FUNCTION_2(legendrel);
celinalg_DECLARE_FUNCTION_2(laguerre);
celinalg_DECLARE_FUNCTION_2(laguerref);
celinalg_DECLARE_FUNCTION_2(laguerrel);
celinalg_DECLARE_FUNCTION(riemann_zeta);
celinalg_DECLARE_FUNCTION(riemann_zetaf);
celinalg_DECLARE_FUNCTION(riemann_zetal);
celinalg_DECLARE_FUNCTION_2(sph_bessel);
celinalg_DECLARE_FUNCTION_2(sph_besself);
celinalg_DECLARE_FUNCTION_2(sph_bessell);
celinalg_DECLARE_FUNCTION_3(sph_legendre);
celinalg_DECLARE_FUNCTION_3(sph_legendref);
celinalg_DECLARE_FUNCTION_3(sph_legendrel);
celinalg_DECLARE_FUNCTION_2(sph_neumann);
celinalg_DECLARE_FUNCTION_2(sph_neumannf);
celinalg_DECLARE_FUNCTION_2(sph_neumannl);

#undef celinalg_DECLARE_FUNCTION
#undef celinalg_DECLARE_FUNCTION_2
#undef celinalg_DECLARE_FUNCTION_3
#undef celinalg_DECLARE_FUNCTION_3_CONTAINER
}