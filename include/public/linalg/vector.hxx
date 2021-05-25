#pragma once

#include <cassert>

#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>

#include "linalg/vector.hpp"
#include "linalg/operation.hpp"

namespace linalg {

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
    constexpr VectorExpression(const V1& v1, const V2& v2): v1{v1}, v2{v2} {}

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


    inline constexpr auto subvector(size_t begin = 0) const& noexcept {
        return detail::VectorView(*this, begin);
    }

    inline constexpr auto subvector(size_t begin, size_t end) const& noexcept {
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

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
};

template<vector V1, vector V2> requires suitable_vector_expression<V1, V2>
class VectorScalarProduct {
    std::conditional_t<V1::is_temporary, V1, const V1&> v1;
    std::conditional_t<V2::is_temporary, V2, const V2&> v2;
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
public:
    using value_type = traits::common_type_t<typename V1::value_type, typename V2::value_type>;
    
    constexpr VectorScalarProduct(const V1& v1, const V2& v2) noexcept: v1{v1}, v2{v2} {}
    constexpr ~VectorScalarProduct() {}
    
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

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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

    constexpr VectorConcatenation(V1& v1, V2& v2): v1{v1}, v2{v2} {}

    inline constexpr size_t size() const noexcept { return v1.size() + v2.size(); }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { return pick(index); }

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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
    constexpr UnaryVectorFunction(V& v): v{v} {}

    inline constexpr size_t size() const noexcept { return v.size(); }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("UnaryVectorFunction"); }

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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
    constexpr BinaryVectorFunction(const V1& v1, const V2& v2): v1{v1}, v2{v2} {}

    inline constexpr size_t size() const { if(v1.size() != v2.size()) throw std::runtime_error("BinaryVectorFunction: size mismatch"); return v1.size(); }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v1.size()) return pick(index); else throw std::out_of_range("BinaryVectorFunction"); }

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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
    constexpr VectorScalarFunction(const V& v1, Scalar s): v{v}, s{s} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) { return v.size(); }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("VectorScalarFunction"); }

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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
    constexpr ScalarVectorFunction(const V& v1, Scalar s): v{v}, s{s} {}

    inline constexpr size_t size() const noexcept(noexcept(v.size())) { return v.size(); }
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) const { if(index < v.size()) return pick(index); else throw std::out_of_range("ScalarVectorFunction"); }

    inline constexpr auto begin() const { return const_iterator(*this, 0); }
    inline constexpr auto cbegin() const { return const_iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(const_iterator(*this, 0)); }
    inline constexpr auto end() const { return const_iterator(*this, size()); }
    inline constexpr auto cend() const { return const_iterator(*this, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(const_iterator(*this, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(const_iterator(*this, size())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, size())); }
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


#define DECLARE_FUNCTION(NAME)                                                                                   \
namespace detail {                                                                                               \
    constexpr auto NAME = [](auto&& x) constexpr noexcept { return ::std::NAME(std::forward<decltype(x)>(x)); }; \
}                                                                                                                \
template<typename V> constexpr auto NAME(const V& v) noexcept {                                                  \
    return detail::UnaryVectorFunction<const V, decltype(detail::NAME)>(v);                                      \
}                                                                                                                \

#define DECLARE_FUNCTION_2(NAME)                                                                                                                         \
namespace detail {                                                                                                                                       \
    constexpr auto NAME = [](auto&& x, auto&& y) constexpr noexcept { return ::std::NAME(std::forward<decltype(x)>(x), std::forward<decltype(y)>(y)); }; \
}                                                                                                                                                        \
constexpr auto NAME(const vector auto& v1, const vector auto& v2) noexcept {                                                                             \
    return detail::BinaryVectorFunction<const std::decay_t<decltype(v1)>, const std::decay_t<decltype(v2)>, decltype(detail::NAME)>(v1, v2);                     \
}                                                                                                                                                        \
template<vector V, std::convertible_to<typename V::value_type> Scalar>                                                                                   \
constexpr auto NAME(const V& v, Scalar&& scalar) noexcept {                                                                                              \
    return detail::VectorScalarFunction<const std::decay_t<V>, std::decay_t<Scalar>, decltype(detail::NAME)>{ v, std::forward<Scalar>(scalar) };                 \
}                                                                                                                                                        \
template<vector V, std::convertible_to<typename V::value_type> Scalar>                                                                                   \
constexpr auto NAME(Scalar&& scalar, const V& v) noexcept {                                                                                              \
    return detail::ScalarVectorFunction<std::decay_t<Scalar>, const std::decay_t<V>, decltype(detail::NAME)>{ std::forward<Scalar>(scalar), v };                 \
}

// Basic operations
// absolute value of a floating point value
    DECLARE_FUNCTION(abs); 
    DECLARE_FUNCTION(fabs); 
// remainder of the floating point division operation
    DECLARE_FUNCTION_2(fmod);
// signed remainder of the division operation
    DECLARE_FUNCTION_2(remainder);
    DECLARE_FUNCTION_2(remainderf);
    DECLARE_FUNCTION_2(remainderl);
// signed remainder as well as the three last bits of the division operation
    // Unimplemented, requires the third argument to be non-const
//     DECLARE_FUNCTION_3_ARRAY(remquo);
//     DECLARE_FUNCTION_3_ARRAY(remquof);
//     DECLARE_FUNCTION_3_ARRAY(remquol);
// // fused multiply-add operation
//     DECLARE_FUNCTION_3(fma);
//     DECLARE_FUNCTION_3(fmaf);
//     DECLARE_FUNCTION_3(fmal);
// larger of two floating point values
    DECLARE_FUNCTION_2(fmax);
    DECLARE_FUNCTION_2(fmaxf);
    DECLARE_FUNCTION_2(fmaxl);
// smaller of two floating point values
    DECLARE_FUNCTION_2(fmin);
    DECLARE_FUNCTION_2(fminf);
    DECLARE_FUNCTION_2(fminl);
// positive difference of two floating point values
    DECLARE_FUNCTION_2(fdim);
    DECLARE_FUNCTION_2(fdimf);
    DECLARE_FUNCTION_2(fdiml);


#undef DECLARE_FUNCTION
#undef DECLARE_FUNCTION_2
// #undef DECLARE_FUNCTION_3
}