#pragma once

#include <limits>
#include <algorithm>

namespace linalg::detail {

template<vector V>
class VectorView {
public:
    using value_type = typename V::value_type;
    inline constexpr static size_t static_size { 0 };
    inline constexpr static bool is_temporary { true };
private:
    std::conditional_t<V::is_temporary, V, V&> v;
    size_t m_begin, m_end;

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return v[m_begin + index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return v[m_begin + index]; }
public:
    constexpr VectorView(V& v, size_t begin): v{v}, m_begin{begin}, m_end{v.size()} {
        if(m_begin > m_end)
            throw std::runtime_error("VectorView: begin > end");
    }

    constexpr VectorView(V& v, size_t begin, size_t end): v{v}, m_begin{begin}, m_end{end} {
        if(m_begin > m_end) {
            throw std::runtime_error("VectorView: begin > end");
        }
        if(m_end > v.size() || m_begin >= v.size()) {
            throw std::out_of_range("VectorView: index mismatch (end or start > v.size())");
        }
    }

    inline constexpr size_t size() const {
        if(v.size() < m_end - m_begin) {
            throw std::runtime_error("VectorView: size mismatch (end or start > v.size())");
        }
        return m_end - m_begin;
    }

    inline constexpr decltype(auto) operator[](size_t index) const noexcept {
        return pick(index);
    }

    inline constexpr decltype(auto) operator[](size_t index) noexcept {
        return pick(index);
    }

    inline constexpr decltype(auto) at(size_t index) const {
        if(m_end > v.size() || m_begin >= v.size()) {
            throw std::out_of_range("VectorView: index mismatch (end or start > v.size())");
        }
        return v.at(m_begin + index);
    }

    inline constexpr decltype(auto) at(size_t index) {
        if(m_end > v.size() || m_begin >= v.size()) {
            throw std::out_of_range("VectorView: index mismatch (end or start > v.size())");
        }
        return v.at(m_begin + index);
    }

    inline constexpr decltype(auto) front() const noexcept { return *begin(); }
    inline constexpr decltype(auto) front() noexcept { return *begin(); }
    inline constexpr decltype(auto) back() const noexcept { return *rbegin(); }
    inline constexpr decltype(auto) back() noexcept { return *rbegin(); }

    inline constexpr auto begin() const noexcept { return v.begin() + m_begin; }
    inline constexpr auto rbegin() const noexcept { return v.rbegin() + (v.size() - m_end); }
    inline constexpr auto cbegin() const noexcept { return v.cbegin() + m_begin; }
    inline constexpr auto crbegin() const noexcept { return v.crbegin() + (v.size() - m_end); }
    inline constexpr auto end() const noexcept { return v.end() - (v.size() - m_end); }
    inline constexpr auto rend() const noexcept { return v.rend() - m_begin; }
    inline constexpr auto cend() const noexcept { return v.cend() - (v.size() - m_end); }
    inline constexpr auto crend() const noexcept { return v.crend() - m_begin; }

    inline constexpr auto begin() noexcept { return v.begin() + m_begin; }
    inline constexpr auto rbegin() noexcept { return v.rbegin() + (v.size() - m_end); }
    inline constexpr auto end() noexcept { return v.end() - (v.size() - m_end); }
    inline constexpr auto rend() noexcept { return v.rend() - m_begin; }

    inline constexpr bool empty() const noexcept requires(dynamic_vector<V> && !V::is_temporary) {
        return size() == 0;
    }

    inline constexpr void resize(size_t size) requires(dynamic_vector<V> && !V::is_temporary) {
        static_assert(vector<V>);
        if constexpr(dynamic_vector<V>) {
            if(size > this->size()) {
                v.resize(v.size() + size - this->size());
                std::shift_right(v.begin() + m_begin, v.end(), size - this->size());
                m_end += size - this->size();
            } else if(size < this->size()) {
                auto tmp_end = m_begin + (this->size() - size);
                v.erase(end()-1, end()-1 + this->size() - size);
                m_end = tmp_end;
            }
        }
    }

    inline constexpr void resize(size_t size, const value_type& val) requires(dynamic_vector<V> && !V::is_temporary) {
        if constexpr(dynamic_vector<V>) {
            if(size > this->size()) {
                v.resize(v.size() + size - this->size());
                std::shift_right(end(), v.end(), size - this->size());
                std::fill_n(end(), size - this->size(), val);
                m_end += size - this->size();
            } else if(size < this->size()) {
                auto tmp_end = m_begin + (this->size() - size);
                v.erase(end()-1, end()-1 + this->size() - size);
                m_end = tmp_end;
            }
        }
    }

    inline constexpr void push_back(value_type val) requires(dynamic_vector<V> && !V::is_temporary) {
        if constexpr(dynamic_vector<V>) {
            v.emplace(m_end++, val);
        }
    }

    inline constexpr void emplace(size_t pos, auto&&... args) requires(dynamic_vector<V> && !V::is_temporary) {
        if constexpr(dynamic_vector<V>) {
            v.emplace(m_begin + pos, std::forward<decltype(args)>(args)...);
            ++m_end;
        }
    }

    inline constexpr void emplace_back(auto&&... args) requires(dynamic_vector<V> && !V::is_temporary) {
        if constexpr(dynamic_vector<V>) {
            v.emplace(m_end++, std::forward<decltype(args)>(args)...);
        }
    }

    inline constexpr void pop_back() noexcept requires(dynamic_vector<V> && !V::is_temporary) {
        if constexpr(dynamic_vector<V>) {
            v.erase(end() - 1);
            m_end--;
        }
    }

    inline constexpr auto subvector(size_t begin = 0) const& noexcept {
        return detail::VectorView(*this, begin);
    }

    inline constexpr auto subvector(size_t begin, size_t end) const& noexcept {
        return detail::VectorView(*this, begin, end);
    }

    template<vector Vec> inline constexpr decltype(auto) operator+=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this + v;
        return *this; 
    }

    template<vector Vec> inline constexpr decltype(auto) operator-=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this - v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator*=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this * v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator/=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this / v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator%=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this % v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator&=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this & v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator|=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
        *this = *this | v;
        return *this;
    } 

    template<vector Vec> inline constexpr decltype(auto) operator^=(const Vec& v) requires suitable_vector_expression<VectorView, Vec> {
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

}