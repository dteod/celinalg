#pragma once

#include "linalg/forward.hpp"
#include "linalg/vector.hxx"

namespace linalg {

enum class MatrixDimension {
    BY_ROWS,
    BY_COLS
};

namespace detail {

template<matrix M, MatrixDimension Dim>
class MatrixDimensionView {
    M& m;
    size_t m_index;
    inline constexpr size_t forward_size() const noexcept {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return m.rows();
        } else {
            return m.cols();
        }
    }

    inline constexpr size_t transversal_size() const noexcept {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return m.cols();
        } else {
            return m.rows();
        }
    }
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { M::static_cols };
    using value_type = typename M::value_type;
    
    class iterator {
        inline constexpr size_t pick(ssize_t index = 0) const noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return m_index*m.rows() + (index + m_subindex);
            } else {
                return m_index + (index + m_subindex)*m.cols();
            }
        }

        std::conditional_t<M::is_temporary, M, M&> m;
        size_t m_index;
        size_t m_subindex;
        // template<matrix Mat, MatrixDimension _Dim> friend constexpr typename MatrixDimensionViewFactory<Mat, _Dim>::iterator operator+(ssize_t, typename MatrixDimensionViewFactory<Mat, _Dim>::iterator&);
        // template<matrix Mat, MatrixDimension _Dim> friend constexpr const typename MatrixDimensionViewFactory<Mat, _Dim>::iterator operator+(ssize_t, const typename MatrixDimensionViewFactory<Mat, _Dim>::iterator&);
    public:
        constexpr iterator(M& m, size_t index, size_t subindex) noexcept: m{m}, m_index{index}, m_subindex{index} {}

        inline constexpr decltype(auto) operator*() const noexcept { return m[pick(0)]; }
        inline constexpr decltype(auto) operator[](ssize_t index) const { return m[pick(index)]; }

        inline constexpr auto& operator++() noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                m_subindex++;
            } else {
                m_index++; 
            }
            return *this; 
        }
        
        inline constexpr auto operator++(int) noexcept { 
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return iterator(m, m_index, m_subindex++);
            } else {
                return iterator(m, m_index++, m_subindex); 
            }
        }
        
        inline constexpr auto& operator--() noexcept { 
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                m_subindex--;
            } else {
                m_index--; 
            }
            return *this; 
        }

        inline constexpr auto operator--(int) noexcept { 
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return iterator(m, m_index, m_subindex--);
            } else {
                return iterator(m, m_index--, m_subindex); 
            }
        }

        inline constexpr auto operator+(ssize_t index) const noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return iterator(m, m_index, m_subindex + index);
            } else {
                return iterator(m, m_index + index, m_subindex);
            }
        }
        
        inline constexpr auto operator-(ssize_t index) const noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return iterator(m, m_index, m_subindex - index);
            } else {
                return iterator(m, m_index - index, m_subindex);
            }
        }

        inline constexpr auto& operator+=(ssize_t index) noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                m_subindex += index;
            } else {
                m_index += index;
            }
            return *this;
        }

        inline constexpr auto& operator-=(ssize_t index) noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                m_subindex -= index;
            } else {
                m_index -= index;
            }
            return *this;
        }


        inline constexpr auto operator-(iterator it) const noexcept { 
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return m_subindex - it.m_subindex;
            } else {
                return m_index - it.m_index;
            }
        }

        inline constexpr bool operator==(iterator it) const noexcept { return (&m == &it.m) && (m_index == it.m_index && m_subindex == it.m_subindex);}
        inline constexpr bool operator!=(iterator it) const noexcept { return (&m != &it.m) || !(m_index == it.m_index && m_subindex == it.m_subindex);}

        inline constexpr std::partial_ordering operator<=>(iterator it) const noexcept {
            if(&m != &it.m)
                return std::partial_ordering::unordered;
            if(m_index < it.m_index) {
                return std::partial_ordering::less;
            } else if(m_index > it.m_index) {
                return std::partial_ordering::greater;
            } else {
                return m_subindex <=> it.m_subindex;
            }
        }

        using difference_type = ptrdiff_t;
        using value_type = typename std::decay_t<M>::value_type;
        // using pointer = value_type*;
        using reference = decltype(std::declval<iterator>()[std::declval<size_t>()]);
        using iterator_category = std::random_access_iterator_tag;
    };
    inline constexpr MatrixDimensionView(M& m, size_t index) noexcept: m{m}, m_index{index} {}

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return m[index + forward_size()*m_index]; }
    inline constexpr decltype(auto) operator[](size_t index) noexcept requires(!std::is_const_v<M>) { return m[index + forward_size()*m_index]; }
    inline constexpr decltype(auto) at(size_t index) const { return index + forward_size()*m_index > m.size() ? throw std::out_of_range() : operator[](index); }
    inline constexpr decltype(auto) at(size_t index) requires(!std::is_const_v<M>) { return index + forward_size()*m_index > m.size() ? throw std::out_of_range() : operator[](index); }

    inline constexpr size_t size() const noexcept { return transversal_size(); } 

    inline constexpr auto begin() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, 0);
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto cbegin() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, 0);
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto rbegin() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, 0));
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto crbegin() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, 0));
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto end() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, size());
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto cend() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, size());
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto rend() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, size()));
        } else {
            return std::reverse_iterator(iterator(m, m_index, size()));
        }
    }

    inline constexpr auto crend() const {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, size()));
        } else {
            return std::reverse_iterator(iterator(m, m_index, size()));
        }
    }


    inline constexpr auto begin() {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, 0);
        } else {
            return iterator(m, 0, m_index);
        }
    }

    inline constexpr auto rbegin() {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, 0));
        } else {
            return std::reverse_iterator(iterator(m, 0, m_index));
        }
    }

    inline constexpr auto end() {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return iterator(m, m_index, size());
        } else {
            return iterator(m, size(), m_index);
        }
    }

    inline constexpr auto rend() {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return std::reverse_iterator(iterator(m, m_index, size()));
        } else {
            return std::reverse_iterator(iterator(m, size(), m_index));
        }
    }
};

template<matrix M, MatrixDimension Dim>
class MatrixDimensionViewFactory {
    std::conditional_t<M::is_temporary, M, M&> m;

    inline constexpr static auto factory(M& m, size_t index) {
        return MatrixDimensionView<M, Dim>(m, index);
    }
public:
    class iterator {
        std::conditional_t<M::is_temporary, M, M&> m;
        size_t m_index;
        template<matrix Mat, MatrixDimension _Dim> friend constexpr typename MatrixDimensionViewFactory<Mat, _Dim>::iterator operator+(ssize_t, typename MatrixDimensionViewFactory<Mat, _Dim>::iterator&);
        template<matrix Mat, MatrixDimension _Dim> friend constexpr const typename MatrixDimensionViewFactory<Mat, _Dim>::iterator operator+(ssize_t, const typename MatrixDimensionViewFactory<Mat, _Dim>::iterator&);
    public:
        constexpr iterator(M& m, size_t index) noexcept: m{m}, m_index{index} {}

        inline constexpr decltype(auto) operator*() const noexcept { return factory(m, m_index); }
        // constexpr decltype(auto) operator->() const { return v.data()+m_index; }

        inline constexpr decltype(auto) operator[](ssize_t index) const { return factory(m, m_index + index); }

        inline constexpr auto& operator++() noexcept { m_index++; return *this; }
        inline constexpr auto operator++(int) noexcept { return iterator(m, m_index++); }
        inline constexpr auto& operator--() noexcept { m_index--; return *this; }
        inline constexpr auto operator--(int) noexcept { return iterator(m, m_index--); }

        inline constexpr auto operator+(ssize_t index) const noexcept { return iterator(m, m_index + index); }
        inline constexpr auto operator-(ssize_t index) const noexcept { return iterator(m, m_index + index); }
        inline constexpr auto& operator+=(ssize_t index) noexcept { m_index += index; return *this; }
        inline constexpr auto& operator-=(ssize_t index) noexcept { m_index += index; return *this; }

        inline constexpr auto operator-(iterator it) const noexcept { return m_index - it.m_index; }

        inline constexpr bool operator==(iterator it) const noexcept { return (&m == &it.m) && (m_index == it.m_index);}
        inline constexpr bool operator!=(iterator it) const noexcept { return (&m != &it.m) || (m_index != it.m_index);}

        inline constexpr std::partial_ordering operator<=>(iterator it) const noexcept { return ((&m != &it.m) ? std::partial_ordering::unordered : 
            (m_index < it.index ? std::partial_ordering::less :
            (m_index == it.index ? std::partial_ordering::equivalent :
            (m_index > it.index ? std::partial_ordering::greater : std::partial_ordering::unordered))))
        ; }

        using difference_type = ptrdiff_t;
        using value_type = typename std::decay_t<M>::value_type;
        // using pointer = value_type*;
        using reference = decltype(std::declval<iterator>()[std::declval<size_t>()]);
        using iterator_category = std::random_access_iterator_tag;
    };

    using value_type = typename M::value_type;
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { M::static_rows };

    inline constexpr MatrixDimensionViewFactory(M& m): m{m} {}
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return factory(m, index); }
    inline constexpr decltype(auto) operator[](size_t index) noexcept requires(!std::is_const_v<M>) { return factory(m, index); }
    inline constexpr decltype(auto) at(size_t index) const { return index > m.rows() ? throw std::out_of_range() : operator[]; }
    inline constexpr decltype(auto) at(size_t index) requires(!std::is_const_v<M>) { return index > m.rows() ? throw std::out_of_range() : operator[](index); }

    inline constexpr size_t size() const noexcept {
        if constexpr(MatrixDimension::BY_ROWS == Dim) {
            return m.cols();
        } else {
            return m.rows();
        }
    }

    inline constexpr auto begin() const { return iterator(m, 0); }
    inline constexpr auto cbegin() const { return iterator(m, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() const { return iterator(m, size()); }
    inline constexpr auto cend() const { return iterator(m, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(m, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(m, size())); }

    inline constexpr auto begin() { return iterator(m, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() { return iterator(m, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(m, size())); }
};

template<matrix M, MatrixDimension Dim>
inline constexpr typename MatrixDimensionViewFactory<M, Dim>::iterator operator+(ssize_t idx, typename MatrixDimensionViewFactory<M, Dim>::iterator& m) {
    return iterator(m, idx);
}
template<matrix M, MatrixDimension Dim>
inline constexpr const typename MatrixDimensionViewFactory<M, Dim>::iterator operator+(ssize_t idx, const typename MatrixDimensionViewFactory<M, Dim>::iterator& m) {
    return iterator(m, idx);
}


template<matrix M, MatrixDimension Dim>
class ElementsView {
    std::conditional_t<M::is_temporary, M, M&> m;

    struct index_picker {
        inline constexpr static size_t pick(const auto& element, size_t index) {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return index;
            } else {
                size_t col = index%element.cols(), row = index%element.rows();
                return col*element.rows() + row;
            }   
        }
    };
public:
    using value_type = typename M::value_type;
    inline constexpr static bool is_temporary { true };
    inline constexpr static size_t static_size { M::static_size };
    using iterator = linear_element_iterator<M, index_picker>;

    constexpr ElementsView(M& m): m{m} {}
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return (m[index_picker::pick(m, index)]); }
    inline constexpr decltype(auto) operator[](size_t index) noexcept requires(!std::is_const_v<M>) { return (m[index_picker::pick(m, index)]); }
    inline constexpr decltype(auto) at(size_t index) const { return index > m.size() ? throw std::out_of_range() : operator[](index); }
    inline constexpr decltype(auto) at(size_t index) requires(!std::is_const_v<M>) { return index > m.size() ? throw std::out_of_range() : operator[](index); }

    inline constexpr size_t size() const noexcept { return m.size(); }

    inline constexpr auto begin() const { return iterator(m, 0); }
    inline constexpr auto cbegin() const { return iterator(m, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() const { return iterator(m, size()); }
    inline constexpr auto cend() const { return iterator(m, size()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(m, size())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(m, size())); }

    inline constexpr auto begin() { return iterator(m, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() { return iterator(m, size()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(m, size())); }
};


}

}