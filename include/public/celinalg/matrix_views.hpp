#pragma once

#include "celinalg/forward.hpp"
#include "celinalg/utils.hpp"
#include "celinalg/vector.hpp"

namespace celinalg {

enum class MatrixDimension {
    BY_ROWS,
    BY_COLS
};

enum class MatrixViewType {
    STRAIGHT,
    TRANSPOSED
};

namespace detail {

template<matrix M, MatrixDimension Dim>
class MatrixDimensionView {
    std::conditional_t<M::is_temporary, M, M&> m;
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

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

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    constexpr decltype(auto) pick(size_t index) const noexcept { return (*this)[index]; }
    constexpr decltype(auto) pick(size_t index) noexcept { return (*this)[index]; }
public:
    inline constexpr static bool is_temporary { true };
    inline constexpr static bool is_view { true };
    inline constexpr static bool is_expression { M::is_expression };
    inline constexpr static size_t static_size { Dim == MatrixDimension::BY_ROWS ? M::static_cols : M::static_rows };
    using matrix_type = M;
    using value_type = typename M::value_type;
    class iterator {
        inline constexpr size_t pick(ssize_t index = 0) const noexcept {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return m_index*m.rows() + (index + m_subindex);
            } else {
                return (index + m_index)*m.rows() + m_subindex;
            }
        }

        std::conditional_t<M::is_temporary, M, M&> m;
        size_t m_index;
        size_t m_subindex;
    public:
        constexpr iterator(M& m, size_t index, size_t subindex) noexcept: m{m}, m_index{index}, m_subindex{subindex} {}

        inline constexpr decltype(auto) operator*() const noexcept { return m.pick(pick(0)); }
        inline constexpr decltype(auto) operator[](ssize_t index) const { return m.pick(pick(index)); }

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
        using reference = decltype(std::declval<iterator>()[std::declval<size_t>()]);
        using iterator_category = std::random_access_iterator_tag;
    };
    constexpr MatrixDimensionView(M& m, size_t index) noexcept: m{m}, m_index{index} {}

    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return m.pick(index + forward_size()*m_index); }
    inline constexpr decltype(auto) operator[](size_t index) noexcept requires(!std::is_const_v<M>) { return m.pick(index + forward_size()*m_index); }
    inline constexpr decltype(auto) at(size_t index) const { return index + forward_size()*m_index > m.size() ? throw std::out_of_range() : operator[](index); }
    inline constexpr decltype(auto) at(size_t index) requires(!std::is_const_v<M>) { return index + forward_size()*m_index > m.size() ? throw std::out_of_range() : operator[](index); }

    inline constexpr size_t size() const noexcept { return transversal_size(); } 

    template<vector T>
    inline constexpr decltype(auto) operator=(T&& t) noexcept requires (!(M::is_expression || std::is_const_v<M>) && suitable_vector_expression<MatrixDimensionView, std::decay_t<T>>) {
        if constexpr(!(M::is_expression || std::is_const_v<M>)) {
            std::copy(begin(), end(), t.begin());
        } else throw std::logic_error("MatrixDimensionView<M>::operator=: M is const");
        return (*this);
    }

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
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

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
        using reference = decltype(std::declval<iterator>()[std::declval<size_t>()]);
        using iterator_category = std::random_access_iterator_tag;
    };

    using value_type = typename M::value_type;
    using matrix_type = M;
    inline constexpr static bool is_temporary  { true };
    inline constexpr static bool is_expression { M::is_expression };
    inline constexpr static bool is_view       { true };
    inline constexpr static size_t static_size { Dim == MatrixDimension::BY_ROWS ? M::static_rows : M::static_cols };

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

    inline constexpr auto begin() const noexcept { return iterator(m, 0); }
    inline constexpr auto cbegin() const noexcept { return iterator(m, 0); }
    inline constexpr auto rbegin() const noexcept { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto crbegin() const noexcept { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() const noexcept { return iterator(m, size()); }
    inline constexpr auto cend() const noexcept { return iterator(m, size()); }
    inline constexpr auto rend() const noexcept { return std::reverse_iterator(iterator(m, size())); }
    inline constexpr auto crend() const noexcept { return std::reverse_iterator(iterator(m, size())); }

    inline constexpr auto begin() noexcept { return iterator(m, 0); }
    inline constexpr auto rbegin() noexcept { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() noexcept { return iterator(m, size()); }
    inline constexpr auto rend() noexcept { return std::reverse_iterator(iterator(m, size())); }
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
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

    struct index_picker {
        inline constexpr static size_t pick(const auto& element, size_t index) {
            if constexpr(MatrixDimension::BY_ROWS == Dim) {
                return index;
            } else {
                size_t row = index%element.cols(), col = index/element.cols();
                // std::swap(row, col);
                return row*element.cols() + col;
            }   
        }
    };
    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    inline constexpr decltype(auto) pick(size_t index) const noexcept { return (m.pick(index)); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return (m.pick(index)); }
public:
    using value_type = typename M::value_type;
    using matrix_type = M;
    inline constexpr static bool is_temporary { true };
    inline constexpr static bool is_expression { M::is_expression };
    inline constexpr static bool is_view { true };
    inline constexpr static size_t static_size { M::static_size };
    using iterator = detail::linear_element_iterator<M, index_picker>;

    constexpr ElementsView(M& m): m{m} {}
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return (m.pick(index_picker::pick(m, index))); }
    inline constexpr decltype(auto) operator[](size_t index) noexcept requires(!std::is_const_v<M>) { return (m.pick(index_picker::pick(m, index))); }
    inline constexpr decltype(auto) at(size_t index) const { return index > m.numel() ? throw std::out_of_range() : operator[](index); }
    inline constexpr decltype(auto) at(size_t index) requires(!std::is_const_v<M>) { return index > m.numel() ? throw std::out_of_range() : operator[](index); }

    inline constexpr std::pair<size_t, size_t> size() const noexcept { return m.size(); }
    inline constexpr size_t numel() const noexcept { return m.numel(); }

    inline constexpr auto begin() const { return iterator(m, 0); }
    inline constexpr auto cbegin() const { return iterator(m, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() const { return iterator(m, numel()); }
    inline constexpr auto cend() const { return iterator(m, numel()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(m, numel())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(m, numel())); }

    inline constexpr auto begin() { return iterator(m, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(m, 0)); }
    inline constexpr auto end() { return iterator(m, numel()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(m, numel())); }
};



template<matrix M, MatrixViewType ViewType = MatrixViewType::STRAIGHT>
class MatrixView {
public:
    using value_type = typename M::value_type;
    using matrix_type = M;
    inline constexpr static bool is_temporary  { true };
    inline constexpr static bool is_expression { M::is_expression };
    inline constexpr static bool is_view       { true };
    inline constexpr static size_t static_rows { 0 };
    inline constexpr static size_t static_cols { 0 };
    inline constexpr static size_t static_size { 0 };
private:
    std::conditional_t<M::is_temporary, M, M&> m;
    size_t m_rowStart, m_rowEnd, m_colStart, m_colEnd;

    template<typename Element, typename IndexPicker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class MatrixView<MatrixView>;
    friend class ElementsView<MatrixView, MatrixDimension::BY_ROWS>;
    friend class ElementsView<MatrixView, MatrixDimension::BY_COLS>;
    friend class MatrixDimensionView<MatrixView, MatrixDimension::BY_ROWS>;
    friend class MatrixDimensionView<MatrixView, MatrixDimension::BY_COLS>;
    friend class MatrixDimensionViewFactory<MatrixView, MatrixDimension::BY_ROWS>;
    friend class MatrixDimensionViewFactory<MatrixView, MatrixDimension::BY_COLS>;

    friend class ElementsView<const MatrixView, MatrixDimension::BY_ROWS>;
    friend class ElementsView<const MatrixView, MatrixDimension::BY_COLS>;
    friend class MatrixDimensionView<const MatrixView, MatrixDimension::BY_ROWS>;
    friend class MatrixDimensionView<const MatrixView, MatrixDimension::BY_COLS>;
    friend class MatrixDimensionViewFactory<const MatrixView, MatrixDimension::BY_ROWS>;
    friend class MatrixDimensionViewFactory<const MatrixView, MatrixDimension::BY_COLS>;

    inline constexpr decltype(auto) pick(size_t index) const noexcept {
        /* 
               
                ┌────────────────────────┐
                │  Original matrix       │
                │                        │
                │        ┌──────────┐    │
                │        │Submatrix │    │
                │        │          │    │
                │        │          │    │
                │        │          │    │
                │        │          │    │
                │        └──────────┘    │
                │                        │
                └────────────────────────┘

                X: Element to access
                ┼: Elements to discard
                ┌────────────────────────┐
                │┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼│
                │┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼┼│
                │┼┼┼┼┼┼┼┼┼──────────┼┼┼┼┼│
                │┼┼┼┼┼┼┼┼│          │┼┼┼┼│
                │┼┼┼┼┼┼┼┼┼───┬─┬────┼────┤
                │┼┼┼┼┼┼┼┼│   │x│    │    │
                ├────────┼───┴─┴────┤    │
                │        │          │    │
                │        └──────────┘    │
                │                        │
                └────────────────────────┘
         */
        if constexpr(ViewType == MatrixViewType::STRAIGHT) {
            size_t idx = 0, tmp = index/rows();
            idx += m.cols()*m_rowStart;
            idx += (1+tmp)*m_colStart;
            idx += tmp == 0 ? 0 : (tmp - 1)*(m.cols()-m_colEnd);
            idx += index;
            return m.pick(idx);
        } else {
            return m.pick(m.cols()*(index%m.rows()) + index/m.rows());
        }
    }

    inline constexpr decltype(auto) pick(size_t index) noexcept {
        if constexpr(ViewType == MatrixViewType::STRAIGHT) {
            size_t idx = 0, _rows = index/rows();
            idx += m.cols()*m_rowStart;
            idx += _rows*m_colStart;
            idx += _rows == 0 ? 0 : (_rows - 1)*(m.cols() - m_colEnd);
            idx += index;
            return m.pick(idx);
        } else {
            return m.pick(m.cols()*(index%m.rows()) + index/m.rows());
        }
    }
public:    
    constexpr MatrixView(M& mat, size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd)
        : m{mat}
        , m_rowStart{rowStart}
        , m_rowEnd{rowEnd}
        , m_colStart{colStart}
        , m_colEnd{colEnd}
    {
        if(m_rowStart >= m_rowEnd)
            throw std::runtime_error("MatrixView: rowStart >= rowEnd");
        if(m_colStart >= m_colEnd)
            throw std::runtime_error("MatrixView: colStart >= colEnd");
        if(m_rowEnd - m_rowStart > m.rows())
            throw std::runtime_error("MatrixView: more rows than matrix");
        if(m_colEnd - m_colStart > m.cols())
            throw std::runtime_error("MatrixView: more cols than matrix");
    }

    constexpr MatrixView(M& mat) requires(ViewType == MatrixViewType::TRANSPOSED)
        : m{mat}
    {}

    template<matrix Mat>
    inline constexpr MatrixView& operator=(const Mat& mat) noexcept requires (!std::is_const_v<M>) {
        if constexpr(!std::is_const_v<M>) {
            std::copy(mat.elements_view().begin(), mat.elements_view().end(), elements_view().begin());
        }
        return *this;
    }

    inline constexpr size_t size() const noexcept {
        return rows()*cols();
    }

    inline constexpr size_t rows() const noexcept {
        if constexpr(ViewType == MatrixViewType::STRAIGHT) {
            return m_rowEnd - m_rowStart;
        } else if constexpr(ViewType == MatrixViewType::TRANSPOSED) {
            return m.cols();
        }
    }

    inline constexpr size_t cols() const noexcept {
        if constexpr(ViewType == MatrixViewType::STRAIGHT) {
            return m_colEnd - m_colStart;
        } else if constexpr(ViewType == MatrixViewType::TRANSPOSED) {
            return m.rows();
        }
    }

    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   noexcept { return detail::ElementsView<MatrixView, Dim>(*this); }
    inline constexpr auto rows_view()       noexcept { return detail::MatrixDimensionViewFactory<MatrixView, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       noexcept { return detail::MatrixDimensionViewFactory<MatrixView, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) noexcept { return detail::MatrixDimensionView<MatrixView, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) noexcept { return detail::MatrixDimensionView<MatrixView, MatrixDimension::BY_COLS>(*this, index); }
                                      
    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   const noexcept { return detail::ElementsView<const MatrixView, Dim>(*this); }
    inline constexpr auto rows_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatrixView, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatrixView, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) const noexcept { return detail::MatrixDimensionView<const MatrixView, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) const noexcept { return detail::MatrixDimensionView<const MatrixView, MatrixDimension::BY_COLS>(*this, index); }

    inline constexpr auto operator[](size_t index) noexcept { if constexpr(ViewType==MatrixViewType::STRAIGHT) { return row(index); } else if constexpr(ViewType==MatrixViewType::TRANSPOSED) { return col(index); } }
    inline constexpr auto operator[](size_t index) const noexcept { if constexpr(ViewType==MatrixViewType::STRAIGHT) { return row(index); } else if constexpr(ViewType==MatrixViewType::TRANSPOSED) { return col(index); } }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) noexcept { return (*this)[row][col]; }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) const noexcept { return (*this)[row][col]; }

    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) noexcept { return MatrixView<std::decay_t<decltype(*this)>>(*this, rowStart, rowEnd, colStart, colEnd); }
    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) const noexcept { return MatrixView<const std::decay_t<decltype(*this)>>(*this, rowStart, rowEnd, colStart, colEnd); }

    inline constexpr MatrixView<MatrixView, MatrixViewType::TRANSPOSED> transpose() noexcept { return {*this}; }
    inline constexpr MatrixView<const MatrixView, MatrixViewType::TRANSPOSED> transpose() const noexcept { return {*this}; }
};

}

template<matrix M> inline constexpr detail::MatrixView<M, MatrixViewType::TRANSPOSED> transpose(M& m) { return {m}; }
template<matrix M> inline constexpr detail::MatrixView<M, MatrixViewType::TRANSPOSED> transpose(const M& m) { return {m}; }

}