#pragma once

#include <array>
#include <vector>
#include <tuple>
#include <span>

#include "celinalg/forward.hpp"
#include "celinalg/matrix_views.hpp"
#include "celinalg/vector.hpp"
#include "celinalg/matrix.hxx"
#include "celinalg/traits.hpp"
#include "celinalg/utils.hpp"

namespace celinalg {

template<req::number T, size_t Rows, size_t Cols>
class Matrix {
public:
    using value_type = T;
    inline constexpr static size_t static_rows { Rows };
    inline constexpr static size_t static_cols { Cols };
    inline constexpr static bool is_temporary  { false };
    inline constexpr static bool is_expression { false };
    inline constexpr static bool is_view       { false };
    inline constexpr static size_t static_size { static_rows*static_cols };
private:
    inline constexpr static bool has_dynamic_rows  { static_rows == 0 };
    inline constexpr static bool has_dynamic_cols  { static_cols == 0 };
    inline constexpr static bool is_dynamic        { has_dynamic_rows && has_dynamic_cols || static_size*sizeof(value_type) > 64 };
    inline constexpr static bool is_static         { static_size != 0 };

    struct StaticImplementation {
        std::array<Vector<value_type, static_rows>, static_cols> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { 
            auto cols = i%static_cols; auto rows = i/static_cols;
            return data[cols][rows]; 
        }

        inline constexpr decltype(auto) operator[](size_t i) noexcept { 
            auto cols = i%static_cols; auto rows = i/static_cols;
            return data[cols][rows]; 
        }
    };

    struct DynamicRowsImplementation {
        std::array<DynamicVector<value_type>, static_cols> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { 
            if(data[0].empty())
                return data[0][-1];
            return data[i%static_cols][i/static_cols];
        }

        inline constexpr decltype(auto) operator[](size_t i) noexcept { 
            if(data[0].empty())
                return data[0][-1];
            return data[i%static_cols][i/static_cols];
        }
    };

    struct DynamicColsImplementation {
        std::vector<Vector<value_type, static_rows>> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept {
            if(data.empty())
                return data[-1][-1];
            return data[i%data.size()][i/data.size()];
        }

        inline constexpr decltype(auto) operator[](size_t i) noexcept {
            if(data.empty())
                return data[-1][-1];
            return data[i%data.size()][i/data.size()];
        }
    };

    struct DynamicImplementation {
        std::vector<value_type> data;
        size_t rows;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept {
            return data[i];
        }

        inline constexpr decltype(auto) operator[](size_t i) noexcept {
            return data[i];
        }
    };

    using implementation_type = std::conditional_t<is_dynamic, DynamicImplementation, 
        std::conditional_t<has_dynamic_rows, DynamicRowsImplementation, 
            std::conditional_t<has_dynamic_cols, DynamicColsImplementation,
                StaticImplementation
            >
        >
    >;

    implementation_type m_impl;
    inline constexpr decltype(auto) m_data() const noexcept { return (m_impl.data); }
    inline constexpr decltype(auto) m_data() noexcept { return (m_impl.data); }

    inline constexpr void static_resize(size_t rows, size_t cols) noexcept(is_static) {
        if constexpr(is_dynamic) {
            m_data().resize(rows*cols);
            m_impl.rows = rows;
        } else if constexpr(has_dynamic_cols) {
            m_data().resize(cols);
        } else if constexpr(has_dynamic_rows) {
            std::for_each(m_data().begin(), m_data().end(), [](auto& row) { row.resize(static_rows); });
        }
    }

    inline constexpr void static_resize() noexcept(is_static) {
        static_resize(static_rows, static_cols);
    }

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class detail::MatrixView<Matrix>;
    friend class detail::ElementsView<Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixView<Matrix, MatrixViewType::STRAIGHT>;
    friend class detail::MatrixView<Matrix, MatrixViewType::TRANSPOSED>;

    friend class detail::ElementsView<const Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<const Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<const Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<const Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<const Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<const Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixView<const Matrix, MatrixViewType::STRAIGHT>;
    friend class detail::MatrixView<const Matrix, MatrixViewType::TRANSPOSED>;

    template<detail::Operation op, matrix Mat1, matrix Mat2> requires suitable_matrix_same_size_expression<Mat1, Mat2> friend class detail::MatExpression;
    template<detail::Operation op, matrix, req::number> friend class detail::MatScalarExpression;
    template<detail::Operation op, req::number, matrix> friend class detail::ScalarMatExpression;
    template<matrix Mat1, matrix Mat2> requires suitable_matrix_cross_size_expression<Mat1, Mat2> friend class detail::MatrixCrossProductExpression;

    template<matrix M1, matrix M2, MatrixDimension Dim> requires(
        (Dim==MatrixDimension::BY_COLS && (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2> || M1::static_rows == M2::static_rows)) || 
        (Dim==MatrixDimension::BY_ROWS && (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2> || M1::static_cols == M2::static_cols))
    )
    friend class MatrixScalarProductExpression;

    inline constexpr decltype(auto) pick(size_t index) const noexcept { return m_impl[index]; }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return m_impl[index]; }  
public:
    inline constexpr static Matrix ones(size_t rows, size_t cols) requires(is_dynamic) {
        Matrix m;
        if constexpr(is_dynamic) {
            m.resize(rows, cols);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix ones(size_t cols) requires(has_dynamic_cols) {
        Matrix m;
        if constexpr(has_dynamic_cols) {
            m.resize(cols);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix ones(size_t rows) requires(has_dynamic_rows) {
        Matrix m;
        if constexpr(has_dynamic_rows) {
            m.resize(rows);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }    
    
    inline constexpr static Matrix ones() noexcept requires(is_static) {
        Matrix m;
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix zeros(size_t rows, size_t cols) requires(is_dynamic) {
        Matrix m;
        if constexpr(is_dynamic) {
            m.resize(rows, cols);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(0));
        return m;
    }

    inline constexpr static Matrix zeros(size_t cols) requires(has_dynamic_cols) {
        Matrix m;
        if constexpr(has_dynamic_cols) {
            m.resize(cols);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(0));
        return m;
    }

    inline constexpr static Matrix zeros(size_t rows) requires(has_dynamic_rows) {
        Matrix m;
        if constexpr(has_dynamic_rows) {
            m.resize(rows);
        }
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(0));
        return m;
    }

    inline constexpr static Matrix zeros() noexcept requires(is_static) {
        return {};
    }

    inline constexpr static Matrix diag(size_t = 0) noexcept requires(square_matrix<Matrix>) {
        Matrix m;
        for(auto i = 0; i != size; ++i) {
            m[i][i] = 1;
        }
        return m;
    }

    inline constexpr static Matrix diag(size_t = 0) noexcept requires(has_dynamic_rows) {
        auto m = zeros(static_cols);
        for(auto i = 0; i != size; ++i) {
            m[i][i] = 1;
        }
        return m;
    }

    inline constexpr static Matrix diag(size_t = 0) noexcept requires(has_dynamic_cols) {
        auto m = zeros(static_rows);
        for(auto i = 0; i != size; ++i) {
            m[i][i] = 1;
        }
        return m;
    }

    inline constexpr static Matrix diag(size_t size, size_t = 0) noexcept requires(is_dynamic) {
        auto m = zeros(size, size);
        for(auto i = 0; i != size; ++i) {
            m[i][i] = 1;
        }
        return m;
    }

    constexpr Matrix() noexcept(is_static){
        static_resize();
        std::fill(elements_view().begin(), elements_view().end(), static_cast<value_type>(0));
    }

    constexpr Matrix(const value_type(&array)[static_rows*static_cols]) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            std::copy(std::begin(array), std::end(array), elements_view().begin());
        }
    }

    constexpr Matrix(const value_type(&array)[static_cols][static_rows]) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy(array[counter], array[counter] + static_cols, row.begin());
                counter++;
            });
        }
    }

    constexpr Matrix(const std::array<std::array<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            for(auto row : rows_view()) {
                std::copy(array[counter].begin(), array[counter].end(), row.begin());
                counter++;
            }
        }
    }
    
    constexpr Matrix(const std::span<std::span<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy(array[counter].begin(), array[counter].end(), row.begin());
                counter++;
            });
        }
    }

    constexpr Matrix(const std::span<std::array<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy(array[counter].begin(), array[counter].end(), row.begin());
                counter++;
            });
        }
    }
    
    constexpr Matrix(const std::array<std::span<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy(array[counter].begin(), array[counter].end(), row.begin());
                counter++;
            });
        }
    }

    constexpr Matrix(const std::span<std::span<value_type>>& array) requires(is_dynamic) {
        if(!std::all_of(array.begin(), array.end(), [](){})) {
            throw std::runtime_error("invalid input matrix (row sizes mismatch)");
        }
        static_resize(array.size(), array.empty() ? 0 : array[0].size());
        size_t counter = 0;
        std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
            std::copy(array[counter].begin(), array[counter].end(), row.begin());
            counter++;
        });
    }

    constexpr Matrix(const std::vector<std::vector<value_type>>& vector) requires(is_dynamic) {
        if(!std::all_of(vector.begin(), vector.end(), [](){})) {
            throw std::runtime_error("invalid input matrix (row sizes mismatch)");
        }
        static_resize(vector.size(), vector.empty() ? 0 : vector[0].size());
        size_t counter = 0;
        std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
            std::copy((vector.begin() + counter)->begin(), (vector.begin() + counter)->end(), row.begin());
            counter++;
        });
    }

    constexpr Matrix(const std::span<value_type>& v, size_t rows) requires(is_dynamic) {
        auto cols = v.size()/rows;
        static_resize(rows, cols);
        std::copy(v.begin(), v.end(), m_data().begin());
    }

    constexpr Matrix(const std::vector<value_type>& v, size_t rows) requires(is_dynamic) {
        auto cols = v.size()/rows;
        static_resize(rows, cols);
        std::copy(v.begin(), v.end(), m_data().begin());
    }

    constexpr Matrix(std::vector<value_type>&& v, size_t rows) noexcept requires(is_dynamic) {
        if constexpr(is_dynamic) {
            m_data() = std::move(v);
            m_impl.rows = rows;
        }
    }

    template<matrix M>
    constexpr Matrix(const M& m) noexcept(is_static) requires suitable_matrix_same_size_expression<Matrix, M> {
        static_resize(m.rows(), m.cols());
        std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
    }

    constexpr Matrix(Matrix&& m) noexcept {
        if constexpr(is_dynamic) {
            m_impl.rows = m.m_impl.rows;
        }
        if constexpr(has_dynamic_cols) {
            m_data().swap(m.m_data());
        } else {
            std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
        }
    }

    inline constexpr Matrix& operator=(value_type(&array)[static_rows*static_cols]) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            std::copy(std::begin(array), std::end(array), elements_view().begin());
        }
        return *this;
    }

    inline constexpr Matrix& operator=(value_type(&array)[static_cols][static_rows]) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy(std::begin(array) + static_rows*counter, std::begin(array) + 2*static_rows*counter -1, row.begin());
                counter++;
            });
        }
        return *this;
    }

    template<matrix M>
    inline constexpr Matrix& operator=(const M& m) noexcept(is_static) requires(suitable_matrix_same_size_expression<Matrix, M>) {
        if constexpr(std::same_as<Matrix, M>)
            if(this == *m) return *this;
        static_resize(m.rows(), m.cols());
        if constexpr(!static_matrix<M>) {
            if(rows() != m.rows() || cols() != m.cols())
                throw std::length_error(
                    "matrix assignment: wrong dimensions (" 
                        + std::to_string(rows()) + ", " + std::to_string(cols()) 
                    + ") vs (" 
                        + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) +  
                    ")");
        }
        if constexpr(contains_fixed_state_operation<M>) {
            // This branch is to be used e.g. for matrix products. Suppose there is this scenario
            //      A = cross(A, B)
            // where size(A) == size(B) == [3, 3]
            // The resulting elements are evaluated one by one and A is replaced without temporaries after the expression expansion, so
            //  A[0][0] = A.row(0)*B.col(0); // A[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0]
            //  A[0][1] = A.row(0)*B.col(1); // A[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1]
            // ... etc
            // You can see here that there is a bug in the second statement. The A[0][0] used in the addendum A[0][0]*B[0][1]
            // was replaced in the previous expression, but it should have been the old one.
            // To be semantically correct and still use the expression templates coherently, a temporary has to be created for this corner
            // case. This can be evaluated directly at compile time, since every expression either requires a fixed state in its evaluation
            // or it does not. The tree can be traversed to highlight if this happens or not.
            
            // Unfortunately, evaluating if the expression ```actually``` leads to a state invalidation is not possible at compile time.
            // Suppose this:
            //     C = cross(A, B)
            // there is no state invalidation here even if the cross product requires a fixed state of its operands.
            // Two solutions:
            //  - check it using references, like `if(expression_reference_check(this, m))`
            //  - assume the state invalidation happens and create a temporary anyway.
            // Either one or both the ways may be used, if a performance evaluation is done with a parameterization on the input types.
            // (e.g. for small sized matrices a copy may be faster than traversing a deep tree, 
            // viceversa if the tree height is small traversing it may be faster than creating a (suppose) 128*128 doubles temporary and copying it)
            // TODO add an if constexpr to skip the reference check for deep trees
            if(celinalg::utils::expression_reference_check_state_invalidation(this, &m)) {
                // std::cout << "state invalidation detected, using temporary" << std::endl;
                Matrix tmp {m};
                std::copy(tmp.elements_view().begin(), tmp.elements_view().end(), elements_view().begin());
            } else {
                std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
            }

            // TODO For dynamic vectors/matrices move semantics may be used to fill the temporary and then move it in the original container.

        } else {
            // If there aren't expressions with fixed state semantics 
            std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
        }
        return *this;
    }

    inline constexpr Matrix& operator=(Matrix&& m) noexcept {
        if(this == &m) return *this;
        if constexpr(is_dynamic) {
            m_impl.rows = m.m_impl.rows;
        }
        if constexpr(is_static) {
            std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
        } else {
            m_data().swap(m.m_data());
        }
        return *this;
    }

    inline constexpr void resize(size_t rows, size_t cols) requires(is_dynamic) {
        static_resize(rows, cols);
    }

    inline constexpr void resize(size_t rows) requires(!is_dynamic && has_dynamic_rows) {
        static_resize(rows, static_cols);
    }

    inline constexpr void resize(size_t cols) requires(!is_dynamic && has_dynamic_cols) {
        static_resize(static_rows, cols);
    }

    inline constexpr std::pair<size_t, size_t> size() const noexcept {
        return {rows(), cols()};
    }

    inline constexpr size_t numel() const noexcept {
        if constexpr(is_dynamic) {
            return m_data().size();
        } else if constexpr(has_dynamic_cols) {
            return m_data().size()*static_rows;
        } else if constexpr(has_dynamic_rows) {
            if(m_data().empty()) return 0;
            return static_cols*m_data()[0].size();
        } else {
            return static_size;
        }
    }

    inline constexpr size_t rows() const noexcept {
        if constexpr(is_dynamic) {
            return m_impl.rows;
        } else if constexpr(has_dynamic_rows) {
            return m_data().size();
        } else {
            return static_rows;
        }
    }

    inline constexpr size_t cols() const noexcept {
        if constexpr(is_dynamic) {
            return m_data().size()/m_impl.rows;
        } else if constexpr(has_dynamic_cols) {
            return m_data()[0].size();
        } else {
            return static_cols;
        }
    }

    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   noexcept { return detail::ElementsView<Matrix, Dim>(*this); }
    inline constexpr auto rows_view()       noexcept { return detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       noexcept { return detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) noexcept { return detail::MatrixDimensionView<Matrix, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) noexcept { return detail::MatrixDimensionView<Matrix, MatrixDimension::BY_COLS>(*this, index); }
                                      
    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   const noexcept { return detail::ElementsView<const Matrix, Dim>(*this); }
    inline constexpr auto rows_view()       const noexcept { return detail::MatrixDimensionViewFactory<const Matrix, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       const noexcept { return detail::MatrixDimensionViewFactory<const Matrix, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) const noexcept { return detail::MatrixDimensionView<const Matrix, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) const noexcept { return detail::MatrixDimensionView<const Matrix, MatrixDimension::BY_COLS>(*this, index); }

    inline constexpr auto operator[](size_t index) noexcept { return row(index); }
    inline constexpr auto operator[](size_t index) const noexcept { return row(index); }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) noexcept { return (*this)[row][col]; }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) const noexcept { return (*this)[row][col]; }

    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) noexcept { return detail::MatrixView<Matrix>(*this, rowStart, rowEnd, colStart, colEnd); }
    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) const noexcept { return detail::MatrixView<const Matrix>(*this, rowStart, rowEnd, colStart, colEnd); }

    template<matrix M> inline constexpr decltype(auto) operator+=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this + m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator-=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this - m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator*=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this * m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator/=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this / m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator%=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this % m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator&=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this & m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator|=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this | m; return *this; }
    template<matrix M> inline constexpr decltype(auto) operator^=(const M& m) requires suitable_matrix_same_size_expression<Matrix, M> { *this = *this ^ m; return *this; }
    inline constexpr decltype(auto) operator+=(value_type v) { *this = *this + v; return *this; }
    inline constexpr decltype(auto) operator-=(value_type v) { *this = *this - v; return *this; }
    inline constexpr decltype(auto) operator*=(value_type v) { *this = *this * v; return *this; }
    inline constexpr decltype(auto) operator/=(value_type v) { *this = *this / v; return *this; }
    inline constexpr decltype(auto) operator%=(value_type v) { *this = *this % v; return *this; }
    inline constexpr decltype(auto) operator&=(value_type v) { *this = *this & v; return *this; }
    inline constexpr decltype(auto) operator|=(value_type v) { *this = *this | v; return *this; }
    inline constexpr decltype(auto) operator^=(value_type v) { *this = *this ^ v; return *this; }
};


template<req::number T, size_t Rows, size_t Cols>
Matrix(const T(&)[Rows][Cols]) -> Matrix<T, Rows, Cols>;

template<req::number T, size_t Rows, size_t Cols>
Matrix(const std::array<std::array<T, Rows>, Cols>&) -> Matrix<T, Rows, Cols>;

template<req::number T, size_t Rows, size_t Cols>
Matrix(const std::span<std::span<T, Rows>, Cols>&) -> Matrix<T, Rows, Cols>;

template<req::number T, size_t Rows, size_t Cols>
Matrix(const std::array<std::span<T, Rows>, Cols>&) -> Matrix<T, Rows, Cols>;

template<req::number T, size_t Rows, size_t Cols>
Matrix(const std::span<std::array<T, Rows>, Cols>&) -> Matrix<T, Rows, Cols>;

template<matrix M>
Matrix(const M& m) -> Matrix<typename M::value_type, M::static_rows, M::static_cols>;

}

#include "celinalg/matrix.hxx"