#pragma once

#include "linalg/forward.hpp"
#include "linalg/matrix_views.hpp"
#include "linalg/vector.hpp"

#include <array>
#include <vector>
#include <tuple>
#include <span>

namespace linalg {

template<req::number T, size_t Rows, size_t Cols>
class Matrix {
public:
    using value_type = T;
    inline constexpr static size_t static_rows { Rows };
    inline constexpr static size_t static_cols { Cols };
    inline constexpr static bool is_temporary { false };
    inline constexpr static size_t static_size     { static_rows*static_cols };
private:
    inline constexpr static bool has_dynamic_rows  { static_rows == 0 };
    inline constexpr static bool has_dynamic_cols  { static_cols == 0 };
    inline constexpr static bool is_dynamic        { has_dynamic_rows && has_dynamic_cols };
    inline constexpr static bool is_static         { static_size != 0 };

    struct StaticImplementation {
        std::array<Vector<value_type, static_rows>, static_cols> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { auto cols = i%static_cols; auto rows = i/static_cols;
            std::cout << "asking for index " << i << ", got data[" << cols << "][" << rows << "]" << std::endl; 
            return data[cols][rows]; }
        inline constexpr decltype(auto) operator[](size_t i)       noexcept { auto cols = i%static_cols; auto rows = i/static_cols;
            std::cout << "asking for index " << i << ", got data[" << cols << "][" << rows << "]" << std::endl; return data[cols][rows]; }
    };

    struct DynamicRowsImplementation {
        std::array<DynamicVector<value_type>, static_cols> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { if(data[0].empty()) return data[0][-1]; return data[i%static_cols][i/static_cols]; }        
        inline constexpr decltype(auto) operator[](size_t i)       noexcept { if(data[0].empty()) return data[0][-1]; return data[i%static_cols][i/static_cols]; }        
    };

    struct DynamicColsImplementation {
        std::vector<Vector<value_type, static_rows>> data;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { if(data.empty()) return data[-1][-1]; return data[i%data.size()][i/data.size()]; }        
        inline constexpr decltype(auto) operator[](size_t i)       noexcept { if(data.empty()) return data[-1][-1]; return data[i%data.size()][i/data.size()]; }        
    };

    struct DynamicImplementation {
        std::vector<value_type> data;
        size_t rows;

        inline constexpr decltype(auto) operator[](size_t i) const noexcept { if(data.empty()) return data[-1]; auto cols = (data.size()/rows); return data[i%cols][i/cols]; }
        inline constexpr decltype(auto) operator[](size_t i)       noexcept { if(data.empty()) return data[-1]; auto cols = (data.size()/rows); return data[i%cols][i/cols]; }
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

    friend class detail::MatrixDimensionView<Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<Matrix, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<Matrix, MatrixDimension::BY_COLS>;
public:
    inline constexpr static Matrix ones(size_t rows, size_t cols) requires(is_dynamic) {
        Matrix m;
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix ones(size_t cols) requires(has_dynamic_cols) {
        Matrix m;
        m.cols_resize(cols);
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix ones(size_t rows) requires(has_dynamic_rows) {
        Matrix m;
        m.rows_resize(rows);
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    inline constexpr static Matrix ones() noexcept requires(is_static) {
        Matrix m;
        std::fill(m.elements_view().begin(), m.elements_view().end(), static_cast<value_type>(1));
        return m;
    }

    constexpr Matrix() noexcept(is_static){
        static_resize();
        std::fill(elements_view().begin(), elements_view().end(), static_cast<value_type>(0));
    }

    constexpr Matrix(auto&&... args) noexcept requires(is_static)
        : m_impl { .data { std::forward<decltype(args)>(args)... } }
    {}

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
                std::copy(std::begin(std::begin(array) + counter), std::end(std::begin(array) + counter), row.begin());
                counter++;
            });
        }
    }

    constexpr Matrix(const std::array<std::array<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            for(auto row : rows_view()) {
                std::copy((array.begin() + counter)->begin(), (array.begin() + counter)->end(), row.begin());
                counter++;
            }
        }
    }
    
    constexpr Matrix(const std::span<std::span<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy((array.begin() + counter)->begin(), (array.begin() + counter)->end(), row.begin());
                counter++;
            });
        }
    }

    constexpr Matrix(const std::span<std::array<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy((array.begin() + counter)->begin(), (array.begin() + counter)->end(), row.begin());
                counter++;
            });
        }
    }
    
    constexpr Matrix(const std::array<std::span<value_type, static_rows>, static_cols>& array) noexcept requires(is_static) {
        static_resize();
        if constexpr(is_static) {
            size_t counter = 0;
            std::for_each(rows_view().begin(), rows_view().end(), [&](auto row){
                std::copy((array.begin() + counter)->begin(), (array.begin() + counter)->end(), row.begin());
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
            std::copy((array.begin() + counter)->begin(), (array.begin() + counter)->end(), row.begin());
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
    constexpr Matrix(const M& m) noexcept(is_static) requires suitable_matrix_addition_expression<Matrix, M> {
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
    inline constexpr Matrix& operator=(const M& m) noexcept(is_static) requires(suitable_matrix_addition_expression<Matrix, M>) {
        static_resize(m.rows(), m.cols());
        std::copy(m.elements_view().begin(), m.elements_view().end(), elements_view().begin());
        return *this;
    }

    inline constexpr Matrix& operator=(Matrix&& m) noexcept {
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

    inline constexpr void resize(size_t rows) requires(has_dynamic_rows) {
        static_resize(rows, static_cols);
    }

    inline constexpr void resize(size_t cols) requires(has_dynamic_cols) {
        static_resize(static_rows, cols);
    }

    inline constexpr size_t size() const noexcept {
        if constexpr(is_dynamic) {
            return m_data().size();
        } else if constexpr(has_dynamic_cols) {
            return m_data().size()*static_rows;
        } else if constexpr(has_dynamic_rows) {
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
            return m_data().size();
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

    
    inline constexpr decltype(auto) operator[](size_t index) const noexcept { return m_impl[index]; }
    inline constexpr decltype(auto) operator[](size_t index) noexcept { return m_impl[index]; }                          
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

Matrix(matrix auto&& m) -> Matrix<
    typename std::decay_t<decltype(m)>::value_type, 
    std::decay_t<decltype(m)>::static_rows, 
    std::decay_t<decltype(m)>::static_cols
>;

}

#include "linalg/matrix.hxx"