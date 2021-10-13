#pragma once

#include "celinalg/number.hpp"
#include "celinalg/numeric_traits.hpp"
#include "celinalg/number_type.hpp"
#include "celinalg/common_type.hpp"
#include "celinalg/forward.hpp"
#include "celinalg/utils.hpp"
#include "celinalg/matrix_views.hpp"
#include "celinalg/operation.hpp"

namespace celinalg {

namespace detail {

template<Operation op, matrix M1, matrix M2> requires suitable_matrix_same_size_expression<M1, M2>
class MatExpression;

template<Operation op, matrix M, req::number S>
class MatScalarExpression;

template<Operation op, req::number S, matrix M>
class ScalarMatExpression;

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
class MatrixSingleRowColProductExpression;

template<matrix M1, matrix M2, MatrixDimension Dim> requires(
    (Dim==MatrixDimension::BY_COLS && (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2> || M1::static_rows == M2::static_rows)) || 
    (Dim==MatrixDimension::BY_ROWS && (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2> || M1::static_cols == M2::static_cols))
)
class MatrixScalarProductExpression;

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
class MatrixCrossProductExpression;

template<Operation op, matrix M1, matrix M2> requires suitable_matrix_same_size_expression<M1, M2>
class MatExpression {
public:
    inline constexpr static bool is_temporary  { true };
    inline constexpr static bool is_expression { true };
    inline constexpr static bool is_view       { false };
    inline constexpr static size_t static_rows { (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2>) ? 0 : M1::static_rows };
    inline constexpr static size_t static_cols { (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2>) ? 0 : M1::static_cols };
    inline constexpr static size_t static_size { static_rows*static_cols };
    using value_type = traits::common_type_t<typename M1::value_type, typename M2::value_type>;

    using operand_type_1 = M1;
    using operand_type_2 = M2;
private:
    std::conditional_t<M1::is_temporary, M1, const M1&> m1;
    std::conditional_t<M2::is_temporary, M2, const M2&> m2;
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class detail::MatrixView<MatExpression>;
    friend class detail::ElementsView<MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<MatExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<MatExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<MatExpression, MatrixDimension::BY_COLS>;

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class detail::MatrixView<const MatExpression>;
    friend class detail::ElementsView<const MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<const MatExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<const MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<const MatExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<const MatExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<const MatExpression, MatrixDimension::BY_COLS>;

    template<Operation, matrix Mat1, matrix Mat2> requires suitable_matrix_same_size_expression<Mat1, Mat2> friend class detail::MatExpression;
    template<Operation, matrix, req::number> friend class detail::MatScalarExpression;
    template<Operation, req::number, matrix> friend class detail::ScalarMatExpression;
    template<matrix Mat1, matrix Mat2> requires suitable_matrix_cross_size_expression<Mat1, Mat2> friend class detail::MatrixCrossProductExpression;

    inline constexpr decltype(auto) pick(size_t index) const noexcept { return expression_operator<op>::call(m1.pick(index), m2.pick(index)); }
    inline constexpr decltype(auto) pick(size_t index) noexcept { return expression_operator<op>::call(m1.pick(index), m2.pick(index)); }
public:
    
    constexpr MatExpression(const M1& m1, const M2& m2): m1{m1}, m2{m2} {}

    inline constexpr std::pair<size_t, size_t> size() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        if constexpr(!(static_matrix<M1> && static_matrix<M2>)) {
            if(m1.size() != m2.size()) {
                throw std::length_error("size mismatch");
            }
        }
        return m1.size();
    }

    inline constexpr size_t numel() const noexcept(static_matrix<M1> && static_matrix<M2>) { 
        if constexpr(!(static_matrix<M1> && static_matrix<M2>)) {
            if(m1.numel() != m2.numel()) {
                throw std::length_error("size mismatch");
            }
        }
        return m1.numel();
    }


    inline constexpr size_t rows() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        if constexpr(!(static_matrix<M1> && static_matrix<M2>)) {
            if(m1.rows() != m2.rows()) {
                throw std::length_error("size mismatch");
            }
        }
        return m1.rows();
    }

    inline constexpr size_t cols() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        if constexpr(!(static_matrix<M1> && static_matrix<M2>)) {
            if(m1.cols() != m2.cols()) {
                throw std::length_error("size mismatch");
            }
        }
        return m1.cols();
    }

    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   noexcept { return detail::ElementsView<MatExpression, Dim>(*this); }
    inline constexpr auto rows_view()       noexcept { return detail::MatrixDimensionViewFactory<MatExpression, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       noexcept { return detail::MatrixDimensionViewFactory<MatExpression, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) noexcept { return detail::MatrixDimensionView<MatExpression, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) noexcept { return detail::MatrixDimensionView<MatExpression, MatrixDimension::BY_COLS>(*this, index); }
                                      
    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   const noexcept { return detail::ElementsView<const MatExpression, Dim>(*this); }
    inline constexpr auto rows_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatExpression, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatExpression, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) const noexcept { return detail::MatrixDimensionView<const MatExpression, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) const noexcept { return detail::MatrixDimensionView<const MatExpression, MatrixDimension::BY_COLS>(*this, index); }

    inline constexpr auto operator[](size_t index) noexcept { return row(index); }
    inline constexpr auto operator[](size_t index) const noexcept { return row(index); }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) noexcept { return (*this)[row][col]; }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) const noexcept { return (*this)[row][col]; }

    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) noexcept { return detail::MatrixView<MatExpression>(*this, rowStart, rowEnd, colStart, colEnd); }
    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) const noexcept { return detail::MatrixView<const MatExpression>(*this, rowStart, rowEnd, colStart, colEnd); }
};

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
class MatrixSingleRowColProductExpression {
public:
    inline constexpr static bool is_temporary  { true };
    inline constexpr static bool is_expression { true };
    inline constexpr static bool is_view       { false };
    using value_type = traits::common_type_t<typename M1::value_type, typename M2::value_type>;

    using operand_type_1 = M1;
    using operand_type_2 = M2;
private:
    std::conditional_t<M1::is_temporary, M1, const M1&> m1;
    std::conditional_t<M2::is_temporary, M2, const M2&> m2;
    size_t m1_row, m2_col;
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

public:
    constexpr MatrixSingleRowColProductExpression(const M1& m1, const M2& m2, size_t m1_row, size_t m2_col): m1{m1}, m2{m2}, m1_row{m1_row}, m2_col{m2_col} {}
    inline constexpr value_type get() const noexcept {
        if constexpr(dynamic_matrix<M1> || dynamic_matrix<M2>)
            if(m1.rows() != m2.cols())
                throw std::runtime_error("size mismatch");
        return std::inner_product(m1.row(m1_row).begin(), m1.row(m1_row).end(), m2.col(m2_col).begin(), static_cast<value_type>(0));
    }
    inline constexpr operator value_type() const noexcept { return get(); }
};

template<matrix M1, matrix M2, MatrixDimension Dim> requires(
    (Dim==MatrixDimension::BY_COLS && (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2> || M1::static_rows == M2::static_rows)) || 
    (Dim==MatrixDimension::BY_ROWS && (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2> || M1::static_cols == M2::static_cols))
)
class MatrixScalarProductExpression {
public:
    inline constexpr static bool is_temporary  { true };
    inline constexpr static bool is_expression { true };
    inline constexpr static bool is_view       { false };
    inline constexpr static size_t static_rows { (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2>) ? 0 : M1::static_rows };
    inline constexpr static size_t static_cols { (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2>) ? 0 : M1::static_cols };
    inline constexpr static size_t static_size { static_rows*static_cols };
    using value_type = traits::common_type_t<typename M1::value_type, typename M2::value_type>;

    using operand_type_1 = M1;
    using operand_type_2 = M2;
private:
    std::conditional_t<M1::is_temporary, M1, const M1&> m1;
    std::conditional_t<M2::is_temporary, M2, const M2&> m2;
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    using iterator = detail::linear_element_iterator<MatrixScalarProductExpression>;

    inline constexpr decltype(auto) pick(size_t index) const noexcept { 
        if constexpr(Dim==MatrixDimension::BY_ROWS) {
            return MatrixSingleRowColProductExpression(m1, transpose(m2), index, index);
        } else if constexpr(Dim==MatrixDimension::BY_COLS) {
            return MatrixSingleRowColProductExpression(transpose(m1), m2, index, index);
        }
    }
    inline constexpr decltype(auto) pick(size_t index) noexcept { 
        if constexpr(Dim==MatrixDimension::BY_ROWS) {
            return MatrixSingleRowColProductExpression(m1, transpose(m2), index, index);
        } else if constexpr(Dim==MatrixDimension::BY_COLS) {
            return MatrixSingleRowColProductExpression(transpose(m1), m2, index, index);
        }
    }
public:
    constexpr MatrixScalarProductExpression(const M1& m1, const M2& m2): m1{m1}, m2{m2} {}

    inline constexpr decltype(auto) operator[](ssize_t index) const noexcept { return pick(index); }
    inline constexpr decltype(auto) operator[](ssize_t index) noexcept { return pick(index); }
    inline constexpr decltype(auto) at(size_t index) { return index > numel() ? throw std::out_of_range() : operator[](index); }
    inline constexpr decltype(auto) at(size_t index) const { return index > numel() ? throw std::out_of_range() : operator[](index); }

    inline constexpr std::pair<size_t, size_t> size() const noexcept { return numel(); }
    inline constexpr size_t numel() const noexcept {
        if constexpr(Dim==MatrixDimension::BY_ROWS) {
            if constexpr(dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2>) {
                if(m1.cols() != m2.cols()) {
                    throw std::runtime_error("MatrixScalarProductExpression: size mismatch");
                }
                return m1.cols();
            } else {
                static_assert(M1::static_cols == M2::static_cols);
                return M1::static_cols;
            }
        } else if constexpr(Dim==MatrixDimension::BY_COLS) {
            if constexpr(dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2>) {
                if(m1.rows() != m2.rows()) {
                    throw std::runtime_error("MatrixScalarProductExpression: size mismatch");
                }
                return m1.rows();
            } else {
                static_assert(M1::static_rows == M2::static_rows);
                return M1::static_rows;
            }
        }
    }

    inline constexpr auto begin() const { return iterator(*this, 0); }
    inline constexpr auto cbegin() const { return iterator(*this, 0); }
    inline constexpr auto rbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto crbegin() const { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() const { return iterator(*this, numel()); }
    inline constexpr auto cend() const { return iterator(*this, numel()); }
    inline constexpr auto rend() const { return std::reverse_iterator(iterator(*this, numel())); }
    inline constexpr auto crend() const { return std::reverse_iterator(iterator(*this, numel())); }

    inline constexpr auto begin() { return iterator(*this, 0); }
    inline constexpr auto rbegin() { return std::reverse_iterator(iterator(*this, 0)); }
    inline constexpr auto end() { return iterator(*this, numel()); }
    inline constexpr auto rend() { return std::reverse_iterator(iterator(*this, numel())); }
};

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
class MatrixCrossProductExpression {
public:
    inline constexpr static bool is_temporary  { true }; 
    inline constexpr static bool is_expression { true };
    inline constexpr static bool is_view       { false };

    using operand_type_1 = M1;
    using operand_type_2 = M2;

    inline constexpr static size_t static_rows { (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2>) ? 0 : M1::static_rows };
    inline constexpr static size_t static_cols { (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2>) ? 0 : M2::static_cols };
    inline constexpr static size_t static_size { static_rows*static_cols };
    using value_type = traits::common_type_t<typename M1::value_type, typename M2::value_type>;
private:
    std::conditional_t<M1::is_temporary, M1, M1&> m1;
    std::conditional_t<M2::is_temporary, M2, M2&> m2;
    template<typename T1, typename T2> friend constexpr bool celinalg::utils::expression_reference_check_state_invalidation(const T1* self, const T2* p);

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class detail::MatrixView<MatrixCrossProductExpression>;
    friend class detail::ElementsView<MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<MatrixCrossProductExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<MatrixCrossProductExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<MatrixCrossProductExpression, MatrixDimension::BY_COLS>;

    template<typename Element, typename index_picker> requires(container<Element> && expression_participant<Element>) friend class detail::linear_element_iterator;
    friend class detail::MatrixView<const MatrixCrossProductExpression>;
    friend class detail::ElementsView<const MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::ElementsView<const MatrixCrossProductExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionView<const MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionView<const MatrixCrossProductExpression, MatrixDimension::BY_COLS>;
    friend class detail::MatrixDimensionViewFactory<const MatrixCrossProductExpression, MatrixDimension::BY_ROWS>;
    friend class detail::MatrixDimensionViewFactory<const MatrixCrossProductExpression, MatrixDimension::BY_COLS>;

    template<Operation, matrix Mat1, matrix Mat2> requires suitable_matrix_same_size_expression<Mat1, Mat2> friend class detail::MatExpression;
    template<Operation, matrix, req::number> friend class detail::MatScalarExpression;
    template<Operation, req::number, matrix> friend class detail::ScalarMatExpression;
    template<matrix Mat1, matrix Mat2> requires suitable_matrix_cross_size_expression<Mat1, Mat2> friend class detail::MatrixCrossProductExpression;

    inline constexpr decltype(auto) pick(size_t index) const noexcept { return MatrixSingleRowColProductExpression<M1, M2>(m1, m2, index/m1.cols(), index%m2.rows()).get(); }
public:
    constexpr MatrixCrossProductExpression(M1& m1, M2& m2): m1{m1}, m2{m2} {}

    inline constexpr std::pair<size_t, size_t> size() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        if constexpr(!(static_matrix<M1> && static_matrix<M2>)) {
            if(m1.cols() != m2.rows()) {
                throw std::runtime_error("size_mismatch");
            }
        }
        return {m1.rows(), m2.cols()};
    }

    inline constexpr size_t numel() const noexcept(static_matrix<M1> && static_matrix<M2>) { 
        return rows()*cols();
    }


    inline constexpr size_t rows() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        return m1.rows();
    }

    inline constexpr size_t cols() const noexcept(static_matrix<M1> && static_matrix<M2>) {
        return m2.cols();
    }

    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   noexcept { return detail::ElementsView<MatrixCrossProductExpression, Dim>(*this); }
    inline constexpr auto rows_view()       noexcept { return detail::MatrixDimensionViewFactory<MatrixCrossProductExpression, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       noexcept { return detail::MatrixDimensionViewFactory<MatrixCrossProductExpression, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) noexcept { return detail::MatrixDimensionView<MatrixCrossProductExpression, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) noexcept { return detail::MatrixDimensionView<MatrixCrossProductExpression, MatrixDimension::BY_COLS>(*this, index); }
                                      
    template<MatrixDimension Dim = MatrixDimension::BY_ROWS> 
    inline constexpr auto elements_view()   const noexcept { return detail::ElementsView<const MatrixCrossProductExpression, Dim>(*this); }
    inline constexpr auto rows_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatrixCrossProductExpression, MatrixDimension::BY_ROWS>(*this); }
    inline constexpr auto cols_view()       const noexcept { return detail::MatrixDimensionViewFactory<const MatrixCrossProductExpression, MatrixDimension::BY_COLS>(*this); }
    inline constexpr auto row(size_t index) const noexcept { return detail::MatrixDimensionView<const MatrixCrossProductExpression, MatrixDimension::BY_ROWS>(*this, index); }
    inline constexpr auto col(size_t index) const noexcept { return detail::MatrixDimensionView<const MatrixCrossProductExpression, MatrixDimension::BY_COLS>(*this, index); }

    inline constexpr auto operator[](size_t index) noexcept { return row(index); }
    inline constexpr auto operator[](size_t index) const noexcept { return row(index); }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) noexcept { return (*this)[row][col]; }
    inline constexpr decltype(auto) operator()(size_t row, size_t col) const noexcept { return (*this)[row][col]; }

    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) noexcept { return detail::MatrixView<MatrixCrossProductExpression>(*this, rowStart, rowEnd, colStart, colEnd); }
    inline constexpr auto submatrix(uint rowStart, uint rowEnd, uint colStart, uint colEnd) const noexcept { return detail::MatrixView<const MatrixCrossProductExpression>(*this, rowStart, rowEnd, colStart, colEnd); }
};

}

template<matrix M1, matrix M2> inline auto operator+ (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::ADDITION, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator- (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::SUBTRACTION, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator* (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::MULTIPLICATION, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator/ (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::DIVISION, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator% (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::MODULO, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator&&(const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::AND, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator||(const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::OR, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator& (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::BITWISE_AND, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator| (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::BITWISE_OR, M1, M2>(m1, m2); }
template<matrix M1, matrix M2> inline auto operator^ (const M1& m1, const M2& m2) { return detail::MatExpression<detail::Operation::BITWISE_XOR, M1, M2>(m1, m2); }

template<matrix M, req::number S> inline auto operator+ (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::ADDITION, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator- (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::SUBTRACTION, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator* (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::MULTIPLICATION, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator/ (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::DIVISION, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator% (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::MODULO, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator&&(const M& m, S s) { return detail::MatScalarExpression<detail::Operation::AND, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator||(const M& m, S s) { return detail::MatScalarExpression<detail::Operation::OR, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator& (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::BITWISE_AND, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator| (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::BITWISE_OR, M, S>(m, s); }
template<matrix M, req::number S> inline auto operator^ (const M& m, S s) { return detail::MatScalarExpression<detail::Operation::BITWISE_XOR, M, S>(m, s); }

template<req::number S, matrix M> inline auto operator+ (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::ADDITION, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator- (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::SUBTRACTION, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator* (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::MULTIPLICATION, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator/ (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::DIVISION, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator% (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::MODULO, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator&&(S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::AND, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator||(S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::OR, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator& (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::BITWISE_AND, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator| (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::BITWISE_OR, S, M>(s, m); }
template<req::number S, matrix M> inline auto operator^ (S s, const M& m) { return detail::ScalarMatExpression<detail::Operation::BITWISE_XOR, S, M>(s, m); }

template<matrix M1, matrix M2> requires suitable_matrix_cross_size_expression<M1, M2>
inline constexpr auto cprod(const M1& m1, const M2& m2) noexcept { return detail::MatrixCrossProductExpression<const M1, const M2>(m1, m2); }

template<MatrixDimension Dim, matrix M1, matrix M2> requires(
    (Dim==MatrixDimension::BY_COLS && (dynamic_rows_matrix<M1> || dynamic_rows_matrix<M2> || M1::static_rows == M2::static_rows)) || 
    (Dim==MatrixDimension::BY_ROWS && (dynamic_cols_matrix<M1> || dynamic_cols_matrix<M2> || M1::static_cols == M2::static_cols))
)
inline constexpr auto sprod(const M1& m1, const M2& m2) noexcept { return detail::MatrixScalarProductExpression<const M1, const M2, Dim>(m1, m2); }

}