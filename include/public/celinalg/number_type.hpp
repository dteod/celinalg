#pragma once

#include "celinalg/number.hpp"

namespace celinalg::traits {

// template<typename T> struct number_type;

template<req::number T>
struct number_type {
    using type = T;
};

template<req::complex T>
struct number_type<T> {
    using type = typename T::value_type;
};

template<req::number T>
using number_type_t = typename number_type<T>::type;

}