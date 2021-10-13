#pragma once

#include <concepts>
#include <complex>

namespace celinalg::req {

template<typename T> 
concept signed_integer = std::signed_integral<T>;

template<typename T> 
concept unsigned_integer = std::unsigned_integral<T>;

template<typename T> 
concept integer = signed_integer<T> || unsigned_integer<T>;

template<typename T> 
concept floating_point = std::floating_point<T>;

template<typename T> 
concept complex = requires { typename T::value_type; } && std::same_as<std::complex<typename T::value_type>, T>;

template<typename T> 
concept number = integer<T> || floating_point<T> || complex<T>;

}