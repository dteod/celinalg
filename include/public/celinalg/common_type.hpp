#pragma once

#include "celinalg/number.hpp"
#include "celinalg/number_type.hpp"

namespace celinalg::traits {

namespace detail {

template<req::number T1, req::number T2>
struct common_type {
private:
    constexpr static auto helper() noexcept {
        if constexpr(req::complex<T1> || req::complex<T2>) {
            return static_cast<std::complex<typename common_type<number_type_t<T1>, number_type_t<T2>>::type>>(0);
        } else if constexpr(std::same_as<T1, double> || std::same_as<T2, double>) {
            return static_cast<double>(0);
        } else if constexpr(std::same_as<T1, float> || std::same_as<T2, float>) {
            return static_cast<float>(0);
        } else if constexpr(std::same_as<T1, __int128_t> || std::same_as<T2, __int128_t>) {
            return static_cast<__int128_t>(0);
        } else if constexpr(std::same_as<T1, __uint128_t> || std::same_as<T2, __uint128_t>) {
            return static_cast<__uint128_t>(0);
        } else if constexpr(std::same_as<T1, int64_t> || std::same_as<T2, int64_t>) {
            return static_cast<int64_t>(0);
        } else if constexpr(std::same_as<T1, uint64_t> || std::same_as<T2, uint64_t>) {
            return static_cast<uint64_t>(0);
        } else if constexpr(std::same_as<T1, int32_t> || std::same_as<T2, int32_t>) {
            return static_cast<int32_t>(0);
        } else if constexpr(std::same_as<T1, uint32_t> || std::same_as<T2, uint32_t>) {
            return static_cast<uint32_t>(0);
        } else if constexpr(std::same_as<T1, int16_t> || std::same_as<T2, int16_t>) {
            return static_cast<int16_t>(0);
        } else if constexpr(std::same_as<T1, uint16_t> || std::same_as<T2, uint16_t>) {
            return static_cast<uint16_t>(0);
        } else if constexpr(std::same_as<T1, int8_t> || std::same_as<T2, int8_t>) {
            return static_cast<int8_t>(0);
        } else if constexpr(std::same_as<T1, uint8_t> || std::same_as<T2, uint8_t>) {
            return static_cast<uint8_t>(0);
        }
    }
public:
    using type = decltype(helper());
};

}

template<req::number T1, req::number T2, req::number... Types>
struct common_type {
    using type = typename common_type<T1, typename common_type<T2, Types...>::type>::type;
};

template<req::number T1, req::number T2>
struct common_type<T1, T2> {
    using type = typename detail::common_type<T1, T2>::type;
};

template<req::number... Ts>
using common_type_t = typename common_type<Ts...>::type;

}