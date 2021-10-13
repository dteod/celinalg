#pragma once

#include <cstdint>
#include <limits>
#include <bit>

#include "celinalg/number.hpp"

namespace celinalg::traits {
template<size_t N> requires (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)
class unsigned_of_size;

template<> class unsigned_of_size<1UL>  { public: using type = uint8_t; };
template<> class unsigned_of_size<2UL>  { public: using type = uint16_t; };
template<> class unsigned_of_size<4UL>  { public: using type = uint32_t; };
template<> class unsigned_of_size<8UL>  { public: using type = uint64_t; };
template<> class unsigned_of_size<16UL> { public: using type = __uint128_t; };

template<size_t N> requires (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)
using unsigned_of_size_t = typename unsigned_of_size<N>::type;


template<size_t N> requires (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)
class signed_of_size;

template<> class signed_of_size<1UL>  { public: using type = int8_t; };
template<> class signed_of_size<2UL>  { public: using type = int16_t; };
template<> class signed_of_size<4UL>  { public: using type = int32_t; };
template<> class signed_of_size<8UL>  { public: using type = int64_t; };
template<> class signed_of_size<16UL> { public: using type = __int128_t; };

template<size_t N> requires (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)
using signed_of_size_t = typename signed_of_size<N>::type;


template<req::integer auto N> requires (N >= 0)
class smallest_unsigned_holder {
    constexpr static auto helper() {
        if constexpr(N <= std::numeric_limits<uint8_t>::max())
            return uint_least8_t(0);
        else if constexpr(N <= std::numeric_limits<uint16_t>::max())
            return uint_least16_t(0);
        else if constexpr(N <= std::numeric_limits<uint32_t>::max())
            return uint_least32_t(0);
        else 
            // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
            return uint_least64_t(0);
    }
public:
    using type = std::decay_t<decltype(helper())>;
};

template<req::integer auto N> requires (N >= 0)
using smallest_unsigned_holder_t = typename smallest_unsigned_holder<N>::type;


template<req::signed_integer auto N>
class smallest_signed_holder {
    constexpr static auto helper() {
        if constexpr(N >= 0 ) {
            if constexpr(N <= std::numeric_limits<int8_t>::max())
                return int_least8_t(0);
            else if constexpr(N <= std::numeric_limits<int16_t>::max())
                return int_least16_t(0);
            else if constexpr(N <= std::numeric_limits<int32_t>::max())
                return int_least32_t(0);
            else 
                // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
                return int_least64_t(0);
        } else {
            if constexpr(N >= std::numeric_limits<int8_t>::min())
                return int_least8_t(0);
            else if constexpr(N >= std::numeric_limits<int16_t>::min())
                return int_least16_t(0);
            else if constexpr(N >= std::numeric_limits<int32_t>::min())
                return int_least32_t(0);
            else 
                // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
                return int_least64_t(0);
        }
    }
public:
    using type = std::decay_t<decltype(helper())>;
};

template<req::signed_integer auto N>
using smallest_signed_holder_t = typename smallest_signed_holder<N>::type;


template<req::integer auto N> requires (N >= 0)
class fastest_unsigned_holder {
    constexpr static auto helper() {
        if constexpr(N <= std::numeric_limits<uint8_t>::max())
            return uint_fast8_t(0);
        else if constexpr(N <= std::numeric_limits<uint16_t>::max())
            return uint_fast16_t(0);
        else if constexpr(N <= std::numeric_limits<uint32_t>::max())
            return uint_fast32_t(0);
        else 
            // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
            return uint_fast64_t(0);
    }
public:
    using type = std::decay_t<decltype(helper())>;
};

template<req::integer auto N> requires (N >= 0)
using fastest_unsigned_holder_t = typename fastest_unsigned_holder<N>::type;


template<req::signed_integer auto N>
class fastest_signed_holder {
    constexpr static auto helper() {
        if constexpr(N >= 0 ) {
            if constexpr(N <= std::numeric_limits<int8_t>::max())
                return int_fast8_t(0);
            else if constexpr(N <= std::numeric_limits<int16_t>::max())
                return int_fast16_t(0);
            else if constexpr(N <= std::numeric_limits<int32_t>::max())
                return int_fast32_t(0);
            else 
                // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
                return int_fast64_t(0);
        } else {
            if constexpr(N >= std::numeric_limits<int8_t>::min())
                return int_fast8_t(0);
            else if constexpr(N >= std::numeric_limits<int16_t>::min())
                return int_fast16_t(0);
            else if constexpr(N >= std::numeric_limits<int32_t>::min())
                return int_fast32_t(0);
            else 
                // gcc support __uint128_t but other compilers don't. I don't think we will ever use them, in case, feel free to add it here
                return int_fast64_t(0);
        }
    }
public:
    using type = std::decay_t<decltype(helper())>;
};

template<req::signed_integer auto N>
using fastest_signed_holder_t = typename fastest_signed_holder<N>::type;

}