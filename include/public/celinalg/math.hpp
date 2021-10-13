#pragma once

#include <cmath>
#include <type_traits>
#include <numbers>

#if defined(CELINALG_USE_GCEM) && CELINALG_USE_GCEM
#include <gcem.hpp>

#define MATH_HPP_DECLARE_GCEM_FUNCTION(NAME)                                 \
inline constexpr auto NAME(auto&&... args) noexcept {                   \
    if(std::is_constant_evaluated()) {                                  \
        return ::gcem::NAME(std::forward<decltype(args)>(args)...);   \
    } else {                                                            \
        return ::std::NAME(std::forward<decltype(args)>(args)...);      \
    }                                                                   \
}
#else
#define MATH_HPP_DECLARE_GCEM_FUNCTION(NAME) MATH_HPP_DECLARE_STD_FUNCTION(NAME)
#endif

#define MATH_HPP_DECLARE_STD_FUNCTION(NAME)                 \
inline auto NAME(auto&&... args) noexcept {                     \
    return ::std::NAME(std::forward<decltype(args)>(args)...);  \
}


namespace math::ct {
    consteval auto abs(auto x) noexcept { 
        return x > 0 ? x : -x;
    }

    consteval float fabsf(float x) noexcept { return math::ct::abs(x); }
    consteval double fabs(double x) noexcept { return math::ct::abs(x); }
    consteval long double fabsl(long double x) noexcept { return math::ct::abs(x); }

    consteval auto trunc(auto x) noexcept {
        if(x > 0) {
            return static_cast<decltype(x)>(static_cast<int64_t>(x));
        } else if(x < 0) {
            return math::ct::trunc(-x);
        } else return std::decay_t<decltype(x)>(0);
    }

    consteval float truncf(float x) noexcept {
        return math::ct::trunc(x);
    }
    consteval long double truncl(long double x) noexcept {
        return math::ct::trunc(x);
    }

    consteval float fmod(float x, float y) noexcept {
        auto quotient = math::ct::trunc(x/y);
        return x - quotient*y;
    }

    consteval auto remainder(auto x, auto y) noexcept {
        auto quotient = static_cast<int>(x/y);
        auto rem = x - quotient*y;
        if(abs(rem) > 0.5)
            quotient += 1;
        else if(abs(rem) == 0.5)
            quotient = (quotient%2)? quotient : quotient + 1;
        return x - quotient*y;
    }

    consteval float remainderf(float x, float y) noexcept {
        return remainder(x, y);
    }

    consteval long double remainderl(long double x, long double y) noexcept {
        return remainder(x, y);
    }
}


namespace math {
    MATH_HPP_DECLARE_STD_FUNCTION(fabs)
    MATH_HPP_DECLARE_STD_FUNCTION(remainder)
    MATH_HPP_DECLARE_STD_FUNCTION(remainderf)
    MATH_HPP_DECLARE_STD_FUNCTION(remainderl)
    MATH_HPP_DECLARE_STD_FUNCTION(remquo)
    MATH_HPP_DECLARE_STD_FUNCTION(remquof)
    MATH_HPP_DECLARE_STD_FUNCTION(remquol)
    MATH_HPP_DECLARE_STD_FUNCTION(fma)
    MATH_HPP_DECLARE_STD_FUNCTION(fmaf)
    MATH_HPP_DECLARE_STD_FUNCTION(fmal)
    MATH_HPP_DECLARE_STD_FUNCTION(fmax)
    MATH_HPP_DECLARE_STD_FUNCTION(fmaxf)
    MATH_HPP_DECLARE_STD_FUNCTION(fmaxl)
    MATH_HPP_DECLARE_STD_FUNCTION(fmin)
    MATH_HPP_DECLARE_STD_FUNCTION(fminf)
    MATH_HPP_DECLARE_STD_FUNCTION(fminl)
    MATH_HPP_DECLARE_STD_FUNCTION(fdim)
    MATH_HPP_DECLARE_STD_FUNCTION(fdimf)
    MATH_HPP_DECLARE_STD_FUNCTION(fdiml)
    MATH_HPP_DECLARE_STD_FUNCTION(nan)
    MATH_HPP_DECLARE_STD_FUNCTION(nanf)
    MATH_HPP_DECLARE_STD_FUNCTION(nanl)
    MATH_HPP_DECLARE_STD_FUNCTION(lerp)
    MATH_HPP_DECLARE_STD_FUNCTION(exp2)
    MATH_HPP_DECLARE_STD_FUNCTION(exp2f)
    MATH_HPP_DECLARE_STD_FUNCTION(exp2l)
    MATH_HPP_DECLARE_STD_FUNCTION(expm1f)
    MATH_HPP_DECLARE_STD_FUNCTION(expm1l)
    MATH_HPP_DECLARE_STD_FUNCTION(log10)
    MATH_HPP_DECLARE_STD_FUNCTION(log2f)
    MATH_HPP_DECLARE_STD_FUNCTION(log2l)
    MATH_HPP_DECLARE_STD_FUNCTION(log1pf)
    MATH_HPP_DECLARE_STD_FUNCTION(log1pl)
    MATH_HPP_DECLARE_STD_FUNCTION(cbrt)
    MATH_HPP_DECLARE_STD_FUNCTION(cbrtf)
    MATH_HPP_DECLARE_STD_FUNCTION(cbrtl)
    MATH_HPP_DECLARE_STD_FUNCTION(hypot)
    MATH_HPP_DECLARE_STD_FUNCTION(hypotf)
    MATH_HPP_DECLARE_STD_FUNCTION(hypotl)
    MATH_HPP_DECLARE_STD_FUNCTION(asinhf)
    MATH_HPP_DECLARE_STD_FUNCTION(asinhl)
    MATH_HPP_DECLARE_STD_FUNCTION(acoshl)
    MATH_HPP_DECLARE_STD_FUNCTION(acoshf)
    MATH_HPP_DECLARE_STD_FUNCTION(atanhl)
    MATH_HPP_DECLARE_STD_FUNCTION(atanhf)
    MATH_HPP_DECLARE_STD_FUNCTION(erff)
    MATH_HPP_DECLARE_STD_FUNCTION(erfl)
    MATH_HPP_DECLARE_STD_FUNCTION(erfc)
    MATH_HPP_DECLARE_STD_FUNCTION(erfcf)
    MATH_HPP_DECLARE_STD_FUNCTION(erfcl)
    MATH_HPP_DECLARE_STD_FUNCTION(tgammaf)
    MATH_HPP_DECLARE_STD_FUNCTION(tgammal)
    MATH_HPP_DECLARE_STD_FUNCTION(lgammaf)
    MATH_HPP_DECLARE_STD_FUNCTION(lgammal)
    MATH_HPP_DECLARE_STD_FUNCTION(truncf)
    MATH_HPP_DECLARE_STD_FUNCTION(truncl)
    MATH_HPP_DECLARE_STD_FUNCTION(roundf)
    MATH_HPP_DECLARE_STD_FUNCTION(roundl)
    MATH_HPP_DECLARE_STD_FUNCTION(lround)
    MATH_HPP_DECLARE_STD_FUNCTION(lroundf)
    MATH_HPP_DECLARE_STD_FUNCTION(lroundl)
    MATH_HPP_DECLARE_STD_FUNCTION(llround)
    MATH_HPP_DECLARE_STD_FUNCTION(llroundf)
    MATH_HPP_DECLARE_STD_FUNCTION(llroundl)
    MATH_HPP_DECLARE_STD_FUNCTION(nearbyint)
    MATH_HPP_DECLARE_STD_FUNCTION(nearbyintf)
    MATH_HPP_DECLARE_STD_FUNCTION(nearbyintl)
    MATH_HPP_DECLARE_STD_FUNCTION(rint)
    MATH_HPP_DECLARE_STD_FUNCTION(rintf)
    MATH_HPP_DECLARE_STD_FUNCTION(rintl)
    MATH_HPP_DECLARE_STD_FUNCTION(lrint)
    MATH_HPP_DECLARE_STD_FUNCTION(lrintf)
    MATH_HPP_DECLARE_STD_FUNCTION(lrintl)
    MATH_HPP_DECLARE_STD_FUNCTION(llrint)
    MATH_HPP_DECLARE_STD_FUNCTION(llrintf)
    MATH_HPP_DECLARE_STD_FUNCTION(llrintl)
    MATH_HPP_DECLARE_STD_FUNCTION(frexp)
    MATH_HPP_DECLARE_STD_FUNCTION(ldexp)
    MATH_HPP_DECLARE_STD_FUNCTION(modf)
    MATH_HPP_DECLARE_STD_FUNCTION(scalbn)
    MATH_HPP_DECLARE_STD_FUNCTION(scalbnf)
    MATH_HPP_DECLARE_STD_FUNCTION(scalbnl)
    MATH_HPP_DECLARE_STD_FUNCTION(scalbln)
    MATH_HPP_DECLARE_STD_FUNCTION(scalblnf)
    MATH_HPP_DECLARE_STD_FUNCTION(scalblnl)
    MATH_HPP_DECLARE_STD_FUNCTION(ilogb)
    MATH_HPP_DECLARE_STD_FUNCTION(ilogbf)
    MATH_HPP_DECLARE_STD_FUNCTION(ilogbl)
    MATH_HPP_DECLARE_STD_FUNCTION(logb)
    MATH_HPP_DECLARE_STD_FUNCTION(logbf)
    MATH_HPP_DECLARE_STD_FUNCTION(logbl)
    MATH_HPP_DECLARE_STD_FUNCTION(nextafter)
    MATH_HPP_DECLARE_STD_FUNCTION(nextafterf)
    MATH_HPP_DECLARE_STD_FUNCTION(nextafterl)
    MATH_HPP_DECLARE_STD_FUNCTION(nexttoward)
    MATH_HPP_DECLARE_STD_FUNCTION(nexttowardf)
    MATH_HPP_DECLARE_STD_FUNCTION(nexttowardl)
    MATH_HPP_DECLARE_STD_FUNCTION(copysignf)
    MATH_HPP_DECLARE_STD_FUNCTION(copysignl)
    MATH_HPP_DECLARE_STD_FUNCTION(fpclassify)
    MATH_HPP_DECLARE_STD_FUNCTION(isfinite)
    MATH_HPP_DECLARE_STD_FUNCTION(isinf)
    MATH_HPP_DECLARE_STD_FUNCTION(isnan)
    MATH_HPP_DECLARE_STD_FUNCTION(isnormal)
    MATH_HPP_DECLARE_STD_FUNCTION(isgreater)
    MATH_HPP_DECLARE_STD_FUNCTION(isgreaterequal)
    MATH_HPP_DECLARE_STD_FUNCTION(isless)
    MATH_HPP_DECLARE_STD_FUNCTION(islessequal)
    MATH_HPP_DECLARE_STD_FUNCTION(islessgreater)
    MATH_HPP_DECLARE_STD_FUNCTION(isunordered)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_laguerre)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_laguerref)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_laguerrel)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_legendre)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_legendref)
    MATH_HPP_DECLARE_STD_FUNCTION(assoc_legendrel)
    MATH_HPP_DECLARE_STD_FUNCTION(betaf)
    MATH_HPP_DECLARE_STD_FUNCTION(betal)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_1)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_1f)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_1l)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_2)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_2f)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_2l)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_3)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_3f)
    MATH_HPP_DECLARE_STD_FUNCTION(comp_ellint_3l)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_i)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_if)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_il)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_j)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_jf)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_jl)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_k)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_kf)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_bessel_kl)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_neumann)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_neumannf)
    MATH_HPP_DECLARE_STD_FUNCTION(cyl_neumannl)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_1)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_1f)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_1l)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_2)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_2f)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_2l)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_3)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_3f)
    MATH_HPP_DECLARE_STD_FUNCTION(ellint_3l)
    MATH_HPP_DECLARE_STD_FUNCTION(expint)
    MATH_HPP_DECLARE_STD_FUNCTION(expintf)
    MATH_HPP_DECLARE_STD_FUNCTION(expintl)
    MATH_HPP_DECLARE_STD_FUNCTION(hermite)
    MATH_HPP_DECLARE_STD_FUNCTION(hermitef)
    MATH_HPP_DECLARE_STD_FUNCTION(hermitel)
    MATH_HPP_DECLARE_STD_FUNCTION(legendre)
    MATH_HPP_DECLARE_STD_FUNCTION(legendref)
    MATH_HPP_DECLARE_STD_FUNCTION(legendrel)
    MATH_HPP_DECLARE_STD_FUNCTION(laguerre)
    MATH_HPP_DECLARE_STD_FUNCTION(laguerref)
    MATH_HPP_DECLARE_STD_FUNCTION(laguerrel)
    MATH_HPP_DECLARE_STD_FUNCTION(riemann_zeta)
    MATH_HPP_DECLARE_STD_FUNCTION(riemann_zetaf)
    MATH_HPP_DECLARE_STD_FUNCTION(riemann_zetal)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_bessel)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_besself)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_bessell)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_legendre)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_legendref)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_legendrel)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_neumann)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_neumannf)
    MATH_HPP_DECLARE_STD_FUNCTION(sph_neumannl)

    MATH_HPP_DECLARE_GCEM_FUNCTION(abs);
    MATH_HPP_DECLARE_GCEM_FUNCTION(fmod);
    MATH_HPP_DECLARE_GCEM_FUNCTION(exp);
    MATH_HPP_DECLARE_GCEM_FUNCTION(expm1);
    MATH_HPP_DECLARE_GCEM_FUNCTION(log);
    MATH_HPP_DECLARE_GCEM_FUNCTION(log2);
    MATH_HPP_DECLARE_GCEM_FUNCTION(log1p);
    MATH_HPP_DECLARE_GCEM_FUNCTION(pow);
    MATH_HPP_DECLARE_GCEM_FUNCTION(sqrt);
    MATH_HPP_DECLARE_GCEM_FUNCTION(sin);
    MATH_HPP_DECLARE_GCEM_FUNCTION(cos);
    MATH_HPP_DECLARE_GCEM_FUNCTION(tan);
    MATH_HPP_DECLARE_GCEM_FUNCTION(asin);
    MATH_HPP_DECLARE_GCEM_FUNCTION(acos);
    MATH_HPP_DECLARE_GCEM_FUNCTION(atan);
    MATH_HPP_DECLARE_GCEM_FUNCTION(atan2);
    MATH_HPP_DECLARE_GCEM_FUNCTION(sinh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(cosh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(tanh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(asinh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(acosh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(atanh);
    MATH_HPP_DECLARE_GCEM_FUNCTION(erf);
    MATH_HPP_DECLARE_GCEM_FUNCTION(tgamma);
    MATH_HPP_DECLARE_GCEM_FUNCTION(lgamma);
    MATH_HPP_DECLARE_GCEM_FUNCTION(ceil);
    MATH_HPP_DECLARE_GCEM_FUNCTION(floor);
    MATH_HPP_DECLARE_GCEM_FUNCTION(trunc);
    MATH_HPP_DECLARE_GCEM_FUNCTION(round);
    MATH_HPP_DECLARE_GCEM_FUNCTION(copysign);
    MATH_HPP_DECLARE_GCEM_FUNCTION(signbit);
    MATH_HPP_DECLARE_GCEM_FUNCTION(beta);
}

namespace math {
    using namespace ::std::numbers;
}

#undef MATH_HPP_DECLARE_GCEM_FUNCTION
#undef MATH_HPP_DECLARE_STD_FUNCTION