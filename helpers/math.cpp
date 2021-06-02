#include <linalg/math.hpp>

#include <iostream>

template<auto X>
class Getter {
public:
    inline static constexpr auto value { X };
};

int main() {
    std::cout << Getter<math::log(5)>::value << std::endl;
}