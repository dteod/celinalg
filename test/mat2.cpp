#include <iostream> 
#include <celinalg/matrix.hpp>

int main() {
    using cd = std::complex<double>;
    celinalg::Matrix<cd, 3, 3> m {{
        {cd(1, 1), cd(2, 3), cd(0, 5)},
        {cd(0, 10), cd(-3.1, 3), cd(-1.2, 2)},
        {cd(10, 0), cd(-1, -43.6), cd(0, 5)}
    }};
    std::cout << m[0][0] << std::endl;
    std::cout << m[0][1] << std::endl;
    std::cout << m[0][2] << std::endl;
    std::cout << m[1][0] << std::endl;
    std::cout << m[1][1] << std::endl;
    std::cout << m[1][2] << std::endl;
    std::cout << m[2][0] << std::endl;
    std::cout << m[2][1] << std::endl;
    std::cout << m[2][2] << std::endl;
}