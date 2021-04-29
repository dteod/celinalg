cmake_minimum_required(VERSION 3.19)

add_library(linalg
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quaternion.cpp
)
target_include_directories(linalg
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include/public
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/public/linalg
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/private
)
target_compile_features(linalg PUBLIC cxx_std_20)
target_link_libraries(linalg PUBLIC utils)

add_executable(vector ${CMAKE_CURRENT_SOURCE_DIR}/helpers/vector.cpp)
target_link_libraries(vector PRIVATE linalg)
add_executable(matrix ${CMAKE_CURRENT_SOURCE_DIR}/helpers/matrix.cpp)
target_link_libraries(matrix PRIVATE linalg)