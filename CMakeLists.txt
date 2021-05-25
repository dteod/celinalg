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

target_compile_definitions(linalg PUBLIC
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_VERSION:12.0.0>>:LINALG_USE_CONSTEXPR_VECTOR_COMPILER=1>
)
target_compile_options(linalg PUBLIC -fconcepts-diagnostics-depth=5)
target_link_libraries(linalg PUBLIC utils)

create_test(vector)
create_test(matrix)
create_test(mat)