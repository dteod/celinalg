cmake_minimum_required(VERSION 3.19)
project(celinalg)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TargetHardCompilation.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Testing.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CreateUtil.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CreateHelper.cmake)


add_library(celinalg
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quaternion.cpp
)
target_include_directories(celinalg
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include/public
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/public/celinalg
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/private
)
target_compile_features(celinalg PUBLIC cxx_std_20)

target_compile_definitions(celinalg PUBLIC
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_VERSION:12.0.0>>:CELINALG_USE_CONSTEXPR_VECTOR_COMPILER=1>
)
target_compile_options(celinalg PUBLIC "$<$<CONFIG:DEBUG>:-fconcepts-diagnostics-depth=5>")

if(${CMAKE_CELINALG_USE_GCEM})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/submodules/gcem)
    target_compile_definitions(celinalg PUBLIC -DCELINALG_USE_GCEM=1)
    target_link_libraries(celinalg PUBLIC gcem)
endif()

target_hard_compilation(celinalg PUBLIC)

create_test(vector)
create_test(matrix)
create_test(mat)
# create_test(mat2)