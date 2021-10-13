
include(CTest)
enable_testing()

find_package(Catch2 REQUIRED)
include(Catch)

function(create_test file_path_no_ext)
    get_filename_component(TARGET_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    add_executable(${TARGET_NAME}_${file_path_no_ext}_test ${CMAKE_CURRENT_SOURCE_DIR}/test/${file_path_no_ext}.cpp)
    target_link_libraries(${TARGET_NAME}_${file_path_no_ext}_test PRIVATE ${TARGET_NAME})
    target_link_libraries(${TARGET_NAME}_${file_path_no_ext}_test PRIVATE Catch2::Catch2)
    target_hard_compilation(${TARGET_NAME}_${file_path_no_ext}_test PUBLIC)
    add_test(NAME test_${TARGET_NAME}_${file_path_no_ext} COMMAND ${TARGET_NAME}_${file_path_no_ext}_test)
    catch_discover_tests(${TARGET_NAME}_${file_path_no_ext}_test)
endfunction()