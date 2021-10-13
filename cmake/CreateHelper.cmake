function(create_helper file_path_no_ext)
get_filename_component(TARGET_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
add_executable(${TARGET_NAME}_${file_path_no_ext}_helper ${CMAKE_CURRENT_SOURCE_DIR}/helpers/${file_path_no_ext}.cpp)
target_link_libraries(${TARGET_NAME}_${file_path_no_ext}_helper PRIVATE ${TARGET_NAME})
target_hard_compilation(${TARGET_NAME}_${file_path_no_ext}_helper PUBLIC)
endfunction()