function(create_util file_path_no_ext)
    get_filename_component(TARGET_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
    add_executable(${TARGET_NAME}_${file_path_no_ext} ${CMAKE_CURRENT_SOURCE_DIR}/utils/${file_path_no_ext}.cpp)
    target_link_libraries(${TARGET_NAME}_${file_path_no_ext} PRIVATE ${TARGET_NAME})
    target_hard_compilation(${TARGET_NAME}_${file_path_no_ext} PUBLIC)
endfunction()