function(target_hard_compilation tgt scope)
    target_compile_options(${tgt} ${scope}
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -Wall -Wextra -Werror -pedantic -pedantic-errors>
        $<$<CXX_COMPILER_ID:MSVC>:
            /W4>
    )
endfunction()