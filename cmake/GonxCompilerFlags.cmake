# GonxCompilerFlags.cmake
# ----------------------
# Common compiler flags for the gonx project.

include_guard(GLOBAL)

function(gonx_set_compiler_flags TARGET)
    target_compile_features(${TARGET} PUBLIC cxx_std_20)
    set_target_properties(${TARGET} PROPERTIES CXX_EXTENSIONS OFF)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${TARGET} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Woverloaded-virtual
            -Wno-unused-parameter
        )
        if(GONX_WARNINGS_AS_ERRORS)
            target_compile_options(${TARGET} PRIVATE -Werror)
        endif()

        # Sanitizers for debug builds
        if(GONX_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
            target_compile_options(${TARGET} PRIVATE
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
            )
            target_link_options(${TARGET} PRIVATE
                -fsanitize=address,undefined
            )
        endif()
    elseif(MSVC)
        target_compile_options(${TARGET} PRIVATE
            /W4
            /permissive-
            /Zc:__cplusplus
            /Zc:preprocessor
            /utf-8
        )
        if(GONX_WARNINGS_AS_ERRORS)
            target_compile_options(${TARGET} PRIVATE /WX)
        endif()
    endif()
endfunction()
