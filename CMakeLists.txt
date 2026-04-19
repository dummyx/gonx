cmake_minimum_required(VERSION 3.21)

project(gonx
    VERSION 0.1.0
    DESCRIPTION "Godot GDExtension for ONNX Runtime inference"
    LANGUAGES CXX
)

# ── Options ──────────────────────────────────────────────────────────────────
option(GONX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(GONX_ENABLE_SANITIZERS "Enable ASan + UBSan in debug builds" OFF)
option(GONX_BUILD_TESTS "Build native tests" ON)
option(GONX_BUILD_GDEXTENSION "Build the Godot-facing GDExtension shared library" ON)

# ── CMake modules ────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GonxCompilerFlags)

# Export compile_commands.json for IDE/tooling support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ─────────────────────────────────────────────────────────────

# ONNX Runtime (pre-built download or user-provided)
include(OnnxRuntime)

# ── Library targets ──────────────────────────────────────────────────────────

set(GONX_CORE_SOURCES
    # Core layer
    src/core/environment.cpp
    src/core/session.cpp
    src/core/provider.cpp
    src/core/type_conversion.cpp
    src/core/tensor_spec.cpp
    src/platform/platform.cpp
)

set(GONX_GODOT_SOURCES

    # Godot boundary layer
    src/godot/ort_session_node.cpp
    src/godot/ort_tensor_spec.cpp
    src/godot/ort_model_metadata.cpp
    src/godot/ort_provider_config.cpp
    src/godot/register_types.cpp
)

add_library(gonx-core STATIC ${GONX_CORE_SOURCES})
add_library(gonx::core ALIAS gonx-core)

target_include_directories(gonx-core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(gonx-core
    PUBLIC
        onnxruntime::onnxruntime
)

gonx_set_compiler_flags(gonx-core)
set_target_properties(gonx-core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(GONX_BUILD_GDEXTENSION)
    if(TARGET godot-cpp)
        set(GONX_GODOT_CPP_TARGET godot-cpp)
    elseif(TARGET godot::cpp)
        set(GONX_GODOT_CPP_TARGET godot::cpp)
    else()
        set(GODOT_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/godot-cpp")
        if(NOT EXISTS "${GODOT_CPP_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "godot-cpp not found at ${GODOT_CPP_DIR}. "
                "Make sure the thirdparty/godot-cpp directory is populated.")
        endif()
        add_subdirectory("${GODOT_CPP_DIR}" godot-cpp EXCLUDE_FROM_ALL)
        set(GONX_GODOT_CPP_TARGET godot::cpp)
    endif()

    add_library(gonx SHARED ${GONX_GODOT_SOURCES})
    add_library(gonx::godot ALIAS gonx)

    target_include_directories(gonx
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )

    target_link_libraries(gonx
        PUBLIC
            ${GONX_GODOT_CPP_TARGET}
        PRIVATE
            gonx-core
    )

    # Suppress warnings from godot-cpp headers by treating them as system includes
    get_target_property(_godot_inc ${GONX_GODOT_CPP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if(_godot_inc)
        target_include_directories(gonx SYSTEM PUBLIC ${_godot_inc})
    endif()

    gonx_set_compiler_flags(gonx)

    # ── Output naming ────────────────────────────────────────────────────────
    # Follow godot-cpp naming conventions for the shared library.
    # The .gdextension manifest references these names.

    set_target_properties(gonx PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "gonx"
        DEBUG_POSTFIX ""
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_target_properties(gonx PROPERTIES
            BUILD_RPATH "\$ORIGIN"
            INSTALL_RPATH "\$ORIGIN"
            BUILD_WITH_INSTALL_RPATH ON
            BUILD_RPATH_USE_ORIGIN ON
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set_target_properties(gonx PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()

    # Platform-specific suffix handling
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set_target_properties(gonx PROPERTIES
            PREFIX ""
            SUFFIX ".dll"
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set_target_properties(gonx PROPERTIES
            SUFFIX ".dylib"
        )
    endif()

    # ── Post-build: copy ORT runtime library next to our output ──────────────
    get_target_property(_ort_location onnxruntime::onnxruntime IMPORTED_LOCATION)
    if(_ort_location)
        add_custom_command(TARGET gonx POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ort_location}"
                "$<TARGET_FILE_DIR:gonx>"
            COMMENT "Copying ONNX Runtime shared library to output directory"
        )
        # On Linux, also copy versioned symlinks
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            get_filename_component(_ort_dir "${_ort_location}" DIRECTORY)
            file(GLOB _ort_symlinks "${_ort_dir}/libonnxruntime.so*")
            foreach(_sym ${_ort_symlinks})
                add_custom_command(TARGET gonx POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_sym}"
                        "$<TARGET_FILE_DIR:gonx>"
                    COMMENT "Copying ${_sym}"
                )
            endforeach()
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            get_filename_component(_ort_dir "${_ort_location}" DIRECTORY)
            file(GLOB _ort_symlinks LIST_DIRECTORIES false "${_ort_dir}/libonnxruntime*.dylib")
            foreach(_sym ${_ort_symlinks})
                add_custom_command(TARGET gonx POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_sym}"
                        "$<TARGET_FILE_DIR:gonx>"
                    COMMENT "Copying ${_sym}"
                )
            endforeach()
        endif()
    endif()

    # ── Install ──────────────────────────────────────────────────────────────
    # Install into an addon-ready layout
    set(GONX_ADDON_DIR "addons/gonx")

    install(TARGETS gonx
        LIBRARY DESTINATION "${GONX_ADDON_DIR}/bin"
        RUNTIME DESTINATION "${GONX_ADDON_DIR}/bin"
    )

    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/example/addons/gonx/gonx.gdextension"
        DESTINATION "${GONX_ADDON_DIR}"
    )

    # Install ORT runtime libs
    if(GONX_ORT_LIB_DIR)
        if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            install(FILES "${_ort_location}" DESTINATION "${GONX_ADDON_DIR}/bin")
        else()
            file(GLOB _ort_libs LIST_DIRECTORIES false "${GONX_ORT_LIB_DIR}/libonnxruntime*")
            install(FILES ${_ort_libs} DESTINATION "${GONX_ADDON_DIR}/bin")
        endif()
    endif()
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(GONX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
