# OnnxRuntime.cmake
# -----------------
# Finds or downloads ONNX Runtime pre-built packages for the current platform.
#
# Provides the imported target: onnxruntime::onnxruntime
#
# Options:
#   GONX_ORT_VERSION    - ORT version to use (default: 1.24.4)
#   GONX_ORT_ROOT       - Path to a pre-existing ORT installation (skips download)
#   GONX_ORT_FROM_SOURCE - Build from the thirdparty/onnxruntime submodule (advanced)

include_guard(GLOBAL)

set(GONX_ORT_VERSION "1.24.4" CACHE STRING "ONNX Runtime version")
set(GONX_ORT_ROOT "" CACHE PATH "Path to pre-installed ONNX Runtime (skips download)")
option(GONX_ORT_FROM_SOURCE "Build ONNX Runtime from source submodule" OFF)

# Determine platform identifier for ORT release packages
function(_gonx_ort_platform_id OUT_VAR)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(${OUT_VAR} "linux-aarch64" PARENT_SCOPE)
        else()
            set(${OUT_VAR} "linux-x64" PARENT_SCOPE)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(${OUT_VAR} "osx-arm64" PARENT_SCOPE)
        else()
            set(${OUT_VAR} "osx-x86_64" PARENT_SCOPE)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(${OUT_VAR} "win-x64" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endfunction()

function(_gonx_ort_archive_ext OUT_VAR)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(${OUT_VAR} "zip" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "tgz" PARENT_SCOPE)
    endif()
endfunction()

# Create the imported target from a root directory containing include/ and lib/
function(_gonx_create_ort_target ORT_ROOT_DIR)
    if(TARGET onnxruntime::onnxruntime)
        return()
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)

    # Headers
    set(_ort_include "${ORT_ROOT_DIR}/include")
    if(NOT EXISTS "${_ort_include}/onnxruntime_cxx_api.h")
        # Try nested path (source tree layout)
        if(EXISTS "${ORT_ROOT_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
            set(_ort_include "${ORT_ROOT_DIR}/include/onnxruntime/core/session")
        else()
            message(FATAL_ERROR
                "Cannot find onnxruntime_cxx_api.h under ${ORT_ROOT_DIR}/include. "
                "Verify your ORT installation or GONX_ORT_ROOT path.")
        endif()
    endif()

    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_ort_include}"
    )

    # Find the shared library
    set(_ort_lib_dir "${ORT_ROOT_DIR}/lib")
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        find_file(_ort_dll
            NAMES onnxruntime.dll
            PATHS "${_ort_lib_dir}" "${ORT_ROOT_DIR}/bin"
            NO_DEFAULT_PATH
        )
        find_file(_ort_implib
            NAMES onnxruntime.lib
            PATHS "${_ort_lib_dir}"
            NO_DEFAULT_PATH
        )
        if(_ort_dll AND _ort_implib)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${_ort_dll}"
                IMPORTED_IMPLIB "${_ort_implib}"
            )
        else()
            message(FATAL_ERROR "Could not find onnxruntime.dll/lib in ${ORT_ROOT_DIR}")
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_library(_ort_dylib
            NAMES onnxruntime
            PATHS "${_ort_lib_dir}"
            NO_DEFAULT_PATH
        )
        if(_ort_dylib)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${_ort_dylib}"
            )
        else()
            message(FATAL_ERROR "Could not find libonnxruntime.dylib in ${_ort_lib_dir}")
        endif()
    else()
        find_library(_ort_so
            NAMES onnxruntime
            PATHS "${_ort_lib_dir}"
            NO_DEFAULT_PATH
        )
        if(_ort_so)
            set_target_properties(onnxruntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${_ort_so}"
            )
        else()
            message(FATAL_ERROR "Could not find libonnxruntime.so in ${_ort_lib_dir}")
        endif()
    endif()

    # Store the lib directory so we can copy runtime deps at install time
    set(GONX_ORT_LIB_DIR "${_ort_lib_dir}" CACHE INTERNAL "ORT library directory")
    set(GONX_ORT_INCLUDE_DIR "${_ort_include}" CACHE INTERNAL "ORT include directory")

    message(STATUS "ONNX Runtime target created from: ${ORT_ROOT_DIR}")
endfunction()

# Main logic
if(GONX_ORT_FROM_SOURCE)
    message(FATAL_ERROR
        "Building ONNX Runtime from source is not yet implemented. "
        "Use pre-built packages via GONX_ORT_ROOT or automatic download.")
endif()

if(GONX_ORT_ROOT)
    # User provided an existing installation
    if(NOT EXISTS "${GONX_ORT_ROOT}/include")
        message(FATAL_ERROR "GONX_ORT_ROOT=${GONX_ORT_ROOT} does not contain an include/ directory")
    endif()
    _gonx_create_ort_target("${GONX_ORT_ROOT}")
else()
    # Download pre-built package
    _gonx_ort_platform_id(_plat)
    _gonx_ort_archive_ext(_ext)

    set(_pkg_name "onnxruntime-${_plat}-${GONX_ORT_VERSION}")
    set(_download_url
        "https://github.com/microsoft/onnxruntime/releases/download/v${GONX_ORT_VERSION}/${_pkg_name}.${_ext}")
    set(_download_dir "${CMAKE_BINARY_DIR}/_deps/ort_package")

    if(NOT EXISTS "${_download_dir}/${_pkg_name}")
        message(STATUS "Downloading ONNX Runtime ${GONX_ORT_VERSION} for ${_plat}...")
        file(DOWNLOAD
            "${_download_url}"
            "${_download_dir}/${_pkg_name}.${_ext}"
            STATUS _dl_status
            SHOW_PROGRESS
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            list(GET _dl_status 1 _dl_msg)
            message(FATAL_ERROR
                "Failed to download ONNX Runtime: ${_dl_msg}\n"
                "URL: ${_download_url}\n"
                "You can manually download and set GONX_ORT_ROOT instead.")
        endif()

        # Extract
        message(STATUS "Extracting ONNX Runtime...")
        file(ARCHIVE_EXTRACT
            INPUT "${_download_dir}/${_pkg_name}.${_ext}"
            DESTINATION "${_download_dir}"
        )
    endif()

    _gonx_create_ort_target("${_download_dir}/${_pkg_name}")
endif()
