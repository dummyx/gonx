# Tests for gonx
# Uses Catch2 for native unit/integration tests.

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# Fixture directory
set(GONX_TEST_FIXTURES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# ── Unit tests ───────────────────────────────────────────────────────────────
add_executable(gonx_unit_tests
    unit/test_error.cpp
    unit/test_tensor_spec.cpp
    unit/test_provider.cpp
    unit/test_type_conversion.cpp
)

target_link_libraries(gonx_unit_tests PRIVATE
    Catch2::Catch2WithMain
    onnxruntime::onnxruntime
)

target_include_directories(gonx_unit_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

# Link against gonx core sources directly (avoid needing Godot for unit tests)
target_sources(gonx_unit_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src/core/tensor_spec.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/provider.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/type_conversion.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/environment.cpp"
)

target_compile_features(gonx_unit_tests PRIVATE cxx_std_20)
set_target_properties(gonx_unit_tests PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(gonx_unit_tests PRIVATE
    GONX_TEST_FIXTURES_DIR="${GONX_TEST_FIXTURES_DIR}"
)

catch_discover_tests(gonx_unit_tests)

# ── Integration tests ────────────────────────────────────────────────────────
add_executable(gonx_integration_tests
    integration/test_session.cpp
    integration/test_inference.cpp
)

target_link_libraries(gonx_integration_tests PRIVATE
    Catch2::Catch2WithMain
    onnxruntime::onnxruntime
)

target_include_directories(gonx_integration_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

target_sources(gonx_integration_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src/core/tensor_spec.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/provider.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/type_conversion.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/environment.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/session.cpp"
)

target_compile_features(gonx_integration_tests PRIVATE cxx_std_20)
set_target_properties(gonx_integration_tests PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(gonx_integration_tests PRIVATE
    GONX_TEST_FIXTURES_DIR="${GONX_TEST_FIXTURES_DIR}"
)

catch_discover_tests(gonx_integration_tests)
