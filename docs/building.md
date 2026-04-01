# Building gonx

## Prerequisites

- **CMake** 3.21+
- **Ninja** (recommended generator)
- **C++20 compiler**: GCC 12+, Clang 15+, or MSVC 2022+
- **[uv](https://docs.astral.sh/uv/)** (for generating test fixtures)

The build system automatically downloads ONNX Runtime pre-built packages from
GitHub releases. No manual ORT installation is required.

## Quick Start

```bash
# Configure (debug build with tests and sanitizers)
cmake --preset debug

# Build
cmake --build --preset debug --parallel

# Run tests
cd build/debug && ctest --output-on-failure
```

## Available Presets

| Preset           | Build Type     | Tests | Sanitizers | Warnings-as-Errors |
|------------------|----------------|-------|------------|---------------------|
| `debug`          | Debug          | Yes   | Yes        | No                  |
| `relwithdebinfo` | RelWithDebInfo | Yes   | No         | No                  |
| `release`        | Release (LTO)  | No    | No         | Yes                 |
| `ci-linux`       | RelWithDebInfo | Yes   | No         | Yes                 |
| `ci-windows`     | RelWithDebInfo | Yes   | No         | Yes                 |
| `ci-macos`       | RelWithDebInfo | Yes   | No         | Yes                 |

## Configuration Options

| CMake Variable            | Default | Description                               |
|---------------------------|---------|-------------------------------------------|
| `GONX_WARNINGS_AS_ERRORS` | OFF     | Treat compiler warnings as errors         |
| `GONX_ENABLE_SANITIZERS`  | OFF     | Enable ASan + UBSan in debug builds       |
| `GONX_BUILD_TESTS`        | ON      | Build native test executables             |
| `GONX_ORT_VERSION`        | 1.24.4  | ONNX Runtime version to download          |
| `GONX_ORT_ROOT`           | (empty) | Path to pre-installed ORT (skips download)|

## Using a Pre-installed ONNX Runtime

If you have ONNX Runtime already installed:

```bash
cmake --preset debug -DGONX_ORT_ROOT=/path/to/onnxruntime
```

The directory must contain `include/` and `lib/` subdirectories.

## Generating Test Fixtures

Tests use small deterministic ONNX models. Generate them before running tests:

```bash
uv run --with onnx --with numpy python3 tests/fixtures/generate_test_models.py
```

Tests will `SKIP` gracefully if fixtures are missing.

## Installing to the Example Project

After building, copy the shared library and ORT runtime library into the
example project's addon directory:

```bash
cp build/debug/libgonx.so example/addons/gonx/bin/
cp build/debug/libonnxruntime* example/addons/gonx/bin/
```

Then open the example project in Godot 4.6.

## Cross-platform Notes

### Windows
Use the Visual Studio Developer Command Prompt or ensure `cl.exe` is on PATH.
The build produces `gonx.dll` and expects `onnxruntime.dll` beside it.

### macOS
The build produces `libgonx.dylib`. Ensure `libonnxruntime.dylib` is in the
same directory or in the addon bin/ path.

### Linux
The build produces `libgonx.so`. ONNX Runtime `.so` files (including versioned
symlinks) must be in the same directory or on `LD_LIBRARY_PATH`.
