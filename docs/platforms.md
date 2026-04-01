# Platform & Provider Support

## Supported Platforms (MVP)

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux | x86_64 | Supported |
| Linux | arm64 | Supported |
| Windows | x86_64 | Supported |
| macOS | arm64 (Apple Silicon) | Supported |

## Execution Providers

| Provider | Status | Notes |
|----------|--------|-------|
| CPU | **Required** | Always available. Default provider. |
| CUDA | Optional | Requires ORT built with CUDA EP + NVIDIA GPU + CUDA toolkit. |
| DirectML | Optional | Windows only. Requires ORT built with DML EP. |
| CoreML | Optional | macOS/iOS only. Requires ORT built with CoreML EP. |

The default pre-built ORT packages include only the CPU provider.
GPU providers require either custom ORT builds or provider-specific
ORT release packages from GitHub.

## ONNX Runtime Version

gonx pins ONNX Runtime v1.24.4. The CMake build system downloads
matching pre-built packages automatically.

## Godot Compatibility

gonx targets **Godot 4.6.x** via the godot-cpp GDExtension bindings.
The `.gdextension` manifest sets `compatibility_minimum = "4.6"`.

## Supported Tensor Types

| ONNX Type | gonx ElementType | GDScript Array Type |
|-----------|------------------|---------------------|
| float32 | `Float32` | `PackedFloat32Array` |
| int64 | `Int64` | `PackedInt64Array` |
| bool | `Bool` | `PackedByteArray` |

### Not Yet Supported

These types are architecturally planned but not in the MVP:

- float16 / bfloat16
- int8 / uint8 / int32
- string tensors
- sequence / map outputs
- image-as-tensor helpers

## Binary Size

With the default CPU-only ORT package:

| Component | Approximate Size |
|-----------|-----------------|
| `libonnxruntime.so` (Linux x64) | ~30 MB |
| `libgonx.so` | ~200 KB |

For size-sensitive deployments, a custom ORT build with reduced operators
can significantly shrink the ORT library. See the ONNX Runtime documentation
on [reduced operator builds](https://onnxruntime.ai/docs/build/custom.html).
