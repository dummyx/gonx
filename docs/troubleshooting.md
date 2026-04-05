# Troubleshooting

## Build Issues

### CMake can't find godot-cpp

```
godot-cpp not found at .../thirdparty/godot-cpp
```

Ensure the `thirdparty/godot-cpp` directory is populated. If you cloned
without submodules:

```bash
git submodule update --init --recursive
```

### ONNX Runtime download fails

```
Failed to download ONNX Runtime
```

The build tries to download pre-built ORT from GitHub. If you're behind a
proxy or firewall:

1. Download the package manually from
   https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4
2. Extract it somewhere
3. Pass `-DGONX_ORT_ROOT=/path/to/extracted/package` to CMake

### Can't find onnxruntime_cxx_api.h

Verify `GONX_ORT_ROOT` points to a directory with `include/onnxruntime_cxx_api.h`.

## Runtime Issues

### Extension doesn't load in Godot

1. Check the Godot console for error messages.
2. Verify `.gdextension` library paths match your platform and build type.
3. Ensure `libonnxruntime.so` / `onnxruntime.dll` / `libonnxruntime.dylib`
   is in the same directory as the gonx library (`addons/gonx/bin/`).
4. On Linux, check `ldd` output for missing dependencies.

### "No model loaded" error

Call `load_model()` or `load_model_with_config()` before `run_inference()`.
Check `get_last_error()` if `load_model()` returns non-zero.

### Shape mismatch errors

```
Data size mismatch: expected 24 bytes (6 elements x 4 bytes), got 20 bytes
```

Your input array length doesn't match the model's expected shape. Use
`get_input_specs()` to inspect expected shapes before calling inference.

For models with dynamic dimensions (shown as -1 in shape), gonx infers
the dynamic dimension from the data length. This only works if there is
exactly one dynamic dimension per input.

### Wrong input type

```
Input 'X' expects PackedFloat32Array, got variant type 5
```

Match the GDScript array type to the model's element type:
- float32 → `PackedFloat32Array`
- int64 → `PackedInt64Array`
- bool → `PackedByteArray`

### Model loads but inference produces wrong results

1. Verify input data order: arrays are flat row-major.
2. Check preprocessing matches what the model expects (normalization, etc.).
3. Test with the same model in Python using `onnxruntime` to establish a baseline.

### Async inference never completes

- Ensure your script stays alive (the node isn't freed) until the signal fires.
- Connect signals *before* calling `run_inference_async()`.
- Check for `inference_failed` signal — it may have errored silently.

## Performance Issues

### Slow model loading

Model loading involves graph optimization. For repeated loads of the same model:

```gdscript
var config := OrtProviderConfig.new()
config.optimized_model_path = "user://model_optimized.onnx"
session.load_model_with_config("res://model.onnx", config)
```

This serializes the optimized model on first load. Subsequent loads from
the optimized path skip graph optimization.

### Slow inference

- Increase `intra_op_threads` in `OrtProviderConfig` for CPU parallelism.
- Use `optimization_level = 99` (default) for maximum graph optimization.
- Consider a GPU execution provider for large models.
- Profile with ONNX Runtime's built-in profiling if needed.

## GPU Provider Issues

### MiGraphX provider loads but inference fails

```
Failed to call function: MGXKernel_graph_main_graph_...
```

MiGraphX targets discrete AMD GPUs (MI-series, Instinct, RX 7000+).
Integrated GPUs such as the Radeon 780M (RDNA 3 iGPU) can load the
provider and compile graphs, but kernel execution fails at runtime.
Verify your GPU is a supported discrete device.

### MiGraphX missing shared libraries

```
libnuma.so.1: cannot open shared object file
```

The MiGraphX provider depends on ROCm libraries (`libmigraphx_c`,
`libamdhip64`, `libhsa-runtime64`, `libnuma`, `libdrm_amdgpu`). These
live under `/opt/rocm-*/lib` and `/opt/amdgpu/lib/`.

For a native Godot binary, set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="/opt/rocm-7.2.0/lib:/opt/amdgpu/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
```

### MiGraphX fails under Flatpak Godot

Flatpak sandboxes cannot access `/opt/rocm-*/lib`. Use a native Godot
binary instead:

```bash
LD_LIBRARY_PATH="/opt/rocm-7.2.0/lib:/opt/amdgpu/lib/x86_64-linux-gnu" \
  /path/to/Godot_v4.6.1-stable_linux.x86_64 --headless --path "$(pwd)" -s res://tools/check_extensions.gd
```

### Provider silently falls back to CPU

If the configured provider string does not match any available provider,
gonx falls back to CPU without error. Verify the provider is available:

```gdscript
print(OrtProviderConfig.get_available_providers())
```

Provider name parsing is case-insensitive: `"MiGraphX"`, `"migraphx"`,
and `"MIGRAPHX"` all resolve correctly.
