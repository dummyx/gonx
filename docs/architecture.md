# Architecture

## Layer Separation

gonx is structured in three layers, following the principle that ONNX Runtime
details should not leak into Godot-facing code, and Godot-specific types should
not appear in inference logic.

```
┌─────────────────────────────────────────────────┐
│  Godot Boundary Layer (src/godot/)              │
│  GDOrtSession, GDOrtTensorSpec, ...             │
│  Variant conversion, signals, class registration│
├─────────────────────────────────────────────────┤
│  Core Inference Layer (src/core/)               │
│  InferenceSession, TensorSpec, TypeConversion   │
│  Pure C++ with STL types, ORT C++ API           │
├─────────────────────────────────────────────────┤
│  Platform Layer (src/platform/)                 │
│  Library naming, packaging, provider loading    │
└─────────────────────────────────────────────────┘
```

### Core Layer (`include/gonx/core/`, `src/core/`)

- **environment.hpp**: Process-wide `Ort::Env` singleton. Lazily initialized,
  explicitly shut down during GDExtension teardown.
- **session.hpp**: `InferenceSession` wraps `Ort::Session` with RAII lifetime.
  Caches input/output metadata after load. Thread-safe for concurrent `run()`.
- **tensor_spec.hpp**: `TensorSpec` describes tensor name, element type, and
  shape. Pure value type, no ORT dependency.
- **provider.hpp**: `ExecutionProvider` enum and `SessionConfig` struct.
  Maps between gonx provider names and ORT provider strings.
- **type_conversion.hpp**: Centralized conversion between flat data buffers
  and `Ort::Value` tensors. Validates shape/type before creation.
- **error.hpp**: `Error`, `Result<T>`, and `Status` types for explicit error
  handling without exceptions crossing boundaries.

### Godot Layer (`include/gonx/godot/`, `src/godot/`)

Thin wrappers that convert between Godot `Variant`/typed arrays and core types:

- **GDOrtSession**: Main user-facing class. Handles Godot path resolution,
  `Dictionary` input marshaling, signal-based async, and `call_deferred`
  for thread-safe result delivery.
- **GDOrtTensorSpec**: Read-only view of `TensorSpec` as Godot properties.
- **GDOrtModelMetadata**: Read-only view of model metadata.
- **GDOrtProviderConfig**: `Resource` subclass for session options. Saveable
  and shareable across sessions.

### Platform Layer (`src/platform/`)

Reserved for OS-specific packaging and provider loading logic. Currently
minimal — the CMake build system handles most platform concerns.

## Key Design Decisions

### Single ORT Environment

One `Ort::Env` per process, initialized at GDExtension load time. ORT
recommends this pattern for session sharing and thread pool reuse.

### Session Reuse

`InferenceSession` is designed to be loaded once and called many times.
Creating a session is expensive (model parsing, graph optimization);
`run()` is the hot path. The Godot-facing `GDOrtSession` enforces this
by keeping one `InferenceSession` member.

### Thread Safety Model

- `InferenceSession::run()` is safe to call from multiple threads
  (ORT sessions support concurrent Run).
- Godot objects and SceneTree are *not* touched from worker threads.
- Async inference runs ORT on a `std::thread`, then uses `call_deferred`
  to deliver results on the main thread via signals.
- The mutex in `GDOrtSession` serializes access between sync calls and
  async preparation, but does not lock during the actual ORT `Run` on
  the worker thread.

### Error Handling

- No exception crosses the GDExtension boundary.
- Core layer uses `Result<T>` / `Status` for error propagation.
- Godot layer catches any remaining exceptions at the boundary and
  converts them to `push_error` messages + `get_last_error()` state.

### Type Conversion

All tensor creation is centralized in `type_conversion.hpp`. This is the
only place that maps between flat data buffers and `Ort::Value`. Shape and
type validation happen here before any ORT call.

## Extension Points

### GPU Providers

gonx supports multiple execution providers. CPU is always available.
CUDA, MiGraphX, OpenVINO, DirectML, and CoreML are available when ORT
is built with the corresponding provider enabled.

Provider names are parsed case-insensitively in `provider.cpp`. The
`OrtProviderConfig.provider` string maps to the internal enum, which
drives session option configuration. Adding a new provider requires:

1. Add the provider to the `ExecutionProvider` enum.
2. Update `provider.cpp` to configure session options for it.
3. Build ORT with the provider enabled (from source or pre-built package).
4. No changes needed in the Godot layer — `OrtProviderConfig.provider`
   already accepts arbitrary strings.

### Adding New Tensor Types

1. Add the type to the `ElementType` enum.
2. Add conversion functions in `type_conversion.hpp/cpp`.
3. Update `GDOrtSession::prepare_inputs` and `convert_outputs`.

### I/O Binding (GPU zero-copy)

For GPU providers, ORT supports I/O binding to avoid host↔device copies.
The architecture is ready for this: `InferenceSession` can be extended
with a `run_with_io_binding()` method without changing the Godot API.
The Godot layer would detect GPU-backed sessions and use I/O binding
automatically.
