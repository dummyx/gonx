# gonx API Reference

## Classes

### OrtSession

**Inherits:** `RefCounted`

Main class for loading ONNX models and running inference.

#### Methods

| Method | Return | Description |
|--------|--------|-------------|
| `load_model(path: String)` | `int` | Load a model from a file path. Returns 0 on success, error code otherwise. Supports `res://` and `user://` paths. |
| `load_model_with_config(path: String, config: OrtProviderConfig)` | `int` | Load with explicit provider/session configuration. |
| `load_model_async(path: String)` | `int` | Start loading a model on a worker thread. Returns a request ID, or `0` if the request was rejected immediately. |
| `load_model_with_config_async(path: String, config: OrtProviderConfig)` | `int` | Async load with explicit provider/session configuration. |
| `is_loaded()` | `bool` | Whether a model is loaded and ready. |
| `is_loading()` | `bool` | Whether an async model load is currently running. |
| `get_input_specs()` | `Array[OrtTensorSpec]` | Metadata for all model inputs. |
| `get_output_specs()` | `Array[OrtTensorSpec]` | Metadata for all model outputs. |
| `get_metadata()` | `OrtModelMetadata` | Model metadata (producer, graph name, etc.). |
| `run_inference(inputs: Dictionary)` | `Dictionary` | Run synchronous inference. |
| `run_inference_async(inputs: Dictionary)` | `int` | Start async inference. Returns a request ID, or `0` if the request was rejected immediately. |
| `is_async_inference_running()` | `bool` | Whether async inference is in flight. |
| `cancel(request_id := 0)` | `void` | Cancel the active async load or async inference. `0` cancels all active async work. |
| `get_last_error()` | `String` | Error message from the last failed operation. |
| `get_model_path()` | `String` | Path of the currently loaded model. |

#### Signals

| Signal | Arguments | Description |
|--------|-----------|-------------|
| `model_load_started` | `request_id: int, path: String` | Emitted when async model load starts. |
| `model_loaded` | `request_id: int, path: String` | Emitted when async model load succeeds. |
| `model_load_failed` | `request_id: int, error_code: int, error: String` | Emitted when async model load fails. |
| `model_load_cancelled` | `request_id: int` | Emitted when async model load is cancelled. |
| `inference_completed` | `request_id: int, result: Dictionary` | Emitted when async inference succeeds. |
| `inference_failed` | `request_id: int, error_code: int, error: String` | Emitted when async inference fails. |
| `inference_cancelled` | `request_id: int` | Emitted when async inference is cancelled. |

If an async request is rejected before it starts, the failure signal uses `request_id == 0`.

#### Input Dictionary Format

Keys are input tensor names (String). Values are typed arrays matching the
model's expected element types:

| Model Type | GDScript Type |
|------------|---------------|
| float32 | `PackedFloat32Array` |
| int64 | `PackedInt64Array` |
| bool | `PackedByteArray` (0 = false, nonzero = true) |

Data is provided as a flat array. For a `[2, 3]` shape, provide 6 elements
in row-major order.

#### Output Dictionary Format

For each output tensor named `"foo"`, the result dictionary contains:
- `"foo"`: typed array with the output data
- `"foo_shape"`: `PackedInt64Array` with the output shape

#### Example

```gdscript
var session := OrtSession.new()
session.model_loaded.connect(func(_request_id: int, _path: String) -> void:
    var result := session.run_inference({
        "A": PackedFloat32Array([1, 2, 3, 4, 5, 6]),
        "B": PackedFloat32Array([10, 20, 30, 40, 50, 60]),
    })
    print(result["C"])  # [11, 22, 33, 44, 55, 66]
)

session.model_load_failed.connect(func(request_id: int, error_code: int, error: String) -> void:
    push_error("load failed (%s / %s): %s" % [request_id, error_code, error])
)

session.load_model_async("res://models/add_floats.onnx")
```

---

### OrtTensorSpec

**Inherits:** `RefCounted`

Describes a single tensor input or output of an ONNX model.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tensor_name` | `String` | Name of the tensor in the model graph. |
| `element_type_name` | `String` | Human-readable type name ("float32", "int64", "bool"). |
| `element_type` | `int` | Numeric type enum (0=Float32, 1=Int64, 2=Bool, 3=Unsupported). |
| `shape` | `PackedInt64Array` | Tensor shape. -1 indicates a dynamic dimension. |
| `rank` | `int` | Number of dimensions. |
| `is_static_shape` | `bool` | True if all dimensions are known (no -1 values). |
| `element_count` | `int` | Total elements for static shapes, -1 if dynamic. |

#### Methods

| Method | Return | Description |
|--------|--------|-------------|
| `describe()` | `String` | Human-readable description, e.g. `"input_0: float32[1, 3, 224, 224]"` |

---

### OrtModelMetadata

**Inherits:** `RefCounted`

Metadata extracted from a loaded ONNX model.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `producer_name` | `String` | Tool that produced the model (e.g. "pytorch"). |
| `graph_name` | `String` | Name of the computation graph. |
| `graph_description` | `String` | Description of the graph. |
| `domain` | `String` | Model domain. |
| `version` | `int` | Model version number. |
| `custom_metadata` | `Dictionary` | User-defined key-value metadata. |

---

### OrtProviderConfig

**Inherits:** `Resource`

Configuration for ONNX Runtime session options. Can be saved as a Godot resource.

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `provider` | `String` | `"CPU"` | Execution provider name. |
| `intra_op_threads` | `int` | `0` | Threads for intra-op parallelism (0 = ORT default). |
| `inter_op_threads` | `int` | `0` | Threads for inter-op parallelism (0 = ORT default). |
| `optimization_level` | `int` | `99` | Graph optimization level (0=none, 1=basic, 2=extended, 99=all). |
| `optimized_model_path` | `String` | `""` | Path to serialize the optimized model. |

#### Static Methods

| Method | Return | Description |
|--------|--------|-------------|
| `get_available_providers()` | `PackedStringArray` | List providers available in the current ORT build. |
