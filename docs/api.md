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
| `is_loaded()` | `bool` | Whether a model is loaded and ready. |
| `get_input_specs()` | `Array[OrtTensorSpec]` | Metadata for all model inputs. |
| `get_output_specs()` | `Array[OrtTensorSpec]` | Metadata for all model outputs. |
| `get_metadata()` | `OrtModelMetadata` | Model metadata (producer, graph name, etc.). |
| `run_inference(inputs: Dictionary)` | `Dictionary` | Run synchronous inference. |
| `run_inference_async(inputs: Dictionary)` | `void` | Start async inference. Results via signals. |
| `get_last_error()` | `String` | Error message from the last failed operation. |
| `get_model_path()` | `String` | Path of the currently loaded model. |

#### Signals

| Signal | Arguments | Description |
|--------|-----------|-------------|
| `inference_completed` | `result: Dictionary` | Emitted when async inference succeeds. |
| `inference_failed` | `error: String` | Emitted when async inference fails. |

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
session.load_model("res://models/add_floats.onnx")

var result := session.run_inference({
    "A": PackedFloat32Array([1, 2, 3, 4, 5, 6]),
    "B": PackedFloat32Array([10, 20, 30, 40, 50, 60]),
})
print(result["C"])  # [11, 22, 33, 44, 55, 66]
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
