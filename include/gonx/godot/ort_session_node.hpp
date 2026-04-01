#pragma once

#include "gonx/core/session.hpp"
#include "gonx/godot/ort_model_metadata.hpp"
#include "gonx/godot/ort_provider_config.hpp"
#include "gonx/godot/ort_tensor_spec.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace gonx {

/// Main Godot-facing class for ONNX model loading and inference.
///
/// Usage from GDScript:
///   var session = OrtSession.new()
///   var err = session.load_model("res://model.onnx")
///   var inputs = {"input_name": PackedFloat32Array([1.0, 2.0, 3.0])}
///   var result = session.run_inference(inputs)
class OrtSession : public godot::RefCounted {
    GDCLASS(OrtSession, godot::RefCounted)

public:
    OrtSession();
    ~OrtSession() override;

    /// Load a model from a file path. Returns OK on success or an error code.
    int load_model(const godot::String& path);

    /// Load a model with explicit provider configuration.
    int load_model_with_config(const godot::String& path,
                               const godot::Ref<OrtProviderConfig>& config);

    /// Whether a model is loaded and ready for inference.
    bool is_loaded() const;

    /// Get metadata for all model inputs.
    godot::Array get_input_specs() const;

    /// Get metadata for all model outputs.
    godot::Array get_output_specs() const;

    /// Get model metadata (producer, graph name, custom properties).
    godot::Ref<OrtModelMetadata> get_metadata() const;

    /// Run synchronous inference.
    /// `inputs` is a Dictionary mapping input names (String) to typed arrays
    /// (PackedFloat32Array, PackedInt64Array, or PackedByteArray for bool).
    /// Returns a Dictionary mapping output names to typed arrays, or an empty
    /// Dictionary on error (check get_last_error()).
    godot::Dictionary run_inference(const godot::Dictionary& inputs);

    /// Start asynchronous inference. Emits `inference_completed` or `inference_failed`.
    void run_inference_async(const godot::Dictionary& inputs);

    /// Get the error message from the last failed operation.
    godot::String get_last_error() const;

    /// Get the model file path.
    godot::String get_model_path() const;

protected:
    static void _bind_methods();

private:
    /// Convert GDScript Dictionary inputs to ORT values.
    Result<std::vector<Ort::Value>> prepare_inputs(const godot::Dictionary& inputs);

    /// Convert ORT output values to GDScript Dictionary.
    godot::Dictionary convert_outputs(std::vector<Ort::Value>& outputs);

    /// Process async results on the main thread (called via call_deferred).
    void _on_async_completed(const godot::Dictionary& result);
    void _on_async_failed(const godot::String& error);

    InferenceSession session_;
    mutable std::mutex mutex_;
    std::atomic<bool> async_running_{false};
    std::thread async_thread_;
    godot::String last_error_;
};

}  // namespace gonx
