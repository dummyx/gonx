#pragma once

#include "gonx/core/error.hpp"
#include "gonx/core/provider.hpp"
#include "gonx/core/tensor_spec.hpp"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gonx {

/// Metadata extracted from a loaded ONNX model.
struct ModelMetadata {
    std::string producer_name;
    std::string graph_name;
    std::string graph_description;
    std::string domain;
    int64_t version = 0;
    std::unordered_map<std::string, std::string> custom_metadata;
};

/// Core inference session wrapping Ort::Session.
/// This class is not Godot-aware. It operates on STL types and ORT types only.
/// Thread safety: a single InferenceSession can be used for concurrent Run() calls
/// (ORT sessions are thread-safe for Run), but load/metadata must complete first.
class InferenceSession {
public:
    InferenceSession();
    ~InferenceSession();

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession(InferenceSession&&) noexcept;
    InferenceSession& operator=(InferenceSession&&) noexcept;

    /// Load a model from a file path with the given configuration.
    [[nodiscard]] Status load(const std::filesystem::path& model_path,
                              const SessionConfig& config = {});

    /// Whether a model is currently loaded.
    [[nodiscard]] bool is_loaded() const noexcept;

    /// Get input tensor specifications.
    [[nodiscard]] const std::vector<TensorSpec>& input_specs() const noexcept;

    /// Get output tensor specifications.
    [[nodiscard]] const std::vector<TensorSpec>& output_specs() const noexcept;

    /// Get model metadata.
    [[nodiscard]] const ModelMetadata& metadata() const noexcept;

    /// Run synchronous inference.
    /// `inputs` must contain one Ort::Value per model input, in order.
    /// Returns one Ort::Value per model output on success.
    [[nodiscard]] Result<std::vector<Ort::Value>> run(
        std::vector<Ort::Value>& inputs,
        Ort::RunOptions* run_options = nullptr);

    /// Get the path of the currently loaded model.
    [[nodiscard]] const std::filesystem::path& model_path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gonx
