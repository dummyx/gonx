#pragma once

#include <string>
#include <vector>

namespace gonx {

/// Execution providers supported by gonx.
/// CPU is always available. Others require matching ORT builds and hardware.
enum class ExecutionProvider {
    CPU,
    CUDA,
    MiGraphX,
    OpenVINO,
    DirectML,
    CoreML,
};

/// Returns the canonical string name for a provider (e.g., "CPUExecutionProvider").
[[nodiscard]] const char* provider_name(ExecutionProvider ep) noexcept;

/// Parse a provider name string. Returns CPU on unrecognized input.
[[nodiscard]] ExecutionProvider parse_provider(const std::string& name) noexcept;

/// Configuration for session creation.
struct SessionConfig {
    std::vector<ExecutionProvider> providers = {ExecutionProvider::CPU};
    int device_id = 0;             // GPU device index for CUDA/MiGraphX
    int intra_op_num_threads = 0;  // 0 = ORT default
    int inter_op_num_threads = 0;
    int optimization_level = 99;   // ORT_ENABLE_ALL
    std::string optimized_model_path;  // empty = don't serialize
};

/// Query which providers are available in the current ORT build.
[[nodiscard]] std::vector<ExecutionProvider> available_providers();

}  // namespace gonx
