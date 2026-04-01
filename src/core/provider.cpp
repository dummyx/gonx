#include "gonx/core/provider.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>

namespace gonx {

const char* provider_name(ExecutionProvider ep) noexcept {
    switch (ep) {
    case ExecutionProvider::CPU:
        return "CPUExecutionProvider";
    case ExecutionProvider::CUDA:
        return "CUDAExecutionProvider";
    case ExecutionProvider::DirectML:
        return "DmlExecutionProvider";
    case ExecutionProvider::CoreML:
        return "CoreMLExecutionProvider";
    }
    return "CPUExecutionProvider";
}

ExecutionProvider parse_provider(const std::string& name) noexcept {
    if (name == "CUDA" || name == "CUDAExecutionProvider") {
        return ExecutionProvider::CUDA;
    }
    if (name == "DirectML" || name == "DmlExecutionProvider") {
        return ExecutionProvider::DirectML;
    }
    if (name == "CoreML" || name == "CoreMLExecutionProvider") {
        return ExecutionProvider::CoreML;
    }
    return ExecutionProvider::CPU;
}

std::vector<ExecutionProvider> available_providers() {
    auto ort_providers = Ort::GetAvailableProviders();
    std::vector<ExecutionProvider> result;
    result.reserve(ort_providers.size());
    for (const auto& name : ort_providers) {
        result.push_back(parse_provider(name));
    }
    // Deduplicate while preserving order
    std::vector<ExecutionProvider> unique;
    for (auto ep : result) {
        if (std::find(unique.begin(), unique.end(), ep) == unique.end()) {
            unique.push_back(ep);
        }
    }
    return unique;
}

}  // namespace gonx
