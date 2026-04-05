#include "gonx/core/provider.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>

namespace gonx {

namespace {

std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace

const char* provider_name(ExecutionProvider ep) noexcept {
    switch (ep) {
    case ExecutionProvider::CPU:
        return "CPUExecutionProvider";
    case ExecutionProvider::CUDA:
        return "CUDAExecutionProvider";
    case ExecutionProvider::MiGraphX:
        return "MIGraphXExecutionProvider";
    case ExecutionProvider::OpenVINO:
        return "OpenVINOExecutionProvider";
    case ExecutionProvider::DirectML:
        return "DmlExecutionProvider";
    case ExecutionProvider::CoreML:
        return "CoreMLExecutionProvider";
    }
    return "CPUExecutionProvider";
}

ExecutionProvider parse_provider(const std::string& name) noexcept {
    auto lower = to_lower(name);
    if (lower == "cuda" || lower == "cudaexecutionprovider") {
        return ExecutionProvider::CUDA;
    }
    if (lower == "migraphx" || lower == "migraphxexecutionprovider") {
        return ExecutionProvider::MiGraphX;
    }
    if (lower == "openvino" || lower == "openvinoexecutionprovider") {
        return ExecutionProvider::OpenVINO;
    }
    if (lower == "directml" || lower == "dml" || lower == "dmlexecutionprovider") {
        return ExecutionProvider::DirectML;
    }
    if (lower == "coreml" || lower == "coremlexecutionprovider") {
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
