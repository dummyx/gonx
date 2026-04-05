#include "gonx/core/provider.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace gonx;

TEST_CASE("provider_name returns canonical ORT names", "[provider]") {
    CHECK(std::string(provider_name(ExecutionProvider::CPU)) == "CPUExecutionProvider");
    CHECK(std::string(provider_name(ExecutionProvider::CUDA)) == "CUDAExecutionProvider");
    CHECK(std::string(provider_name(ExecutionProvider::MiGraphX)) == "MIGraphXExecutionProvider");
    CHECK(std::string(provider_name(ExecutionProvider::OpenVINO)) == "OpenVINOExecutionProvider");
    CHECK(std::string(provider_name(ExecutionProvider::DirectML)) == "DmlExecutionProvider");
    CHECK(std::string(provider_name(ExecutionProvider::CoreML)) == "CoreMLExecutionProvider");
}

TEST_CASE("parse_provider recognizes known names", "[provider]") {
    CHECK(parse_provider("CPU") == ExecutionProvider::CPU);
    CHECK(parse_provider("CPUExecutionProvider") == ExecutionProvider::CPU);
    CHECK(parse_provider("CUDA") == ExecutionProvider::CUDA);
    CHECK(parse_provider("CUDAExecutionProvider") == ExecutionProvider::CUDA);
    CHECK(parse_provider("MiGraphX") == ExecutionProvider::MiGraphX);
    CHECK(parse_provider("MIGraphX") == ExecutionProvider::MiGraphX);
    CHECK(parse_provider("MIGraphXExecutionProvider") == ExecutionProvider::MiGraphX);
    CHECK(parse_provider("OpenVINO") == ExecutionProvider::OpenVINO);
    CHECK(parse_provider("OpenVINOExecutionProvider") == ExecutionProvider::OpenVINO);
    CHECK(parse_provider("DirectML") == ExecutionProvider::DirectML);
    CHECK(parse_provider("DmlExecutionProvider") == ExecutionProvider::DirectML);
    CHECK(parse_provider("CoreML") == ExecutionProvider::CoreML);
    CHECK(parse_provider("CoreMLExecutionProvider") == ExecutionProvider::CoreML);
}

TEST_CASE("parse_provider defaults to CPU for unknown names", "[provider]") {
    CHECK(parse_provider("UnknownProvider") == ExecutionProvider::CPU);
    CHECK(parse_provider("") == ExecutionProvider::CPU);
}

TEST_CASE("available_providers includes CPU", "[provider]") {
    auto providers = available_providers();
    bool has_cpu = false;
    for (auto ep : providers) {
        if (ep == ExecutionProvider::CPU) {
            has_cpu = true;
            break;
        }
    }
    CHECK(has_cpu);
}
