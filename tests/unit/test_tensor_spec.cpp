#include "gonx/core/tensor_spec.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace gonx;

TEST_CASE("element_type_name returns correct names", "[tensor_spec]") {
    CHECK(std::string(element_type_name(ElementType::Float32)) == "float32");
    CHECK(std::string(element_type_name(ElementType::Int64)) == "int64");
    CHECK(std::string(element_type_name(ElementType::Bool)) == "bool");
    CHECK(std::string(element_type_name(ElementType::Unsupported)) == "unsupported");
}

TEST_CASE("element_type_size returns correct sizes", "[tensor_spec]") {
    CHECK(element_type_size(ElementType::Float32) == sizeof(float));
    CHECK(element_type_size(ElementType::Int64) == sizeof(int64_t));
    CHECK(element_type_size(ElementType::Bool) == sizeof(bool));
    CHECK(element_type_size(ElementType::Unsupported) == 0);
}

TEST_CASE("TensorSpec::element_count for static shape", "[tensor_spec]") {
    TensorSpec spec{"x", ElementType::Float32, {2, 3, 4}};
    CHECK(spec.element_count() == 24);
    CHECK(spec.is_static_shape());
}

TEST_CASE("TensorSpec::element_count for dynamic shape", "[tensor_spec]") {
    TensorSpec spec{"x", ElementType::Float32, {-1, 3, 4}};
    CHECK(spec.element_count() == -1);
    CHECK_FALSE(spec.is_static_shape());
}

TEST_CASE("TensorSpec::element_count for empty shape", "[tensor_spec]") {
    TensorSpec spec{"x", ElementType::Float32, {}};
    CHECK(spec.element_count() == 0);
}

TEST_CASE("TensorSpec::to_string produces readable output", "[tensor_spec]") {
    TensorSpec spec{"input_0", ElementType::Float32, {1, 3, 224, 224}};
    auto str = spec.to_string();
    CHECK(str.find("input_0") != std::string::npos);
    CHECK(str.find("float32") != std::string::npos);
    CHECK(str.find("224") != std::string::npos);
}

TEST_CASE("TensorSpec::to_string with dynamic dims shows ?", "[tensor_spec]") {
    TensorSpec spec{"batch_input", ElementType::Int64, {-1, 10}};
    auto str = spec.to_string();
    CHECK(str.find("?") != std::string::npos);
    CHECK(str.find("10") != std::string::npos);
}
