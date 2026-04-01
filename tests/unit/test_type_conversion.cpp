#include "gonx/core/type_conversion.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using namespace gonx;

TEST_CASE("from_ort_element_type maps known types", "[type_conversion]") {
    CHECK(from_ort_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) == ElementType::Float32);
    CHECK(from_ort_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) == ElementType::Int64);
    CHECK(from_ort_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) == ElementType::Bool);
}

TEST_CASE("from_ort_element_type returns Unsupported for unknown types", "[type_conversion]") {
    CHECK(from_ort_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) ==
          ElementType::Unsupported);
    CHECK(from_ort_element_type(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ==
          ElementType::Unsupported);
}

TEST_CASE("to_ort_element_type roundtrips with from_ort_element_type", "[type_conversion]") {
    CHECK(from_ort_element_type(to_ort_element_type(ElementType::Float32)) ==
          ElementType::Float32);
    CHECK(from_ort_element_type(to_ort_element_type(ElementType::Int64)) == ElementType::Int64);
    CHECK(from_ort_element_type(to_ort_element_type(ElementType::Bool)) == ElementType::Bool);
}

TEST_CASE("validate_tensor_data accepts correct data", "[type_conversion]") {
    std::vector<int64_t> shape = {2, 3};
    auto status =
        validate_tensor_data(6 * sizeof(float), ElementType::Float32, shape);
    CHECK(status.has_value());
}

TEST_CASE("validate_tensor_data rejects wrong size", "[type_conversion]") {
    std::vector<int64_t> shape = {2, 3};
    auto status =
        validate_tensor_data(5 * sizeof(float), ElementType::Float32, shape);
    CHECK(status.has_error());
    CHECK(status.error().code == ErrorCode::InvalidShape);
}

TEST_CASE("validate_tensor_data rejects negative dimensions", "[type_conversion]") {
    std::vector<int64_t> shape = {-1, 3};
    auto status =
        validate_tensor_data(3 * sizeof(float), ElementType::Float32, shape);
    CHECK(status.has_error());
    CHECK(status.error().code == ErrorCode::InvalidShape);
}

TEST_CASE("validate_tensor_data rejects zero dimensions", "[type_conversion]") {
    std::vector<int64_t> shape = {0, 3};
    auto status =
        validate_tensor_data(0, ElementType::Float32, shape);
    CHECK(status.has_error());
    CHECK(status.error().code == ErrorCode::InvalidShape);
}

TEST_CASE("validate_tensor_data rejects unsupported type", "[type_conversion]") {
    std::vector<int64_t> shape = {2, 3};
    auto status = validate_tensor_data(24, ElementType::Unsupported, shape);
    CHECK(status.has_error());
    CHECK(status.error().code == ErrorCode::InvalidType);
}

TEST_CASE("create_float_tensor succeeds with valid data", "[type_conversion]") {
    std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<int64_t, 2> shape = {2, 3};

    auto result = create_float_tensor(data, shape);
    REQUIRE(result.has_value());

    auto& tensor = result.value();
    CHECK(tensor.IsTensor());
    CHECK(get_tensor_element_type(tensor) == ElementType::Float32);

    auto out_shape = get_tensor_shape(tensor);
    REQUIRE(out_shape.size() == 2);
    CHECK(out_shape[0] == 2);
    CHECK(out_shape[1] == 3);
}

TEST_CASE("create_float_tensor fails with wrong data size", "[type_conversion]") {
    std::array<float, 5> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::array<int64_t, 2> shape = {2, 3};

    auto result = create_float_tensor(data, shape);
    CHECK(result.has_error());
}

TEST_CASE("create_int64_tensor succeeds with valid data", "[type_conversion]") {
    std::array<int64_t, 4> data = {10, 20, 30, 40};
    std::array<int64_t, 2> shape = {2, 2};

    auto result = create_int64_tensor(data, shape);
    REQUIRE(result.has_value());
    CHECK(result.value().IsTensor());
}
