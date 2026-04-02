#include "gonx/core/type_conversion.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace gonx {

ElementType from_ort_element_type(ONNXTensorElementDataType ort_type) noexcept {
    switch (ort_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return ElementType::Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return ElementType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return ElementType::Bool;
    default:
        return ElementType::Unsupported;
    }
}

ONNXTensorElementDataType to_ort_element_type(ElementType type) noexcept {
    switch (type) {
    case ElementType::Float32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case ElementType::Int64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case ElementType::Bool:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case ElementType::Unsupported:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

namespace {

int64_t compute_element_count(std::span<const int64_t> shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
}

}  // namespace

Status validate_tensor_data(std::size_t data_byte_size, ElementType element_type,
                            std::span<const int64_t> shape) {
    if (element_type == ElementType::Unsupported) {
        return Error::make(ErrorCode::InvalidType, "Unsupported element type");
    }

    auto elem_size = element_type_size(element_type);
    if (elem_size == 0) {
        return Error::make(ErrorCode::InvalidType, "Element type has zero size");
    }

    for (auto dim : shape) {
        if (dim < 0) {
            return Error::make(ErrorCode::InvalidShape,
                               "Shape contains negative dimension, which is not allowed for "
                               "concrete tensor creation");
        }
        if (dim == 0) {
            return Error::make(ErrorCode::InvalidShape, "Shape contains zero dimension");
        }
    }

    auto expected_count = compute_element_count(shape);
    auto expected_bytes = static_cast<std::size_t>(expected_count) * elem_size;

    if (data_byte_size != expected_bytes) {
        std::ostringstream oss;
        oss << "Data size mismatch: expected " << expected_bytes << " bytes ("
            << expected_count << " elements x " << elem_size << " bytes), got "
            << data_byte_size << " bytes";
        return Error::make(ErrorCode::InvalidShape, oss.str());
    }

    return Status::ok();
}

Result<Ort::Value> create_float_tensor(std::span<const float> data,
                                       std::span<const int64_t> shape) {
    auto status = validate_tensor_data(data.size_bytes(), ElementType::Float32, shape);
    if (status.has_error()) {
        return status.error();
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(tensor.GetTensorMutableData<float>(), data.data(), data.size_bytes());
    return tensor;
}

Result<Ort::Value> create_int64_tensor(std::span<const int64_t> data,
                                       std::span<const int64_t> shape) {
    auto status = validate_tensor_data(data.size_bytes(), ElementType::Int64, shape);
    if (status.has_error()) {
        return status.error();
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    std::memcpy(tensor.GetTensorMutableData<int64_t>(), data.data(), data.size_bytes());
    return tensor;
}

Result<Ort::Value> create_bool_tensor(std::span<const uint8_t> data,
                                      std::span<const int64_t> shape) {
    auto status = validate_tensor_data(data.size() * sizeof(bool), ElementType::Bool, shape);
    if (status.has_error()) {
        return status.error();
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto tensor = Ort::Value::CreateTensor(
        allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
    bool* destination = tensor.GetTensorMutableData<bool>();
    for (std::size_t index = 0; index < data.size(); ++index) {
        destination[index] = data[index] != 0;
    }
    return tensor;
}

std::vector<int64_t> get_tensor_shape(const Ort::Value& value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    return info.GetShape();
}

ElementType get_tensor_element_type(const Ort::Value& value) {
    auto info = value.GetTensorTypeAndShapeInfo();
    return from_ort_element_type(info.GetElementType());
}

}  // namespace gonx
