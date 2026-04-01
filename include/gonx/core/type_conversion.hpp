#pragma once

#include "gonx/core/error.hpp"
#include "gonx/core/tensor_spec.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gonx {

/// Convert an ORT tensor element type enum to our ElementType.
[[nodiscard]] ElementType from_ort_element_type(ONNXTensorElementDataType ort_type) noexcept;

/// Convert our ElementType to ORT tensor element type enum.
[[nodiscard]] ONNXTensorElementDataType to_ort_element_type(ElementType type) noexcept;

/// Create an Ort::Value tensor from a contiguous buffer of float32 data.
/// Validates shape against the data length.
[[nodiscard]] Result<Ort::Value> create_float_tensor(std::span<const float> data,
                                                     std::span<const int64_t> shape);

/// Create an Ort::Value tensor from a contiguous buffer of int64 data.
[[nodiscard]] Result<Ort::Value> create_int64_tensor(std::span<const int64_t> data,
                                                     std::span<const int64_t> shape);

/// Create an Ort::Value tensor from a contiguous buffer of bool data (as uint8_t).
/// Uses uint8_t because std::vector<bool> is not a contiguous container.
[[nodiscard]] Result<Ort::Value> create_bool_tensor(std::span<const uint8_t> data,
                                                    std::span<const int64_t> shape);

/// Extract shape from an Ort::Value tensor.
[[nodiscard]] std::vector<int64_t> get_tensor_shape(const Ort::Value& value);

/// Extract element type from an Ort::Value tensor.
[[nodiscard]] ElementType get_tensor_element_type(const Ort::Value& value);

/// Validate that a given data buffer matches the expected shape and element type.
[[nodiscard]] Status validate_tensor_data(std::size_t data_byte_size, ElementType element_type,
                                          std::span<const int64_t> shape);

}  // namespace gonx
