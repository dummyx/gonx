#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gonx {

/// Supported element types for the MVP.
enum class ElementType {
    Float32,
    Int64,
    Bool,
    Unsupported,
};

/// Returns the human-readable name for an element type.
[[nodiscard]] const char* element_type_name(ElementType type) noexcept;

/// Returns the byte size of one element, or 0 for unsupported.
[[nodiscard]] std::size_t element_type_size(ElementType type) noexcept;

/// Metadata describing a single tensor input or output of an ONNX model.
struct TensorSpec {
    std::string name;
    ElementType element_type = ElementType::Unsupported;
    std::vector<int64_t> shape;  // -1 for dynamic dimensions

    /// Total number of elements for a fully static shape. Returns -1 if any dimension is dynamic.
    [[nodiscard]] int64_t element_count() const noexcept;

    /// Whether all dimensions are known (no -1 values).
    [[nodiscard]] bool is_static_shape() const noexcept;

    /// Human-readable description for error messages.
    [[nodiscard]] std::string to_string() const;
};

}  // namespace gonx
