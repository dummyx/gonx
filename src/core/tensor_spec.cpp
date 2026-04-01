#include "gonx/core/tensor_spec.hpp"

#include <numeric>
#include <sstream>

namespace gonx {

const char* element_type_name(ElementType type) noexcept {
    switch (type) {
    case ElementType::Float32:
        return "float32";
    case ElementType::Int64:
        return "int64";
    case ElementType::Bool:
        return "bool";
    case ElementType::Unsupported:
        return "unsupported";
    }
    return "unknown";
}

std::size_t element_type_size(ElementType type) noexcept {
    switch (type) {
    case ElementType::Float32:
        return sizeof(float);
    case ElementType::Int64:
        return sizeof(int64_t);
    case ElementType::Bool:
        return sizeof(bool);
    case ElementType::Unsupported:
        return 0;
    }
    return 0;
}

int64_t TensorSpec::element_count() const noexcept {
    if (shape.empty()) {
        return 0;
    }
    for (auto dim : shape) {
        if (dim < 0) {
            return -1;
        }
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
}

bool TensorSpec::is_static_shape() const noexcept {
    for (auto dim : shape) {
        if (dim < 0) {
            return false;
        }
    }
    return true;
}

std::string TensorSpec::to_string() const {
    std::ostringstream oss;
    oss << name << ": " << element_type_name(element_type) << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        if (shape[i] < 0) {
            oss << "?";
        } else {
            oss << shape[i];
        }
    }
    oss << "]";
    return oss.str();
}

}  // namespace gonx
