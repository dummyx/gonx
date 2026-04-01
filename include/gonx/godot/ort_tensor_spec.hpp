#pragma once

#include "gonx/core/tensor_spec.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace gonx {

/// Godot-facing wrapper for TensorSpec.
/// Exposes tensor name, element type, and shape to GDScript/C#.
class OrtTensorSpec : public godot::RefCounted {
    GDCLASS(OrtTensorSpec, godot::RefCounted)

public:
    OrtTensorSpec() = default;

    void set_from_spec(const TensorSpec& spec);

    godot::String get_tensor_name() const;
    godot::String get_element_type_name() const;
    int get_element_type() const;
    godot::PackedInt64Array get_shape() const;
    int get_rank() const;
    bool get_is_static_shape() const;
    int64_t get_element_count() const;
    godot::String describe() const;

protected:
    static void _bind_methods();

private:
    TensorSpec spec_;
};

}  // namespace gonx
