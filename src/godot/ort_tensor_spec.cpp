#include "gonx/godot/ort_tensor_spec.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace gonx {

void OrtTensorSpec::set_from_spec(const TensorSpec& spec) {
    spec_ = spec;
}

godot::String OrtTensorSpec::get_tensor_name() const {
    return godot::String(spec_.name.c_str());
}

godot::String OrtTensorSpec::get_element_type_name() const {
    return godot::String(element_type_name(spec_.element_type));
}

int OrtTensorSpec::get_element_type() const {
    return static_cast<int>(spec_.element_type);
}

godot::PackedInt64Array OrtTensorSpec::get_shape() const {
    godot::PackedInt64Array arr;
    arr.resize(static_cast<int64_t>(spec_.shape.size()));
    for (std::size_t i = 0; i < spec_.shape.size(); ++i) {
        arr[static_cast<int64_t>(i)] = spec_.shape[i];
    }
    return arr;
}

int OrtTensorSpec::get_rank() const {
    return static_cast<int>(spec_.shape.size());
}

bool OrtTensorSpec::get_is_static_shape() const {
    return spec_.is_static_shape();
}

int64_t OrtTensorSpec::get_element_count() const {
    return spec_.element_count();
}

godot::String OrtTensorSpec::describe() const {
    return godot::String(spec_.to_string().c_str());
}

void OrtTensorSpec::_bind_methods() {
    using namespace godot;

    ClassDB::bind_method(D_METHOD("get_tensor_name"), &OrtTensorSpec::get_tensor_name);
    ClassDB::bind_method(D_METHOD("get_element_type_name"),
                         &OrtTensorSpec::get_element_type_name);
    ClassDB::bind_method(D_METHOD("get_element_type"), &OrtTensorSpec::get_element_type);
    ClassDB::bind_method(D_METHOD("get_shape"), &OrtTensorSpec::get_shape);
    ClassDB::bind_method(D_METHOD("get_rank"), &OrtTensorSpec::get_rank);
    ClassDB::bind_method(D_METHOD("get_is_static_shape"), &OrtTensorSpec::get_is_static_shape);
    ClassDB::bind_method(D_METHOD("get_element_count"), &OrtTensorSpec::get_element_count);
    ClassDB::bind_method(D_METHOD("describe"), &OrtTensorSpec::describe);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "tensor_name"), "", "get_tensor_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "element_type_name"), "",
                 "get_element_type_name");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "element_type"), "", "get_element_type");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT64_ARRAY, "shape"), "", "get_shape");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "rank"), "", "get_rank");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_static_shape"), "", "get_is_static_shape");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "element_count"), "", "get_element_count");
}

}  // namespace gonx
