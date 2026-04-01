#include "gonx/godot/ort_model_metadata.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace gonx {

void OrtModelMetadata::set_from_metadata(const ModelMetadata& meta) {
    meta_ = meta;
}

godot::String OrtModelMetadata::get_producer_name() const {
    return godot::String(meta_.producer_name.c_str());
}

godot::String OrtModelMetadata::get_graph_name() const {
    return godot::String(meta_.graph_name.c_str());
}

godot::String OrtModelMetadata::get_graph_description() const {
    return godot::String(meta_.graph_description.c_str());
}

godot::String OrtModelMetadata::get_domain() const {
    return godot::String(meta_.domain.c_str());
}

int64_t OrtModelMetadata::get_version() const {
    return meta_.version;
}

godot::Dictionary OrtModelMetadata::get_custom_metadata() const {
    godot::Dictionary dict;
    for (const auto& [key, value] : meta_.custom_metadata) {
        dict[godot::String(key.c_str())] = godot::String(value.c_str());
    }
    return dict;
}

void OrtModelMetadata::_bind_methods() {
    using namespace godot;

    ClassDB::bind_method(D_METHOD("get_producer_name"), &OrtModelMetadata::get_producer_name);
    ClassDB::bind_method(D_METHOD("get_graph_name"), &OrtModelMetadata::get_graph_name);
    ClassDB::bind_method(D_METHOD("get_graph_description"),
                         &OrtModelMetadata::get_graph_description);
    ClassDB::bind_method(D_METHOD("get_domain"), &OrtModelMetadata::get_domain);
    ClassDB::bind_method(D_METHOD("get_version"), &OrtModelMetadata::get_version);
    ClassDB::bind_method(D_METHOD("get_custom_metadata"),
                         &OrtModelMetadata::get_custom_metadata);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "producer_name"), "", "get_producer_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "graph_name"), "", "get_graph_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "graph_description"), "",
                 "get_graph_description");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "domain"), "", "get_domain");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "version"), "", "get_version");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "custom_metadata"), "",
                 "get_custom_metadata");
}

}  // namespace gonx
