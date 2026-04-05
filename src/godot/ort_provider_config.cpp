#include "gonx/godot/ort_provider_config.hpp"

#include <godot_cpp/core/class_db.hpp>

namespace gonx {

void OrtProviderConfig::set_provider(const godot::String& provider) {
    provider_ = provider;
}

godot::String OrtProviderConfig::get_provider() const {
    return provider_;
}

void OrtProviderConfig::set_intra_op_threads(int threads) {
    intra_op_threads_ = threads;
}

int OrtProviderConfig::get_intra_op_threads() const {
    return intra_op_threads_;
}

void OrtProviderConfig::set_inter_op_threads(int threads) {
    inter_op_threads_ = threads;
}

int OrtProviderConfig::get_inter_op_threads() const {
    return inter_op_threads_;
}

void OrtProviderConfig::set_optimization_level(int level) {
    optimization_level_ = level;
}

int OrtProviderConfig::get_optimization_level() const {
    return optimization_level_;
}

void OrtProviderConfig::set_optimized_model_path(const godot::String& path) {
    optimized_model_path_ = path;
}

godot::String OrtProviderConfig::get_optimized_model_path() const {
    return optimized_model_path_;
}

void OrtProviderConfig::set_device_id(int id) {
    device_id_ = id;
}

int OrtProviderConfig::get_device_id() const {
    return device_id_;
}

SessionConfig OrtProviderConfig::to_session_config() const {
    SessionConfig config;
    config.providers = {parse_provider(provider_.utf8().get_data())};
    config.device_id = device_id_;
    config.intra_op_num_threads = intra_op_threads_;
    config.inter_op_num_threads = inter_op_threads_;
    config.optimization_level = optimization_level_;
    if (!optimized_model_path_.is_empty()) {
        config.optimized_model_path = optimized_model_path_.utf8().get_data();
    }
    return config;
}

godot::PackedStringArray OrtProviderConfig::get_available_providers() {
    godot::PackedStringArray result;
    for (auto ep : available_providers()) {
        result.push_back(godot::String(provider_name(ep)));
    }
    return result;
}

void OrtProviderConfig::_bind_methods() {
    using namespace godot;

    ClassDB::bind_method(D_METHOD("set_provider", "provider"),
                         &OrtProviderConfig::set_provider);
    ClassDB::bind_method(D_METHOD("get_provider"), &OrtProviderConfig::get_provider);

    ClassDB::bind_method(D_METHOD("set_intra_op_threads", "threads"),
                         &OrtProviderConfig::set_intra_op_threads);
    ClassDB::bind_method(D_METHOD("get_intra_op_threads"),
                         &OrtProviderConfig::get_intra_op_threads);

    ClassDB::bind_method(D_METHOD("set_inter_op_threads", "threads"),
                         &OrtProviderConfig::set_inter_op_threads);
    ClassDB::bind_method(D_METHOD("get_inter_op_threads"),
                         &OrtProviderConfig::get_inter_op_threads);

    ClassDB::bind_method(D_METHOD("set_optimization_level", "level"),
                         &OrtProviderConfig::set_optimization_level);
    ClassDB::bind_method(D_METHOD("get_optimization_level"),
                         &OrtProviderConfig::get_optimization_level);

    ClassDB::bind_method(D_METHOD("set_optimized_model_path", "path"),
                         &OrtProviderConfig::set_optimized_model_path);
    ClassDB::bind_method(D_METHOD("get_optimized_model_path"),
                         &OrtProviderConfig::get_optimized_model_path);

    ClassDB::bind_method(D_METHOD("set_device_id", "id"),
                         &OrtProviderConfig::set_device_id);
    ClassDB::bind_method(D_METHOD("get_device_id"),
                         &OrtProviderConfig::get_device_id);

    ClassDB::bind_static_method("OrtProviderConfig",
                                D_METHOD("get_available_providers"),
                                &OrtProviderConfig::get_available_providers);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "provider"), "set_provider", "get_provider");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "device_id"), "set_device_id", "get_device_id");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "intra_op_threads"), "set_intra_op_threads",
                 "get_intra_op_threads");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "inter_op_threads"), "set_inter_op_threads",
                 "get_inter_op_threads");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "optimization_level"), "set_optimization_level",
                 "get_optimization_level");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "optimized_model_path"),
                 "set_optimized_model_path", "get_optimized_model_path");
}

}  // namespace gonx
