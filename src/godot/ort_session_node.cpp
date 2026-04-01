#include "gonx/godot/ort_session_node.hpp"
#include "gonx/core/type_conversion.hpp"

#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <sstream>

namespace gonx {

OrtSession::OrtSession() = default;

OrtSession::~OrtSession() {
    // Ensure async work finishes before destruction
    if (async_thread_.joinable()) {
        async_running_.store(false, std::memory_order_release);
        async_thread_.join();
    }
}

int OrtSession::load_model(const godot::String& path) {
    return load_model_with_config(path, {});
}

int OrtSession::load_model_with_config(const godot::String& path,
                                          const godot::Ref<OrtProviderConfig>& config) {
    std::lock_guard lock(mutex_);

    // Resolve Godot resource path to filesystem path
    godot::String resolved = path;
    if (path.begins_with("res://") || path.begins_with("user://")) {
        resolved = godot::ProjectSettings::get_singleton()->globalize_path(path);
    }

    std::string native_path = resolved.utf8().get_data();

    SessionConfig session_config;
    if (config.is_valid()) {
        session_config = config->to_session_config();
    }

    auto status = session_.load(native_path, session_config);
    if (status.has_error()) {
        last_error_ = godot::String(status.error().message.c_str());
        godot::UtilityFunctions::push_error(
            godot::String("[gonx] Failed to load model: ") + last_error_);
        return static_cast<int>(status.error().code);
    }

    last_error_ = "";
    return 0;
}

bool OrtSession::is_loaded() const {
    return session_.is_loaded();
}

godot::Array OrtSession::get_input_specs() const {
    godot::Array result;
    if (!session_.is_loaded()) {
        return result;
    }
    for (const auto& spec : session_.input_specs()) {
        godot::Ref<OrtTensorSpec> gd_spec;
        gd_spec.instantiate();
        gd_spec->set_from_spec(spec);
        result.push_back(gd_spec);
    }
    return result;
}

godot::Array OrtSession::get_output_specs() const {
    godot::Array result;
    if (!session_.is_loaded()) {
        return result;
    }
    for (const auto& spec : session_.output_specs()) {
        godot::Ref<OrtTensorSpec> gd_spec;
        gd_spec.instantiate();
        gd_spec->set_from_spec(spec);
        result.push_back(gd_spec);
    }
    return result;
}

godot::Ref<OrtModelMetadata> OrtSession::get_metadata() const {
    godot::Ref<OrtModelMetadata> meta;
    meta.instantiate();
    if (session_.is_loaded()) {
        meta->set_from_metadata(session_.metadata());
    }
    return meta;
}

Result<std::vector<Ort::Value>> OrtSession::prepare_inputs(const godot::Dictionary& inputs) {
    const auto& specs = session_.input_specs();
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.reserve(specs.size());

    for (const auto& spec : specs) {
        godot::String key(spec.name.c_str());
        if (!inputs.has(key)) {
            std::ostringstream oss;
            oss << "Missing input '" << spec.name << "'. Expected inputs: ";
            for (std::size_t j = 0; j < specs.size(); ++j) {
                if (j > 0) {
                    oss << ", ";
                }
                oss << specs[j].name;
            }
            return Error::make(ErrorCode::InvalidArgument, oss.str());
        }

        godot::Variant val = inputs[key];

        switch (spec.element_type) {
        case ElementType::Float32: {
            if (val.get_type() != godot::Variant::PACKED_FLOAT32_ARRAY) {
                std::ostringstream oss;
                oss << "Input '" << spec.name
                    << "' expects PackedFloat32Array, got variant type "
                    << static_cast<int>(val.get_type());
                return Error::make(ErrorCode::InvalidType, oss.str());
            }
            godot::PackedFloat32Array arr = val;

            // Determine concrete shape
            std::vector<int64_t> shape = spec.shape;
            for (auto& dim : shape) {
                if (dim < 0) {
                    // Infer dynamic batch dimension from data
                    // For a single dynamic dim, compute from total elements
                    int64_t known = 1;
                    int dynamic_count = 0;
                    for (auto d : shape) {
                        if (d > 0) {
                            known *= d;
                        } else {
                            dynamic_count++;
                        }
                    }
                    if (dynamic_count == 1 && known > 0) {
                        dim = static_cast<int64_t>(arr.size()) / known;
                    } else {
                        // Cannot infer multiple dynamic dims
                        std::ostringstream oss;
                        oss << "Input '" << spec.name
                            << "' has multiple dynamic dimensions; provide "
                               "explicit shape via input_shapes parameter";
                        return Error::make(ErrorCode::InvalidShape, oss.str());
                    }
                    break;  // Only one pass needed
                }
            }

            auto result = create_float_tensor(
                {arr.ptr(), static_cast<std::size_t>(arr.size())}, shape);
            if (result.has_error()) {
                return result.error();
            }
            ort_inputs.push_back(std::move(result).value());
            break;
        }
        case ElementType::Int64: {
            if (val.get_type() != godot::Variant::PACKED_INT64_ARRAY) {
                std::ostringstream oss;
                oss << "Input '" << spec.name
                    << "' expects PackedInt64Array, got variant type "
                    << static_cast<int>(val.get_type());
                return Error::make(ErrorCode::InvalidType, oss.str());
            }
            godot::PackedInt64Array arr = val;

            std::vector<int64_t> shape = spec.shape;
            for (auto& dim : shape) {
                if (dim < 0) {
                    int64_t known = 1;
                    int dynamic_count = 0;
                    for (auto d : shape) {
                        if (d > 0) {
                            known *= d;
                        } else {
                            dynamic_count++;
                        }
                    }
                    if (dynamic_count == 1 && known > 0) {
                        dim = static_cast<int64_t>(arr.size()) / known;
                    } else {
                        return Error::make(ErrorCode::InvalidShape,
                                           "Multiple dynamic dimensions in input '" +
                                               spec.name + "'");
                    }
                    break;
                }
            }

            auto result = create_int64_tensor(
                {arr.ptr(), static_cast<std::size_t>(arr.size())}, shape);
            if (result.has_error()) {
                return result.error();
            }
            ort_inputs.push_back(std::move(result).value());
            break;
        }
        case ElementType::Bool: {
            if (val.get_type() != godot::Variant::PACKED_BYTE_ARRAY) {
                std::ostringstream oss;
                oss << "Input '" << spec.name
                    << "' expects PackedByteArray (for bool), got variant type "
                    << static_cast<int>(val.get_type());
                return Error::make(ErrorCode::InvalidType, oss.str());
            }
            godot::PackedByteArray arr = val;

            std::vector<int64_t> shape = spec.shape;
            for (auto& dim : shape) {
                if (dim < 0) {
                    int64_t known = 1;
                    int dynamic_count = 0;
                    for (auto d : shape) {
                        if (d > 0) {
                            known *= d;
                        } else {
                            dynamic_count++;
                        }
                    }
                    if (dynamic_count == 1 && known > 0) {
                        dim = static_cast<int64_t>(arr.size()) / known;
                    } else {
                        return Error::make(ErrorCode::InvalidShape,
                                           "Multiple dynamic dimensions in input '" +
                                               spec.name + "'");
                    }
                    break;
                }
            }

            auto result = create_bool_tensor(
                {arr.ptr(), static_cast<std::size_t>(arr.size())}, shape);
            if (result.has_error()) {
                return result.error();
            }
            ort_inputs.push_back(std::move(result).value());
            break;
        }
        default: {
            std::ostringstream oss;
            oss << "Input '" << spec.name << "' has unsupported element type: "
                << element_type_name(spec.element_type);
            return Error::make(ErrorCode::InvalidType, oss.str());
        }
        }
    }

    return ort_inputs;
}

godot::Dictionary OrtSession::convert_outputs(std::vector<Ort::Value>& outputs) {
    godot::Dictionary result;
    const auto& specs = session_.output_specs();

    for (std::size_t i = 0; i < outputs.size() && i < specs.size(); ++i) {
        godot::String name(specs[i].name.c_str());
        auto& val = outputs[i];

        if (!val.IsTensor()) {
            godot::UtilityFunctions::push_warning(
                godot::String("[gonx] Output '") + name +
                godot::String("' is not a tensor, skipping"));
            continue;
        }

        auto elem_type = get_tensor_element_type(val);
        auto shape = get_tensor_shape(val);
        int64_t count = 1;
        for (auto dim : shape) {
            count *= dim;
        }

        switch (elem_type) {
        case ElementType::Float32: {
            const float* data = val.GetTensorData<float>();
            godot::PackedFloat32Array arr;
            arr.resize(count);
            memcpy(arr.ptrw(), data, static_cast<std::size_t>(count) * sizeof(float));
            result[name] = arr;
            break;
        }
        case ElementType::Int64: {
            const int64_t* data = val.GetTensorData<int64_t>();
            godot::PackedInt64Array arr;
            arr.resize(count);
            memcpy(arr.ptrw(), data, static_cast<std::size_t>(count) * sizeof(int64_t));
            result[name] = arr;
            break;
        }
        case ElementType::Bool: {
            const bool* data = val.GetTensorData<bool>();
            godot::PackedByteArray arr;
            arr.resize(count);
            for (int64_t j = 0; j < count; ++j) {
                arr[j] = data[j] ? 1 : 0;
            }
            result[name] = arr;
            break;
        }
        default:
            godot::UtilityFunctions::push_warning(
                godot::String("[gonx] Output '") + name +
                godot::String("' has unsupported type: ") +
                godot::String(element_type_name(elem_type)));
            break;
        }

        // Also provide shape metadata
        godot::PackedInt64Array gd_shape;
        gd_shape.resize(static_cast<int64_t>(shape.size()));
        for (std::size_t j = 0; j < shape.size(); ++j) {
            gd_shape[static_cast<int64_t>(j)] = shape[j];
        }
        result[name + godot::String("_shape")] = gd_shape;
    }

    return result;
}

godot::Dictionary OrtSession::run_inference(const godot::Dictionary& inputs) {
    std::lock_guard lock(mutex_);

    if (!session_.is_loaded()) {
        last_error_ = "No model loaded. Call load_model() first.";
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    auto prepared = prepare_inputs(inputs);
    if (prepared.has_error()) {
        last_error_ = godot::String(prepared.error().message.c_str());
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    auto result = session_.run(prepared.value());
    if (result.has_error()) {
        last_error_ = godot::String(result.error().message.c_str());
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    last_error_ = "";
    return convert_outputs(result.value());
}

void OrtSession::run_inference_async(const godot::Dictionary& inputs) {
    if (!session_.is_loaded()) {
        last_error_ = "No model loaded. Call load_model() first.";
        call_deferred("_on_async_failed", last_error_);
        return;
    }

    if (async_running_.load(std::memory_order_acquire)) {
        last_error_ = "Async inference already in progress.";
        call_deferred("_on_async_failed", last_error_);
        return;
    }

    // Join any previous thread
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    // Prepare inputs on the main thread (accesses Godot Variant data)
    auto prepared = prepare_inputs(inputs);
    if (prepared.has_error()) {
        last_error_ = godot::String(prepared.error().message.c_str());
        call_deferred("_on_async_failed", last_error_);
        return;
    }

    auto shared_inputs = std::make_shared<std::vector<Ort::Value>>(std::move(prepared).value());
    async_running_.store(true, std::memory_order_release);

    async_thread_ = std::thread([this, shared_inputs]() {
        auto result = session_.run(*shared_inputs);

        if (!async_running_.load(std::memory_order_acquire)) {
            return;  // Cancelled / shutting down
        }

        if (result.has_error()) {
            godot::String err(result.error().message.c_str());
            call_deferred("_on_async_failed", err);
        } else {
            auto output_dict = convert_outputs(result.value());
            call_deferred("_on_async_completed", output_dict);
        }

        async_running_.store(false, std::memory_order_release);
    });
}

void OrtSession::_on_async_completed(const godot::Dictionary& result) {
    last_error_ = "";
    emit_signal("inference_completed", result);
}

void OrtSession::_on_async_failed(const godot::String& error) {
    last_error_ = error;
    godot::UtilityFunctions::push_error(godot::String("[gonx] Async inference failed: ") + error);
    emit_signal("inference_failed", error);
}

godot::String OrtSession::get_last_error() const {
    return last_error_;
}

godot::String OrtSession::get_model_path() const {
    if (!session_.is_loaded()) {
        return {};
    }
    return godot::String(session_.model_path().string().c_str());
}

void OrtSession::_bind_methods() {
    using namespace godot;

    ClassDB::bind_method(D_METHOD("load_model", "path"), &OrtSession::load_model);
    ClassDB::bind_method(D_METHOD("load_model_with_config", "path", "config"),
                         &OrtSession::load_model_with_config);
    ClassDB::bind_method(D_METHOD("is_loaded"), &OrtSession::is_loaded);
    ClassDB::bind_method(D_METHOD("get_input_specs"), &OrtSession::get_input_specs);
    ClassDB::bind_method(D_METHOD("get_output_specs"), &OrtSession::get_output_specs);
    ClassDB::bind_method(D_METHOD("get_metadata"), &OrtSession::get_metadata);
    ClassDB::bind_method(D_METHOD("run_inference", "inputs"), &OrtSession::run_inference);
    ClassDB::bind_method(D_METHOD("run_inference_async", "inputs"),
                         &OrtSession::run_inference_async);
    ClassDB::bind_method(D_METHOD("get_last_error"), &OrtSession::get_last_error);
    ClassDB::bind_method(D_METHOD("get_model_path"), &OrtSession::get_model_path);

    // Internal deferred callbacks
    ClassDB::bind_method(D_METHOD("_on_async_completed", "result"),
                         &OrtSession::_on_async_completed);
    ClassDB::bind_method(D_METHOD("_on_async_failed", "error"),
                         &OrtSession::_on_async_failed);

    ADD_SIGNAL(MethodInfo("inference_completed",
                          PropertyInfo(Variant::DICTIONARY, "result")));
    ADD_SIGNAL(MethodInfo("inference_failed",
                          PropertyInfo(Variant::STRING, "error")));
}

}  // namespace gonx
