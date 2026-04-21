#include "gonx/godot/ort_session_node.hpp"
#include "gonx/core/type_conversion.hpp"

#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <cstring>
#include <sstream>
#include <utility>

namespace gonx {

namespace {

godot::String to_godot_string(const std::string& value) {
    return godot::String(value.c_str());
}

}  // namespace

OrtSession::OrtSession() = default;

OrtSession::~OrtSession() {
    // Higher-level wrappers cancel in-flight work before releasing the session.
    // Avoid taking the session mutex again during RefCounted teardown, which can
    // race shutdown on macOS headless runs after the useful work is complete.
    if (load_thread_.joinable()) {
        load_thread_.join();
    }
    if (async_thread_.joinable()) {
        async_thread_.join();
    }
}

int OrtSession::load_model(const godot::String& path) {
    return load_model_with_config(path, {});
}

int OrtSession::load_model_with_config(const godot::String& path,
                                       const godot::Ref<OrtProviderConfig>& config) {
    std::thread finished_load_thread;
    std::thread finished_async_thread;

    {
        std::lock_guard lock(mutex_);

        if (load_running_.load(std::memory_order_acquire)) {
            last_error_ = "A model load is already in progress.";
            godot::UtilityFunctions::push_error(
                godot::String("[gonx] Failed to load model: ") + last_error_);
            return static_cast<int>(ErrorCode::InvalidArgument);
        }
        if (async_running_.load(std::memory_order_acquire)) {
            last_error_ = "Async inference is in progress. Cancel it before loading a new model.";
            godot::UtilityFunctions::push_error(
                godot::String("[gonx] Failed to load model: ") + last_error_);
            return static_cast<int>(ErrorCode::InvalidArgument);
        }

        if (load_thread_.joinable()) {
            finished_load_thread = std::move(load_thread_);
        }
        if (async_thread_.joinable()) {
            finished_async_thread = std::move(async_thread_);
        }
    }

    if (finished_load_thread.joinable()) {
        finished_load_thread.join();
    }
    if (finished_async_thread.joinable()) {
        finished_async_thread.join();
    }

    const godot::String resolved = resolve_model_path(path);
    std::string native_path = resolved.utf8().get_data();
    SessionConfig session_config = make_session_config(config);

    std::lock_guard lock(mutex_);
    auto status = session_.load(native_path, session_config);
    if (status.has_error()) {
        last_error_ = to_godot_string(status.error().message);
        godot::UtilityFunctions::push_error(
            godot::String("[gonx] Failed to load model: ") + last_error_);
        return static_cast<int>(status.error().code);
    }

    last_error_ = "";
    return 0;
}

int64_t OrtSession::load_model_async(const godot::String& path) {
    return load_model_with_config_async(path, {});
}

int64_t OrtSession::load_model_with_config_async(const godot::String& path,
                                                 const godot::Ref<OrtProviderConfig>& config) {
    const godot::String resolved = resolve_model_path(path);
    std::string native_path = resolved.utf8().get_data();
    SessionConfig session_config = make_session_config(config);

    int64_t request_id = 0;
    std::thread finished_load_thread;

    {
        std::lock_guard lock(mutex_);

        if (load_running_.load(std::memory_order_acquire)) {
            last_error_ = "A model load is already in progress.";
            call_deferred("_on_model_load_failed", int64_t(0),
                          static_cast<int>(ErrorCode::InvalidArgument), last_error_);
            return 0;
        }
        if (async_running_.load(std::memory_order_acquire)) {
            last_error_ = "Async inference is in progress. Cancel it before loading a new model.";
            call_deferred("_on_model_load_failed", int64_t(0),
                          static_cast<int>(ErrorCode::InvalidArgument), last_error_);
            return 0;
        }

        if (load_thread_.joinable()) {
            finished_load_thread = std::move(load_thread_);
        }

        request_id = next_request_id();
        session_ = InferenceSession();
        last_error_ = "";

        auto cancel_flag = std::make_shared<std::atomic_bool>(false);
        active_load_cancel_flag_ = cancel_flag;
        active_load_request_id_.store(request_id, std::memory_order_release);
        load_running_.store(true, std::memory_order_release);

        load_thread_ = std::thread([this, request_id, resolved, native_path,
                                    session_config, cancel_flag]() mutable {
            InferenceSession loaded_session;
            auto status = loaded_session.load(native_path, session_config);
            const bool cancelled = cancel_flag->load(std::memory_order_acquire);

            {
                std::lock_guard state_lock(mutex_);
                if (!cancelled && status.has_value()) {
                    session_ = std::move(loaded_session);
                }

                if (active_load_request_id_.load(std::memory_order_acquire) == request_id) {
                    active_load_request_id_.store(0, std::memory_order_release);
                }
                if (active_load_cancel_flag_ == cancel_flag) {
                    active_load_cancel_flag_.reset();
                }
                load_running_.store(false, std::memory_order_release);
            }

            if (cancelled) {
                call_deferred("_on_model_load_cancelled", request_id);
                return;
            }

            if (status.has_error()) {
                call_deferred("_on_model_load_failed", request_id,
                              static_cast<int>(status.error().code),
                              to_godot_string(status.error().message));
                return;
            }

            call_deferred("_on_model_load_completed", request_id, resolved);
        });
    }

    if (finished_load_thread.joinable()) {
        finished_load_thread.join();
    }

    emit_signal("model_load_started", request_id, path);
    return request_id;
}

bool OrtSession::is_loaded() const {
    std::lock_guard lock(mutex_);
    return session_.is_loaded();
}

bool OrtSession::is_loading() const {
    return load_running_.load(std::memory_order_acquire);
}

godot::Array OrtSession::get_input_specs() const {
    godot::Array result;

    std::lock_guard lock(mutex_);
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

    std::lock_guard lock(mutex_);
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

    std::lock_guard lock(mutex_);
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
                        std::ostringstream oss;
                        oss << "Input '" << spec.name
                            << "' has multiple dynamic dimensions; provide "
                               "explicit shape via input_shapes parameter";
                        return Error::make(ErrorCode::InvalidShape, oss.str());
                    }
                    break;
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

    if (load_running_.load(std::memory_order_acquire)) {
        last_error_ = "A model load is in progress. Wait for it to finish before running inference.";
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }
    if (async_running_.load(std::memory_order_acquire)) {
        last_error_ = "Async inference is already in progress.";
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }
    if (!session_.is_loaded()) {
        last_error_ = "No model loaded. Call load_model() first.";
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    auto prepared = prepare_inputs(inputs);
    if (prepared.has_error()) {
        last_error_ = to_godot_string(prepared.error().message);
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    auto result = session_.run(prepared.value());
    if (result.has_error()) {
        last_error_ = to_godot_string(result.error().message);
        godot::UtilityFunctions::push_error(godot::String("[gonx] ") + last_error_);
        return {};
    }

    last_error_ = "";
    return convert_outputs(result.value());
}

int64_t OrtSession::run_inference_async(const godot::Dictionary& inputs) {
    int64_t request_id = 0;
    std::thread finished_async_thread;

    {
        std::lock_guard lock(mutex_);

        if (load_running_.load(std::memory_order_acquire)) {
            last_error_ = "A model load is in progress. Wait for it to finish before running inference.";
            call_deferred("_on_async_failed", int64_t(0),
                          static_cast<int>(ErrorCode::InvalidArgument), last_error_);
            return 0;
        }
        if (!session_.is_loaded()) {
            last_error_ = "No model loaded. Call load_model() first.";
            call_deferred("_on_async_failed", int64_t(0),
                          static_cast<int>(ErrorCode::SessionNotLoaded), last_error_);
            return 0;
        }
        if (async_running_.load(std::memory_order_acquire)) {
            last_error_ = "Async inference already in progress.";
            call_deferred("_on_async_failed", int64_t(0),
                          static_cast<int>(ErrorCode::InvalidArgument), last_error_);
            return 0;
        }

        if (async_thread_.joinable()) {
            finished_async_thread = std::move(async_thread_);
        }

        auto prepared = prepare_inputs(inputs);
        if (prepared.has_error()) {
            last_error_ = to_godot_string(prepared.error().message);
            call_deferred("_on_async_failed", int64_t(0),
                          static_cast<int>(prepared.error().code), last_error_);
            return 0;
        }

        request_id = next_request_id();
        auto shared_inputs =
            std::make_shared<std::vector<Ort::Value>>(std::move(prepared).value());
        auto cancel_flag = std::make_shared<std::atomic_bool>(false);
        auto run_options = std::make_shared<Ort::RunOptions>(nullptr);

        active_inference_cancel_flag_ = cancel_flag;
        active_run_options_ = run_options;
        active_inference_request_id_.store(request_id, std::memory_order_release);
        async_running_.store(true, std::memory_order_release);
        last_error_ = "";

        async_thread_ = std::thread([this, request_id, shared_inputs, cancel_flag,
                                     run_options]() mutable {
            auto result = session_.run(*shared_inputs, run_options.get());
            const bool cancelled = cancel_flag->load(std::memory_order_acquire);

            {
                std::lock_guard state_lock(mutex_);
                if (active_inference_request_id_.load(std::memory_order_acquire) == request_id) {
                    active_inference_request_id_.store(0, std::memory_order_release);
                }
                if (active_inference_cancel_flag_ == cancel_flag) {
                    active_inference_cancel_flag_.reset();
                }
                if (active_run_options_ == run_options) {
                    active_run_options_.reset();
                }
                async_running_.store(false, std::memory_order_release);
            }

            if (cancelled) {
                call_deferred("_on_async_cancelled", request_id);
                return;
            }

            if (result.has_error()) {
                call_deferred("_on_async_failed", request_id,
                              static_cast<int>(result.error().code),
                              to_godot_string(result.error().message));
                return;
            }

            auto output_dict = convert_outputs(result.value());
            call_deferred("_on_async_completed", request_id, output_dict);
        });
    }

    if (finished_async_thread.joinable()) {
        finished_async_thread.join();
    }

    return request_id;
}

bool OrtSession::is_async_inference_running() const {
    return async_running_.load(std::memory_order_acquire);
}

void OrtSession::cancel(int64_t request_id) {
    std::shared_ptr<std::atomic_bool> load_cancel_flag;
    std::shared_ptr<std::atomic_bool> inference_cancel_flag;
    std::shared_ptr<Ort::RunOptions> run_options;

    {
        std::lock_guard lock(mutex_);

        const int64_t active_load_request_id =
            active_load_request_id_.load(std::memory_order_acquire);
        if ((request_id == 0 || request_id == active_load_request_id) &&
            active_load_cancel_flag_) {
            load_cancel_flag = active_load_cancel_flag_;
        }

        const int64_t active_inference_request_id =
            active_inference_request_id_.load(std::memory_order_acquire);
        if ((request_id == 0 || request_id == active_inference_request_id) &&
            active_inference_cancel_flag_) {
            inference_cancel_flag = active_inference_cancel_flag_;
            run_options = active_run_options_;
        }
    }

    if (load_cancel_flag) {
        load_cancel_flag->store(true, std::memory_order_release);
    }
    if (inference_cancel_flag) {
        inference_cancel_flag->store(true, std::memory_order_release);
    }
    if (run_options) {
        run_options->SetTerminate();
    }
}

godot::String OrtSession::get_last_error() const {
    std::lock_guard lock(mutex_);
    return last_error_;
}

godot::String OrtSession::get_model_path() const {
    std::lock_guard lock(mutex_);
    if (!session_.is_loaded()) {
        return {};
    }
    return godot::String(session_.model_path().string().c_str());
}

godot::String OrtSession::resolve_model_path(const godot::String& path) const {
    if (path.begins_with("res://") || path.begins_with("user://")) {
        return godot::ProjectSettings::get_singleton()->globalize_path(path);
    }
    return path;
}

SessionConfig OrtSession::make_session_config(
    const godot::Ref<OrtProviderConfig>& config) const {
    if (config.is_valid()) {
        return config->to_session_config();
    }
    return {};
}

int64_t OrtSession::next_request_id() {
    return request_sequence_.fetch_add(1, std::memory_order_acq_rel);
}

void OrtSession::_on_model_load_completed(int64_t request_id,
                                          const godot::String& model_path) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = "";
    }
    emit_signal("model_loaded", request_id, model_path);
}

void OrtSession::_on_model_load_failed(int64_t request_id, int error_code,
                                       const godot::String& error) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = error;
    }
    godot::UtilityFunctions::push_error(
        godot::String("[gonx] Async model load failed: ") + error);
    emit_signal("model_load_failed", request_id, error_code, error);
}

void OrtSession::_on_model_load_cancelled(int64_t request_id) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = "";
    }
    emit_signal("model_load_cancelled", request_id);
}

void OrtSession::_on_async_completed(int64_t request_id,
                                     const godot::Dictionary& result) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = "";
    }
    emit_signal("inference_completed", request_id, result);
}

void OrtSession::_on_async_failed(int64_t request_id, int error_code,
                                  const godot::String& error) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = error;
    }
    godot::UtilityFunctions::push_error(
        godot::String("[gonx] Async inference failed: ") + error);
    emit_signal("inference_failed", request_id, error_code, error);
}

void OrtSession::_on_async_cancelled(int64_t request_id) {
    {
        std::lock_guard lock(mutex_);
        last_error_ = "";
    }
    emit_signal("inference_cancelled", request_id);
}

void OrtSession::_bind_methods() {
    using namespace godot;

    ClassDB::bind_method(D_METHOD("load_model", "path"), &OrtSession::load_model);
    ClassDB::bind_method(D_METHOD("load_model_with_config", "path", "config"),
                         &OrtSession::load_model_with_config);
    ClassDB::bind_method(D_METHOD("load_model_async", "path"),
                         &OrtSession::load_model_async);
    ClassDB::bind_method(D_METHOD("load_model_with_config_async", "path", "config"),
                         &OrtSession::load_model_with_config_async);
    ClassDB::bind_method(D_METHOD("is_loaded"), &OrtSession::is_loaded);
    ClassDB::bind_method(D_METHOD("is_loading"), &OrtSession::is_loading);
    ClassDB::bind_method(D_METHOD("get_input_specs"), &OrtSession::get_input_specs);
    ClassDB::bind_method(D_METHOD("get_output_specs"), &OrtSession::get_output_specs);
    ClassDB::bind_method(D_METHOD("get_metadata"), &OrtSession::get_metadata);
    ClassDB::bind_method(D_METHOD("run_inference", "inputs"), &OrtSession::run_inference);
    ClassDB::bind_method(D_METHOD("run_inference_async", "inputs"),
                         &OrtSession::run_inference_async);
    ClassDB::bind_method(D_METHOD("is_async_inference_running"),
                         &OrtSession::is_async_inference_running);
    ClassDB::bind_method(D_METHOD("cancel", "request_id"), &OrtSession::cancel,
                         DEFVAL(int64_t(0)));
    ClassDB::bind_method(D_METHOD("get_last_error"), &OrtSession::get_last_error);
    ClassDB::bind_method(D_METHOD("get_model_path"), &OrtSession::get_model_path);

    ClassDB::bind_method(D_METHOD("_on_model_load_completed", "request_id", "model_path"),
                         &OrtSession::_on_model_load_completed);
    ClassDB::bind_method(D_METHOD("_on_model_load_failed", "request_id", "error_code", "error"),
                         &OrtSession::_on_model_load_failed);
    ClassDB::bind_method(D_METHOD("_on_model_load_cancelled", "request_id"),
                         &OrtSession::_on_model_load_cancelled);
    ClassDB::bind_method(D_METHOD("_on_async_completed", "request_id", "result"),
                         &OrtSession::_on_async_completed);
    ClassDB::bind_method(D_METHOD("_on_async_failed", "request_id", "error_code", "error"),
                         &OrtSession::_on_async_failed);
    ClassDB::bind_method(D_METHOD("_on_async_cancelled", "request_id"),
                         &OrtSession::_on_async_cancelled);

    ADD_SIGNAL(MethodInfo("model_load_started",
                          PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::STRING, "path")));
    ADD_SIGNAL(MethodInfo("model_loaded",
                          PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::STRING, "path")));
    ADD_SIGNAL(MethodInfo("model_load_failed",
                          PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::INT, "error_code"),
                          PropertyInfo(Variant::STRING, "error")));
    ADD_SIGNAL(MethodInfo("model_load_cancelled",
                          PropertyInfo(Variant::INT, "request_id")));
    ADD_SIGNAL(MethodInfo("inference_completed",
                          PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::DICTIONARY, "result")));
    ADD_SIGNAL(MethodInfo("inference_failed",
                          PropertyInfo(Variant::INT, "request_id"),
                          PropertyInfo(Variant::INT, "error_code"),
                          PropertyInfo(Variant::STRING, "error")));
    ADD_SIGNAL(MethodInfo("inference_cancelled",
                          PropertyInfo(Variant::INT, "request_id")));
}

}  // namespace gonx
