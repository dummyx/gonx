#include "gonx/core/session.hpp"
#include "gonx/core/environment.hpp"
#include "gonx/core/type_conversion.hpp"

#include <sstream>
#include <utility>

namespace gonx {

struct InferenceSession::Impl {
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TensorSpec> input_specs;
    std::vector<TensorSpec> output_specs;
    ModelMetadata metadata;
    std::filesystem::path model_path;

    // Cached name strings for Run() — ORT wants const char* arrays
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
};

InferenceSession::InferenceSession() : impl_(std::make_unique<Impl>()) {}
InferenceSession::~InferenceSession() = default;

InferenceSession::InferenceSession(InferenceSession&&) noexcept = default;
InferenceSession& InferenceSession::operator=(InferenceSession&&) noexcept = default;

namespace {

TensorSpec extract_tensor_spec(Ort::Session& session, std::size_t index, bool is_input,
                               Ort::AllocatorWithDefaultOptions& allocator) {
    TensorSpec spec;

    // Get name
    auto name = is_input ? session.GetInputNameAllocated(index, allocator)
                         : session.GetOutputNameAllocated(index, allocator);
    spec.name = name.get();

    // Get type info
    auto type_info = is_input ? session.GetInputTypeInfo(index)
                              : session.GetOutputTypeInfo(index);

    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        spec.element_type = from_ort_element_type(tensor_info.GetElementType());
        spec.shape = tensor_info.GetShape();
    }

    return spec;
}

Ort::SessionOptions build_session_options(const SessionConfig& config) {
    Ort::SessionOptions options;

    if (config.intra_op_num_threads > 0) {
        options.SetIntraOpNumThreads(config.intra_op_num_threads);
    }
    if (config.inter_op_num_threads > 0) {
        options.SetInterOpNumThreads(config.inter_op_num_threads);
    }

    // Map optimization level
    GraphOptimizationLevel opt_level = ORT_ENABLE_ALL;
    switch (config.optimization_level) {
    case 0:
        opt_level = ORT_DISABLE_ALL;
        break;
    case 1:
        opt_level = ORT_ENABLE_BASIC;
        break;
    case 2:
        opt_level = ORT_ENABLE_EXTENDED;
        break;
    default:
        opt_level = ORT_ENABLE_ALL;
        break;
    }
    options.SetGraphOptimizationLevel(opt_level);

    if (!config.optimized_model_path.empty()) {
#ifdef _WIN32
        std::wstring wide_path(config.optimized_model_path.begin(),
                               config.optimized_model_path.end());
        options.SetOptimizedModelFilePath(wide_path.c_str());
#else
        options.SetOptimizedModelFilePath(config.optimized_model_path.c_str());
#endif
    }

    return options;
}

ModelMetadata extract_metadata(Ort::Session& session,
                               Ort::AllocatorWithDefaultOptions& allocator) {
    ModelMetadata meta;

    try {
        auto model_meta = session.GetModelMetadata();

        auto producer = model_meta.GetProducerNameAllocated(allocator);
        meta.producer_name = producer.get();

        auto graph = model_meta.GetGraphNameAllocated(allocator);
        meta.graph_name = graph.get();

        auto desc = model_meta.GetGraphDescriptionAllocated(allocator);
        meta.graph_description = desc.get();

        auto domain = model_meta.GetDomainAllocated(allocator);
        meta.domain = domain.get();

        meta.version = model_meta.GetVersion();

        auto keys = model_meta.GetCustomMetadataMapKeysAllocated(allocator);
        for (const auto& key : keys) {
            auto val = model_meta.LookupCustomMetadataMapAllocated(key.get(), allocator);
            meta.custom_metadata[key.get()] = val.get();
        }
    } catch (const Ort::Exception& e) {
        // Metadata extraction is best-effort; some models have none
        (void)e;
    }

    return meta;
}

}  // namespace

Status InferenceSession::load(const std::filesystem::path& model_path,
                              const SessionConfig& config) {
    // Validate file exists
    std::error_code ec;
    if (!std::filesystem::exists(model_path, ec) || ec) {
        std::ostringstream oss;
        oss << "Model file not found: " << model_path;
        return Error::make(ErrorCode::InvalidModel, oss.str());
    }

    // Build session options
    auto options = build_session_options(config);

    // Create the ORT session
    try {
#ifdef _WIN32
        impl_->session = std::make_unique<Ort::Session>(
            OrtEnvironment::instance().env(),
            model_path.wstring().c_str(),
            options);
#else
        impl_->session = std::make_unique<Ort::Session>(
            OrtEnvironment::instance().env(),
            model_path.string().c_str(),
            options);
#endif
    } catch (const Ort::Exception& e) {
        std::ostringstream oss;
        oss << "Failed to load model '" << model_path << "': " << e.what();
        return Error::make(ErrorCode::InvalidModel, oss.str());
    }

    impl_->model_path = model_path;

    // Extract input specs
    auto num_inputs = impl_->session->GetInputCount();
    impl_->input_specs.clear();
    impl_->input_specs.reserve(num_inputs);
    impl_->input_names.clear();
    impl_->input_name_ptrs.clear();

    for (std::size_t i = 0; i < num_inputs; ++i) {
        auto spec = extract_tensor_spec(*impl_->session, i, true, impl_->allocator);
        impl_->input_names.push_back(spec.name);
        impl_->input_specs.push_back(std::move(spec));
    }

    // Extract output specs
    auto num_outputs = impl_->session->GetOutputCount();
    impl_->output_specs.clear();
    impl_->output_specs.reserve(num_outputs);
    impl_->output_names.clear();
    impl_->output_name_ptrs.clear();

    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto spec = extract_tensor_spec(*impl_->session, i, false, impl_->allocator);
        impl_->output_names.push_back(spec.name);
        impl_->output_specs.push_back(std::move(spec));
    }

    // Build const char* arrays for Run()
    for (const auto& name : impl_->input_names) {
        impl_->input_name_ptrs.push_back(name.c_str());
    }
    for (const auto& name : impl_->output_names) {
        impl_->output_name_ptrs.push_back(name.c_str());
    }

    // Extract metadata
    impl_->metadata = extract_metadata(*impl_->session, impl_->allocator);

    return Status::ok();
}

bool InferenceSession::is_loaded() const noexcept {
    return impl_ && impl_->session != nullptr;
}

const std::vector<TensorSpec>& InferenceSession::input_specs() const noexcept {
    return impl_->input_specs;
}

const std::vector<TensorSpec>& InferenceSession::output_specs() const noexcept {
    return impl_->output_specs;
}

const ModelMetadata& InferenceSession::metadata() const noexcept {
    return impl_->metadata;
}

const std::filesystem::path& InferenceSession::model_path() const noexcept {
    return impl_->model_path;
}

Result<std::vector<Ort::Value>> InferenceSession::run(std::vector<Ort::Value>& inputs) {
    if (!is_loaded()) {
        return Error::make(ErrorCode::SessionNotLoaded,
                           "No model is loaded. Call load() first.");
    }

    if (inputs.size() != impl_->input_specs.size()) {
        std::ostringstream oss;
        oss << "Input count mismatch: model expects " << impl_->input_specs.size()
            << " inputs, got " << inputs.size();
        return Error::make(ErrorCode::InvalidArgument, oss.str());
    }

    try {
        auto results = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_ptrs.data(),
            inputs.data(),
            inputs.size(),
            impl_->output_name_ptrs.data(),
            impl_->output_specs.size());

        return results;
    } catch (const Ort::Exception& e) {
        std::ostringstream oss;
        oss << "Inference failed: " << e.what();
        return Error::make(ErrorCode::InferenceFailed, oss.str());
    }
}

}  // namespace gonx
