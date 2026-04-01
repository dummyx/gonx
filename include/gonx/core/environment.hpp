#pragma once

#include <onnxruntime_cxx_api.h>

namespace gonx {

/// Process-wide ONNX Runtime environment.
/// Must be initialized before any session creation and torn down after all sessions are destroyed.
/// Thread-safe after initialization.
class OrtEnvironment {
public:
    OrtEnvironment(const OrtEnvironment&) = delete;
    OrtEnvironment& operator=(const OrtEnvironment&) = delete;
    OrtEnvironment(OrtEnvironment&&) = delete;
    OrtEnvironment& operator=(OrtEnvironment&&) = delete;

    /// Get the singleton instance. Lazily initializes on first call.
    [[nodiscard]] static OrtEnvironment& instance();

    /// Access the underlying Ort::Env. Only valid after instance() has been called.
    [[nodiscard]] Ort::Env& env() noexcept { return env_; }

    /// Shut down the environment. Call during GDExtension uninitialize.
    /// After this, no further ORT operations are valid.
    static void shutdown();

    ~OrtEnvironment();

private:
    OrtEnvironment();

    Ort::Env env_;
};

}  // namespace gonx
