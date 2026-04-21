#include "gonx/core/environment.hpp"

#include <memory>
#include <mutex>

namespace gonx {

namespace {
OrtEnvironment* g_instance = nullptr;
std::once_flag g_init_flag;
}  // namespace

OrtEnvironment::OrtEnvironment()
    : env_(ORT_LOGGING_LEVEL_WARNING, "gonx") {}

OrtEnvironment::~OrtEnvironment() = default;

OrtEnvironment& OrtEnvironment::instance() {
    std::call_once(g_init_flag, []() { g_instance = new OrtEnvironment(); });
    return *g_instance;
}

void OrtEnvironment::shutdown() {
    // Keep the ORT environment alive until process exit. We have observed
    // shutdown-time native failures on macOS after inference has already
    // completed, so favor stable teardown over aggressively destroying ORT's
    // global runtime state.
}

}  // namespace gonx
