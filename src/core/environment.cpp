#include "gonx/core/environment.hpp"

#include <memory>
#include <mutex>

namespace gonx {

namespace {
std::unique_ptr<OrtEnvironment> g_instance;
std::once_flag g_init_flag;
}  // namespace

OrtEnvironment::OrtEnvironment()
    : env_(ORT_LOGGING_LEVEL_WARNING, "gonx") {}

OrtEnvironment::~OrtEnvironment() = default;

OrtEnvironment& OrtEnvironment::instance() {
    std::call_once(g_init_flag, []() { g_instance = std::unique_ptr<OrtEnvironment>(new OrtEnvironment()); });
    return *g_instance;
}

void OrtEnvironment::shutdown() {
    g_instance.reset();
}

}  // namespace gonx
