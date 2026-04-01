#include "gonx/core/environment.hpp"
#include "gonx/godot/ort_model_metadata.hpp"
#include "gonx/godot/ort_provider_config.hpp"
#include "gonx/godot/ort_session_node.hpp"
#include "gonx/godot/ort_tensor_spec.hpp"

#include <gdextension_interface.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_gonx_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Ensure the ORT environment is initialized
    gonx::OrtEnvironment::instance();

    ClassDB::register_class<gonx::OrtTensorSpec>();
    ClassDB::register_class<gonx::OrtModelMetadata>();
    ClassDB::register_class<gonx::OrtProviderConfig>();
    ClassDB::register_class<gonx::OrtSession>();
}

void uninitialize_gonx_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    gonx::OrtEnvironment::shutdown();
}

extern "C" {

GDExtensionBool GDE_EXPORT gonx_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization) {
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_gonx_module);
    init_obj.register_terminator(uninitialize_gonx_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

    return init_obj.init();
}

}  // extern "C"
