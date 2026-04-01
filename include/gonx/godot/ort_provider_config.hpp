#pragma once

#include "gonx/core/provider.hpp"

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

namespace gonx {

/// Godot-facing resource for configuring ORT session options.
/// Can be saved/loaded as a Godot resource and shared across sessions.
class OrtProviderConfig : public godot::Resource {
    GDCLASS(OrtProviderConfig, godot::Resource)

public:
    OrtProviderConfig() = default;

    void set_provider(const godot::String& provider);
    godot::String get_provider() const;

    void set_intra_op_threads(int threads);
    int get_intra_op_threads() const;

    void set_inter_op_threads(int threads);
    int get_inter_op_threads() const;

    void set_optimization_level(int level);
    int get_optimization_level() const;

    void set_optimized_model_path(const godot::String& path);
    godot::String get_optimized_model_path() const;

    /// Build a core SessionConfig from this resource's state.
    [[nodiscard]] SessionConfig to_session_config() const;

    /// List providers available in the current ORT build.
    static godot::PackedStringArray get_available_providers();

protected:
    static void _bind_methods();

private:
    godot::String provider_ = "CPU";
    int intra_op_threads_ = 0;
    int inter_op_threads_ = 0;
    int optimization_level_ = 99;
    godot::String optimized_model_path_;
};

}  // namespace gonx
