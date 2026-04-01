#pragma once

#include "gonx/core/session.hpp"

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

namespace gonx {

/// Godot-facing wrapper for ModelMetadata.
class OrtModelMetadata : public godot::RefCounted {
    GDCLASS(OrtModelMetadata, godot::RefCounted)

public:
    OrtModelMetadata() = default;

    void set_from_metadata(const ModelMetadata& meta);

    godot::String get_producer_name() const;
    godot::String get_graph_name() const;
    godot::String get_graph_description() const;
    godot::String get_domain() const;
    int64_t get_version() const;
    godot::Dictionary get_custom_metadata() const;

protected:
    static void _bind_methods();

private:
    ModelMetadata meta_;
};

}  // namespace gonx
