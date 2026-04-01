#include "gonx/core/session.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

using namespace gonx;

static std::filesystem::path fixtures_dir() {
    return std::filesystem::path(GONX_TEST_FIXTURES_DIR);
}

TEST_CASE("InferenceSession starts unloaded", "[session]") {
    InferenceSession session;
    CHECK_FALSE(session.is_loaded());
    CHECK(session.input_specs().empty());
    CHECK(session.output_specs().empty());
}

TEST_CASE("InferenceSession load fails on nonexistent file", "[session]") {
    InferenceSession session;
    auto status = session.load("/nonexistent/model.onnx");
    CHECK(status.has_error());
    CHECK(status.error().code == ErrorCode::InvalidModel);
    CHECK_FALSE(session.is_loaded());
}

TEST_CASE("InferenceSession loads add_floats.onnx", "[session]") {
    auto path = fixtures_dir() / "add_floats.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    auto status = session.load(path);
    REQUIRE(status.has_value());
    CHECK(session.is_loaded());

    CHECK(session.input_specs().size() == 2);
    CHECK(session.output_specs().size() == 1);

    CHECK(session.input_specs()[0].name == "A");
    CHECK(session.input_specs()[0].element_type == ElementType::Float32);
    CHECK(session.input_specs()[0].shape == std::vector<int64_t>{2, 3});

    CHECK(session.input_specs()[1].name == "B");

    CHECK(session.output_specs()[0].name == "C");
    CHECK(session.output_specs()[0].element_type == ElementType::Float32);
}

TEST_CASE("InferenceSession loads identity_float.onnx", "[session]") {
    auto path = fixtures_dir() / "identity_float.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    auto status = session.load(path);
    REQUIRE(status.has_value());

    REQUIRE(session.input_specs().size() == 1);
    CHECK(session.input_specs()[0].name == "X");
    CHECK(session.input_specs()[0].shape == std::vector<int64_t>{1, 4});

    REQUIRE(session.output_specs().size() == 1);
    CHECK(session.output_specs()[0].name == "Y");
}

TEST_CASE("InferenceSession metadata from add_floats.onnx", "[session]") {
    auto path = fixtures_dir() / "add_floats.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    auto status = session.load(path);
    REQUIRE(status.has_value());

    auto& meta = session.metadata();
    CHECK(meta.producer_name == "gonx-test-gen");
    CHECK(meta.graph_name == "add_graph");
}

TEST_CASE("InferenceSession run fails without loading", "[session]") {
    InferenceSession session;
    std::vector<Ort::Value> inputs;
    auto result = session.run(inputs);
    CHECK(result.has_error());
    CHECK(result.error().code == ErrorCode::SessionNotLoaded);
}

TEST_CASE("InferenceSession detects dynamic batch dimension", "[session]") {
    auto path = fixtures_dir() / "dynamic_batch.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    auto status = session.load(path);
    REQUIRE(status.has_value());

    REQUIRE(session.input_specs().size() == 1);
    CHECK_FALSE(session.input_specs()[0].is_static_shape());
    CHECK(session.input_specs()[0].shape[0] == -1);
    CHECK(session.input_specs()[0].shape[1] == 3);
}
