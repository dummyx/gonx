#include "gonx/core/session.hpp"
#include "gonx/core/type_conversion.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <filesystem>

using namespace gonx;

static std::filesystem::path fixtures_dir() {
    return std::filesystem::path(GONX_TEST_FIXTURES_DIR);
}

TEST_CASE("Inference: add_floats produces correct result", "[inference]") {
    auto path = fixtures_dir() / "add_floats.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    REQUIRE(session.load(path).has_value());

    // A = [[1,2,3],[4,5,6]], B = [[10,20,30],[40,50,60]]
    std::array<float, 6> a_data = {1, 2, 3, 4, 5, 6};
    std::array<float, 6> b_data = {10, 20, 30, 40, 50, 60};
    std::array<int64_t, 2> shape = {2, 3};

    auto a_tensor = create_float_tensor(a_data, shape);
    auto b_tensor = create_float_tensor(b_data, shape);
    REQUIRE(a_tensor.has_value());
    REQUIRE(b_tensor.has_value());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(a_tensor).value());
    inputs.push_back(std::move(b_tensor).value());

    auto result = session.run(inputs);
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == 1);

    auto& output = result.value()[0];
    CHECK(output.IsTensor());

    auto out_shape = get_tensor_shape(output);
    REQUIRE(out_shape.size() == 2);
    CHECK(out_shape[0] == 2);
    CHECK(out_shape[1] == 3);

    const float* out_data = output.GetTensorData<float>();
    // Expected: A + B = {11, 22, 33, 44, 55, 66}
    CHECK_THAT(out_data[0], Catch::Matchers::WithinAbs(11.0, 1e-5));
    CHECK_THAT(out_data[1], Catch::Matchers::WithinAbs(22.0, 1e-5));
    CHECK_THAT(out_data[2], Catch::Matchers::WithinAbs(33.0, 1e-5));
    CHECK_THAT(out_data[3], Catch::Matchers::WithinAbs(44.0, 1e-5));
    CHECK_THAT(out_data[4], Catch::Matchers::WithinAbs(55.0, 1e-5));
    CHECK_THAT(out_data[5], Catch::Matchers::WithinAbs(66.0, 1e-5));
}

TEST_CASE("Inference: identity_float passes through data", "[inference]") {
    auto path = fixtures_dir() / "identity_float.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    REQUIRE(session.load(path).has_value());

    std::array<float, 4> data = {1.5f, 2.5f, 3.5f, 4.5f};
    std::array<int64_t, 2> shape = {1, 4};

    auto tensor = create_float_tensor(data, shape);
    REQUIRE(tensor.has_value());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(tensor).value());

    auto result = session.run(inputs);
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == 1);

    const float* out_data = result.value()[0].GetTensorData<float>();
    for (int i = 0; i < 4; ++i) {
        CHECK_THAT(out_data[i], Catch::Matchers::WithinAbs(data[static_cast<std::size_t>(i)], 1e-6));
    }
}

TEST_CASE("Inference: int64_add produces correct result", "[inference]") {
    auto path = fixtures_dir() / "int64_add.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    REQUIRE(session.load(path).has_value());

    std::array<int64_t, 2> a_data = {100, 200};
    std::array<int64_t, 2> b_data = {10, 20};
    std::array<int64_t, 1> shape = {2};

    auto a_tensor = create_int64_tensor(a_data, shape);
    auto b_tensor = create_int64_tensor(b_data, shape);
    REQUIRE(a_tensor.has_value());
    REQUIRE(b_tensor.has_value());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(a_tensor).value());
    inputs.push_back(std::move(b_tensor).value());

    auto result = session.run(inputs);
    REQUIRE(result.has_value());

    const int64_t* out_data = result.value()[0].GetTensorData<int64_t>();
    CHECK(out_data[0] == 110);
    CHECK(out_data[1] == 220);
}

TEST_CASE("Inference: wrong input count fails gracefully", "[inference]") {
    auto path = fixtures_dir() / "add_floats.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    REQUIRE(session.load(path).has_value());

    // Only provide 1 input when model expects 2
    std::array<float, 6> data = {1, 2, 3, 4, 5, 6};
    std::array<int64_t, 2> shape = {2, 3};
    auto tensor = create_float_tensor(data, shape);
    REQUIRE(tensor.has_value());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(tensor).value());

    auto result = session.run(inputs);
    CHECK(result.has_error());
    CHECK(result.error().code == ErrorCode::InvalidArgument);
}

TEST_CASE("Inference: dynamic batch with different batch sizes", "[inference]") {
    auto path = fixtures_dir() / "dynamic_batch.onnx";
    if (!std::filesystem::exists(path)) {
        SKIP("Test fixture not found: " << path);
    }

    InferenceSession session;
    REQUIRE(session.load(path).has_value());

    // Batch size 2
    {
        std::array<float, 6> data = {1, 2, 3, 4, 5, 6};
        std::array<int64_t, 2> shape = {2, 3};
        auto tensor = create_float_tensor(data, shape);
        REQUIRE(tensor.has_value());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(tensor).value());

        auto result = session.run(inputs);
        REQUIRE(result.has_value());
        auto out_shape = get_tensor_shape(result.value()[0]);
        CHECK(out_shape[0] == 2);
        CHECK(out_shape[1] == 3);
    }

    // Batch size 5
    {
        std::array<float, 15> data = {};
        std::array<int64_t, 2> shape = {5, 3};
        auto tensor = create_float_tensor(data, shape);
        REQUIRE(tensor.has_value());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(tensor).value());

        auto result = session.run(inputs);
        REQUIRE(result.has_value());
        auto out_shape = get_tensor_shape(result.value()[0]);
        CHECK(out_shape[0] == 5);
        CHECK(out_shape[1] == 3);
    }
}
