#include "gonx/core/error.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace gonx;

TEST_CASE("Error::ok creates a no-error state", "[error]") {
    auto err = Error::ok();
    CHECK(err.is_ok());
    CHECK(err.code == ErrorCode::Ok);
    CHECK(err.message.empty());
}

TEST_CASE("Error::make creates an error with code and message", "[error]") {
    auto err = Error::make(ErrorCode::InvalidModel, "bad model");
    CHECK_FALSE(err.is_ok());
    CHECK(err.code == ErrorCode::InvalidModel);
    CHECK(err.message == "bad model");
}

TEST_CASE("Result<int> holds a value", "[error]") {
    Result<int> r(42);
    CHECK(r.has_value());
    CHECK_FALSE(r.has_error());
    CHECK(r.value() == 42);
}

TEST_CASE("Result<int> holds an error", "[error]") {
    Result<int> r(Error::make(ErrorCode::InternalError, "oops"));
    CHECK_FALSE(r.has_value());
    CHECK(r.has_error());
    CHECK(r.error().code == ErrorCode::InternalError);
    CHECK(r.error().message == "oops");
}

TEST_CASE("Result<int>::value_or returns fallback on error", "[error]") {
    Result<int> r(Error::make(ErrorCode::InternalError, "oops"));
    CHECK(r.value_or(99) == 99);
}

TEST_CASE("Result<int>::value_or returns value on success", "[error]") {
    Result<int> r(42);
    CHECK(r.value_or(99) == 42);
}

TEST_CASE("Status (Result<void>) success", "[error]") {
    auto s = Status::ok();
    CHECK(s.has_value());
    CHECK_FALSE(s.has_error());
}

TEST_CASE("Status (Result<void>) failure", "[error]") {
    Status s(Error::make(ErrorCode::InvalidArgument, "bad"));
    CHECK_FALSE(s.has_value());
    CHECK(s.has_error());
    CHECK(s.error().code == ErrorCode::InvalidArgument);
}
