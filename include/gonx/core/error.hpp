#pragma once

#include <optional>
#include <string>
#include <variant>

namespace gonx {

enum class ErrorCode {
    Ok,
    InvalidArgument,
    InvalidModel,
    InvalidShape,
    InvalidType,
    SessionNotLoaded,
    InferenceFailed,
    ProviderNotAvailable,
    InternalError,
};

struct Error {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    [[nodiscard]] bool is_ok() const noexcept { return code == ErrorCode::Ok; }

    [[nodiscard]] static Error ok() noexcept { return {ErrorCode::Ok, {}}; }

    [[nodiscard]] static Error make(ErrorCode code, std::string message) {
        return {code, std::move(message)};
    }
};

/// Result type: holds either a value T or an Error.
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}  // NOLINT(google-explicit-constructor)
    Result(Error error) : data_(std::move(error)) {}  // NOLINT(google-explicit-constructor)

    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    [[nodiscard]] bool has_error() const noexcept {
        return std::holds_alternative<Error>(data_);
    }

    [[nodiscard]] const T& value() const& { return std::get<T>(data_); }
    [[nodiscard]] T& value() & { return std::get<T>(data_); }
    [[nodiscard]] T&& value() && { return std::get<T>(std::move(data_)); }

    [[nodiscard]] const Error& error() const& { return std::get<Error>(data_); }

    /// Convenience: get value or a default
    [[nodiscard]] T value_or(T fallback) const& {
        if (has_value()) {
            return value();
        }
        return fallback;
    }

private:
    std::variant<T, Error> data_;
};

/// Specialization for void results (status only)
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}  // NOLINT(google-explicit-constructor)

    [[nodiscard]] bool has_value() const noexcept { return !error_.has_value(); }
    [[nodiscard]] bool has_error() const noexcept { return error_.has_value(); }

    [[nodiscard]] const Error& error() const& { return error_.value(); }

    [[nodiscard]] static Result ok() { return {}; }

private:
    std::optional<Error> error_;
};

using Status = Result<void>;

}  // namespace gonx
