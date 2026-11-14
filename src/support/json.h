// A small JSON reader. Read-only, strict, and just big enough for graph.json.
//
// We wrote this instead of vendoring a library because the whole job is about
// 200 lines, it keeps debug builds fast, and it lets integers stay integers:
// the importer has to tell `add(x, 0)` (an int64 add) from `mul(x, 0.5)`.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mlc::json {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Member;

class Value {
 public:
  using Array = std::vector<Value>;
  using Object = std::vector<Member>;  // insertion order kept; lookups are linear

  // Declared here, defined below Member: building or destroying a Value needs
  // the complete Member type, which in turn needs the complete Value type.
  Value();
  Value(std::nullptr_t);
  Value(bool b);
  Value(int64_t i);
  Value(double d);
  Value(std::string s);
  Value(Array a);
  Value(Object o);
  Value(const Value&);
  Value(Value&&) noexcept;
  Value& operator=(const Value&);
  Value& operator=(Value&&) noexcept;
  ~Value();

  bool is_null() const { return std::holds_alternative<std::monostate>(v_); }
  bool is_bool() const { return std::holds_alternative<bool>(v_); }
  bool is_int() const { return std::holds_alternative<int64_t>(v_); }
  bool is_float() const { return std::holds_alternative<double>(v_); }
  bool is_number() const { return is_int() || is_float(); }
  bool is_string() const { return std::holds_alternative<std::string>(v_); }
  bool is_array() const { return std::holds_alternative<Array>(v_); }
  bool is_object() const { return std::holds_alternative<Object>(v_); }

  // Each accessor throws json::Error when the value has a different kind.
  bool as_bool() const;
  int64_t as_int() const;
  double as_double() const;  // accepts an int or a float
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Object member access. find() returns nullptr when the key is absent.
  const Value* find(std::string_view key) const;
  const Value& at(std::string_view key) const;
  // Array element access with a bounds check.
  const Value& at(size_t index) const;
  size_t size() const;  // array or object

  const char* kind_name() const;

 private:
  std::variant<std::monostate, bool, int64_t, double, std::string, Array, Object> v_;
};

struct Member {
  std::string key;
  Value value;
};

inline Value::Value() = default;
inline Value::Value(std::nullptr_t) {}
inline Value::Value(bool b) : v_(b) {}
inline Value::Value(int64_t i) : v_(i) {}
inline Value::Value(double d) : v_(d) {}
inline Value::Value(std::string s) : v_(std::move(s)) {}
inline Value::Value(Array a) : v_(std::move(a)) {}
inline Value::Value(Object o) : v_(std::move(o)) {}
inline Value::Value(const Value&) = default;
inline Value::Value(Value&&) noexcept = default;
inline Value& Value::operator=(const Value&) = default;
inline Value& Value::operator=(Value&&) noexcept = default;
inline Value::~Value() = default;

// Parses a complete JSON document. Throws json::Error with a line number.
Value parse(std::string_view text);
Value parse_file(const std::string& path);

}  // namespace mlc::json
