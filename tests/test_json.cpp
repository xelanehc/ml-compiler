#include "check.h"
#include "support/json.h"

using namespace mlc;

TEST(json_scalars) {
  CHECK(json::parse("null").is_null());
  CHECK(json::parse("true").as_bool());
  CHECK(!json::parse(" false ").as_bool());
  CHECK(json::parse("\"hi\"").as_string() == "hi");
}

TEST(json_keeps_ints_and_floats_apart) {
  // The importer depends on this: `0` is an int64 scalar, `0.5` and `3.0` are floats.
  CHECK(json::parse("0").is_int());
  CHECK(json::parse("-17").as_int() == -17);
  CHECK(json::parse("3.0").is_float());
  CHECK(json::parse("1e-05").is_float());
  CHECK_NEAR(json::parse("0.7978845608028654").as_double(), 0.7978845608028654, 0);
  CHECK(json::parse("9223372036854775807").as_int() == INT64_MAX);
  CHECK_NEAR(json::parse("7").as_double(), 7.0, 0);  // as_double accepts an int
}

TEST(json_nested) {
  json::Value v = json::parse(R"({"name": "view", "args": [{"ref": "x"}, [-1, 16]], "kwargs": {}})");
  CHECK(v.at("name").as_string() == "view");
  CHECK(v.at("args").size() == 2);
  CHECK(v.at("args").at(0).at("ref").as_string() == "x");
  CHECK(v.at("args").at(1).at(1).as_int() == 16);
  CHECK(v.at("kwargs").size() == 0);
  CHECK(v.find("missing") == nullptr);
}

TEST(json_string_escapes) {
  CHECK(json::parse(R"("a\"b\\c\n")").as_string() == "a\"b\\c\n");
  CHECK(json::parse(R"("é")").as_string() == "\xC3\xA9");          // e-acute, 2 bytes of UTF-8
  CHECK(json::parse(R"("😀")").as_string() == "\xF0\x9F\x98\x80");  // a surrogate pair
}

TEST(json_rejects_malformed_input) {
  CHECK_THROWS(json::parse(""), json::Error);
  CHECK_THROWS(json::parse("[1, 2"), json::Error);
  CHECK_THROWS(json::parse("[1,]"), json::Error);
  CHECK_THROWS(json::parse("{\"a\" 1}"), json::Error);
  CHECK_THROWS(json::parse("1 2"), json::Error);
  CHECK_THROWS(json::parse("-Infinity"), json::Error);  // Python would write this; it is not JSON
  CHECK_THROWS(json::parse("\"abc"), json::Error);
}

TEST(json_wrong_kind_throws) {
  json::Value v = json::parse("{\"a\": [1]}");
  CHECK_THROWS(v.as_int(), json::Error);
  CHECK_THROWS(v.at("b"), json::Error);
  CHECK_THROWS(v.at("a").at(5), json::Error);
  CHECK_THROWS(json::parse("1.5").as_int(), json::Error);
}
