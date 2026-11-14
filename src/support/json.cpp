#include "support/json.h"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace mlc::json {

namespace {

[[noreturn]] void wrong_kind(const Value& v, const char* wanted) {
  throw Error(std::string("json: expected ") + wanted + ", found " + v.kind_name());
}

// Recursive descent: one function per grammar rule, each consuming exactly
// the text of the thing it parses and leaving `pos_` just past it.
class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value parse_document() {
    Value v = parse_value();
    skip_ws();
    if (pos_ != text_.size()) fail("unexpected text after the document");
    return v;
  }

 private:
  std::string_view text_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    size_t line = 1;
    for (size_t i = 0; i < pos_ && i < text_.size(); ++i) line += text_[i] == '\n';
    throw Error("json: line " + std::to_string(line) + ": " + what);
  }

  void skip_ws() {
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c != ' ' && c != '\n' && c != '\t' && c != '\r') break;
      ++pos_;
    }
  }

  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    ++pos_;
  }

  bool consume_literal(std::string_view word) {
    if (text_.substr(pos_, word.size()) != word) return false;
    pos_ += word.size();
    return true;
  }

  Value parse_value() {
    skip_ws();
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value(parse_string());
      case 't': if (consume_literal("true")) return Value(true); break;
      case 'f': if (consume_literal("false")) return Value(false); break;
      case 'n': if (consume_literal("null")) return Value(nullptr); break;
      default: return parse_number();
    }
    fail("invalid literal");
  }

  Value parse_object() {
    expect('{');
    Value::Object members;
    skip_ws();
    if (peek() == '}') { ++pos_; return Value(std::move(members)); }
    for (;;) {
      skip_ws();
      if (peek() != '"') fail("expected a string key");
      std::string key = parse_string();
      skip_ws();
      expect(':');
      members.push_back(Member{std::move(key), parse_value()});
      skip_ws();
      if (peek() == ',') { ++pos_; continue; }
      expect('}');
      return Value(std::move(members));
    }
  }

  Value parse_array() {
    expect('[');
    Value::Array items;
    skip_ws();
    if (peek() == ']') { ++pos_; return Value(std::move(items)); }
    for (;;) {
      items.push_back(parse_value());
      skip_ws();
      if (peek() == ',') { ++pos_; continue; }
      expect(']');
      return Value(std::move(items));
    }
  }

  unsigned parse_hex4() {
    if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
    unsigned code = 0;
    auto [end, ec] = std::from_chars(text_.data() + pos_, text_.data() + pos_ + 4, code, 16);
    if (ec != std::errc() || end != text_.data() + pos_ + 4) fail("bad \\u escape");
    pos_ += 4;
    return code;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    for (;;) {
      if (pos_ >= text_.size()) fail("unterminated string");
      char c = text_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') { out += c; continue; }
      if (pos_ >= text_.size()) fail("unterminated escape");
      switch (text_[pos_++]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = parse_hex4();
          // A code point above U+FFFF arrives as two \u escapes (a surrogate pair).
          if (cp >= 0xD800 && cp <= 0xDBFF && consume_literal("\\u")) {
            unsigned low = parse_hex4();
            if (low < 0xDC00 || low > 0xDFFF) fail("bad surrogate pair");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(out, cp);
          break;
        }
        default: fail("unknown escape");
      }
    }
  }

  Value parse_number() {
    size_t start = pos_;
    bool is_float = false;
    if (peek() == '-') ++pos_;
    while (pos_ < text_.size()) {
      char c = text_[pos_];
      if (c >= '0' && c <= '9') { ++pos_; continue; }
      if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { is_float = true; ++pos_; continue; }
      break;
    }
    std::string token(text_.substr(start, pos_ - start));
    if (token.empty() || token == "-") { pos_ = start; fail("expected a value"); }
    if (!is_float) {
      int64_t i = 0;
      auto [end, ec] = std::from_chars(token.data(), token.data() + token.size(), i);
      if (ec == std::errc() && end == token.data() + token.size()) return Value(i);
      // Too large for int64: fall through and keep it as a double.
    }
    char* end = nullptr;
    double d = std::strtod(token.c_str(), &end);
    if (end != token.c_str() + token.size()) { pos_ = start; fail("malformed number '" + token + "'"); }
    return Value(d);
  }
};

}  // namespace

bool Value::as_bool() const {
  if (auto* p = std::get_if<bool>(&v_)) return *p;
  wrong_kind(*this, "a bool");
}

int64_t Value::as_int() const {
  if (auto* p = std::get_if<int64_t>(&v_)) return *p;
  wrong_kind(*this, "an integer");
}

double Value::as_double() const {
  if (auto* p = std::get_if<double>(&v_)) return *p;
  if (auto* p = std::get_if<int64_t>(&v_)) return static_cast<double>(*p);
  wrong_kind(*this, "a number");
}

const std::string& Value::as_string() const {
  if (auto* p = std::get_if<std::string>(&v_)) return *p;
  wrong_kind(*this, "a string");
}

const Value::Array& Value::as_array() const {
  if (auto* p = std::get_if<Array>(&v_)) return *p;
  wrong_kind(*this, "an array");
}

const Value::Object& Value::as_object() const {
  if (auto* p = std::get_if<Object>(&v_)) return *p;
  wrong_kind(*this, "an object");
}

const Value* Value::find(std::string_view key) const {
  for (const Member& m : as_object())
    if (m.key == key) return &m.value;
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  if (const Value* v = find(key)) return *v;
  throw Error("json: missing key \"" + std::string(key) + "\"");
}

const Value& Value::at(size_t index) const {
  const Array& a = as_array();
  if (index >= a.size()) throw Error("json: array index " + std::to_string(index) + " out of range");
  return a[index];
}

size_t Value::size() const {
  if (auto* p = std::get_if<Array>(&v_)) return p->size();
  if (auto* p = std::get_if<Object>(&v_)) return p->size();
  wrong_kind(*this, "an array or object");
}

const char* Value::kind_name() const {
  static const char* names[] = {"null", "bool", "integer", "float", "string", "array", "object"};
  return names[v_.index()];
}

Value parse(std::string_view text) { return Parser(text).parse_document(); }

Value parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("json: cannot open " + path);
  std::ostringstream buf;
  buf << in.rdbuf();
  return parse(buf.str());
}

}  // namespace mlc::json
