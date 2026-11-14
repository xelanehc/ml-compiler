// A test framework in one screen.
//
//   TEST(name) { CHECK(1 + 1 == 2); }
//
// TEST defines a function and a static object whose constructor adds that
// function to a global list before main() runs. test_main.cpp walks the list.
// CHECK records a failure and keeps going, so one run reports every problem.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mlc::test {

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

inline int& failures() {
  static int n = 0;
  return n;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void fail(const char* file, int line, const std::string& what) {
  std::printf("    %s:%d: %s\n", file, line, what.c_str());
  ++failures();
}

}  // namespace mlc::test

#define TEST(name)                                                    \
  static void test_##name();                                          \
  static ::mlc::test::Registrar registrar_##name(#name, test_##name); \
  static void test_##name()

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) ::mlc::test::fail(__FILE__, __LINE__, "CHECK(" #cond ") failed"); \
  } while (0)

// Two floats within `tol` of each other.
#define CHECK_NEAR(a, b, tol)                                                                                     \
  do {                                                                                                            \
    double a_ = (a), b_ = (b);                                                                                    \
    if (!(std::fabs(a_ - b_) <= (tol)))                                                                           \
      ::mlc::test::fail(__FILE__, __LINE__, #a " = " + std::to_string(a_) + ", expected " + std::to_string(b_)); \
  } while (0)

// The expression must throw an exception of the given type.
#define CHECK_THROWS(expr, type)                                                              \
  do {                                                                                        \
    bool threw_ = false;                                                                      \
    try {                                                                                     \
      (void)(expr);                                                                           \
    } catch (const type&) {                                                                   \
      threw_ = true;                                                                          \
    }                                                                                         \
    if (!threw_) ::mlc::test::fail(__FILE__, __LINE__, #expr " did not throw " #type);        \
  } while (0)
