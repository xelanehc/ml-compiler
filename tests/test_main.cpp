#include <cstring>
#include <exception>

#include "check.h"

// Runs every registered test, or only those whose name contains argv[1].
int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : "";
  int ran = 0, failed = 0;
  for (const mlc::test::Case& c : mlc::test::registry()) {
    if (!std::strstr(c.name, filter)) continue;
    int before = mlc::test::failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      mlc::test::fail(c.name, 0, std::string("unexpected exception: ") + e.what());
    }
    bool ok = mlc::test::failures() == before;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", c.name);
    ++ran;
    failed += !ok;
  }
  std::printf("\n%d tests, %d failed\n", ran, failed);
  return failed ? 1 : 0;
}
