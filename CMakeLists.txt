cmake_minimum_required(VERSION 3.20)
project(mlc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The test configuration: address and undefined-behaviour sanitizers on
# everything. Slow, so it is a separate build directory (see README).
option(MLC_SANITIZE "Build with -fsanitize=address,undefined" OFF)

add_compile_options(-Wall -Wextra -Wpedantic)
if(MLC_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Everything except main() lives in a library so the tests can link it.
add_library(mlc_core STATIC
  src/support/json.cpp
  src/support/files.cpp
  src/graph/ir.cpp
  src/graph/printer.cpp
  src/graph/verifier.cpp
  src/graph/interpreter.cpp
  src/frontend/importer.cpp
)
target_include_directories(mlc_core PUBLIC src)

add_executable(mlc src/main.cpp)
target_link_libraries(mlc PRIVATE mlc_core)

enable_testing()

add_executable(mlc_tests
  tests/test_main.cpp
  tests/test_json.cpp
  tests/test_ir.cpp
  tests/test_interpreter.cpp
  tests/test_importer.cpp
)
target_link_libraries(mlc_tests PRIVATE mlc_core)
target_include_directories(mlc_tests PRIVATE tests)
add_test(NAME unit COMMAND mlc_tests)

# End-to-end: interpret the real exported GPT-2 and compare with PyTorch's
# golden logits. Needs `python export/export_gpt2.py` to have been run. The
# two long sequence lengths are labelled "slow" (ctest -LE slow skips them).
set(ARTIFACTS ${CMAKE_SOURCE_DIR}/artifacts)
foreach(T 16 64 256 1024)
  add_test(NAME golden_T${T}
    COMMAND mlc run ${ARTIFACTS}/graphs/gpt2_T${T}.json
            --input ${ARTIFACTS}/golden/T${T}_ids.npy
            --expect ${ARTIFACTS}/golden/T${T}_logits.npy)
  set_tests_properties(golden_T${T} PROPERTIES REQUIRED_FILES ${ARTIFACTS}/graphs/gpt2_T${T}.json)
endforeach()
set_tests_properties(golden_T256 golden_T1024 PROPERTIES LABELS slow TIMEOUT 1800)
