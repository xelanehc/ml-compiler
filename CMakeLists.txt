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
  src/graph/passes.cpp
  src/frontend/importer.cpp
  src/loop/ir.cpp
  src/loop/lower.cpp
  src/loop/schedule.cpp
  src/loop/interpreter.cpp
  src/codegen/emit_cpp.cpp
  src/codegen/build.cpp
)
target_include_directories(mlc_core PUBLIC src)
# Generated code is compiled together with the runtime; mlc needs to know where it is.
target_compile_definitions(mlc_core PRIVATE MLC_RUNTIME_DIR="${CMAKE_SOURCE_DIR}/runtime")
target_link_libraries(mlc_core PUBLIC ${CMAKE_DL_LIBS})

add_executable(mlc src/main.cpp)
target_link_libraries(mlc PRIVATE mlc_core)

enable_testing()

add_executable(mlc_tests
  tests/test_main.cpp
  tests/test_json.cpp
  tests/test_ir.cpp
  tests/test_interpreter.cpp
  tests/test_importer.cpp
  tests/test_loop.cpp
  tests/test_codegen.cpp
  tests/test_passes.cpp
  tests/test_schedule.cpp
)
target_link_libraries(mlc_tests PRIVATE mlc_core)
target_include_directories(mlc_tests PRIVATE tests)
add_test(NAME unit COMMAND mlc_tests)

# End-to-end, against PyTorch's golden logits. Needs `python export/export_gpt2.py`
# to have been run. For each sequence length:
#   golden_T<T>    the graph interpreter on the imported graph
#   optimized_T<T> the graph interpreter on the graph after the optimization passes
#   compile_T<T>   graph -> loop IR -> schedule -> C++ -> shared library
#   compiled_T<T>  the shared library, run on the golden input with the default number of threads
# The two long sequence lengths are labelled "slow" (ctest -LE slow skips them).
set(ARTIFACTS ${CMAKE_SOURCE_DIR}/artifacts)
set(GEN ${CMAKE_BINARY_DIR}/gen)
# In the sanitizer configuration the generated code is sanitized too.
if(MLC_SANITIZE)
  set(GEN_FLAGS "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer")
else()
  set(GEN_FLAGS "-O3 -mcpu=native")
endif()
foreach(T 16 64 256 1024)
  set(GRAPH ${ARTIFACTS}/graphs/gpt2_T${T}.json)
  set(GOLDEN --input ${ARTIFACTS}/golden/T${T}_ids.npy --expect ${ARTIFACTS}/golden/T${T}_logits.npy)
  add_test(NAME golden_T${T} COMMAND mlc run ${GRAPH} ${GOLDEN})
  add_test(NAME optimized_T${T} COMMAND mlc run ${GRAPH} --opt ${GOLDEN})
  add_test(NAME compile_T${T} COMMAND mlc compile ${GRAPH} -o ${GEN}/gpt2_T${T} --flags ${GEN_FLAGS})
  add_test(NAME compiled_T${T}
    COMMAND mlc exec ${GEN}/gpt2_T${T}/model.so --weights ${ARTIFACTS}/graphs/gpt2_weights.bin ${GOLDEN})
  set_tests_properties(golden_T${T} optimized_T${T} compile_T${T} compiled_T${T} PROPERTIES REQUIRED_FILES ${GRAPH})
  # A fixture makes ctest build the library before running it, and skip the run if the build failed.
  set_tests_properties(compile_T${T} PROPERTIES FIXTURES_SETUP lib_T${T})
  set_tests_properties(compiled_T${T} PROPERTIES FIXTURES_REQUIRED lib_T${T})
endforeach()
set_tests_properties(golden_T256 golden_T1024 optimized_T256 optimized_T1024 compile_T256 compile_T1024 compiled_T256 compiled_T1024
  PROPERTIES LABELS slow TIMEOUT 1800)

# The same library on one thread: the serial path through every parallel loop.
add_test(NAME compiled_T16_one_thread
  COMMAND mlc exec ${GEN}/gpt2_T16/model.so --weights ${ARTIFACTS}/graphs/gpt2_weights.bin --threads 1
          --input ${ARTIFACTS}/golden/T16_ids.npy --expect ${ARTIFACTS}/golden/T16_logits.npy)
set_tests_properties(compiled_T16_one_thread PROPERTIES FIXTURES_REQUIRED lib_T16)

# The unoptimized path (every pass off, no memory reuse) stays tested, because
# it is the baseline every ablation is measured against.
add_test(NAME compile_unoptimized_T16
  COMMAND mlc compile ${ARTIFACTS}/graphs/gpt2_T16.json -o ${GEN}/gpt2_T16_unoptimized --disable all --flags ${GEN_FLAGS})
add_test(NAME compiled_unoptimized_T16
  COMMAND mlc exec ${GEN}/gpt2_T16_unoptimized/model.so --weights ${ARTIFACTS}/graphs/gpt2_weights.bin
          --input ${ARTIFACTS}/golden/T16_ids.npy --expect ${ARTIFACTS}/golden/T16_logits.npy)
set_tests_properties(compile_unoptimized_T16 PROPERTIES FIXTURES_SETUP lib_unoptimized REQUIRED_FILES ${ARTIFACTS}/graphs/gpt2_T16.json)
set_tests_properties(compiled_unoptimized_T16 PROPERTIES FIXTURES_REQUIRED lib_unoptimized)
