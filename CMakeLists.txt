cmake_minimum_required(VERSION 3.16)

# The version has one source: include/namturbo/version.h.
file(READ "${CMAKE_CURRENT_LIST_DIR}/include/namturbo/version.h" namturbo_version_text)
string(REGEX MATCH "kVersion = \{([0-9]+), ([0-9]+), ([0-9]+)\}" namturbo_version_match "${namturbo_version_text}")
if(NOT namturbo_version_match)
  message(FATAL_ERROR "cannot read the version from include/namturbo/version.h")
endif()
project(namturbo VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" LANGUAGES CXX)

# Used as the main project: the tests, tools and benchmarks are built and the default build type is Release. Used with
# add_subdirectory or FetchContent: only the library is built, and the settings of the parent project are left alone.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(NAMTURBO_MAIN_PROJECT ON)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()
else()
  set(NAMTURBO_MAIN_PROJECT OFF)
endif()

option(NAMTURBO_BUILD_TESTS "Build the tests, tools and benchmarks" ${NAMTURBO_MAIN_PROJECT})
option(NAMTURBO_INSTALL "Generate the install rules" ${NAMTURBO_MAIN_PROJECT})
option(NAMTURBO_BASELINE "Build the comparison with NAM Core: baseline rows in the benchmarks, the precision comparison, and the tools that make the reference outputs. Needs the NAM Core submodule and Python" OFF)
option(NAMTURBO_WERROR "Treat warnings as errors in every target of this project (not in NAM Core); the project must build without warnings with MSVC, clang-cl, GCC and Clang" OFF)
option(NAMTURBO_SKIP_ALLOCATOR_TEST "Do not build test_realtime, which replaces the global allocator (sanitizers replace it as well)" OFF)
option(NAMTURBO_PROFILE "Build with per-section cycle counters (bench prints them)" OFF)

# The model and the data the tests use. The model is the example A2 slimmable model of NAM Core (MIT licence, see NOTICE);
# any A2 slimmable .nam works: -DNAMTURBO_TEST_MODEL=<file> (the reference outputs in the data folder belong to the
# bundled model and must be regenerated for another one, see tests/data/README.md).
set(NAMTURBO_TEST_MODEL "${CMAKE_CURRENT_LIST_DIR}/tests/data/a2-example.nam" CACHE FILEPATH "A2 slimmable model for the tests")
set(NAMTURBO_TEST_DATA "${CMAKE_CURRENT_LIST_DIR}/tests/data" CACHE PATH "Folder with the test inputs and the reference outputs")

# ---------------------------------------------------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------------------------------------------------
add_library(namturbo STATIC src/a2_weights.cpp src/a2_ref.cpp src/a2_avx2.cpp src/a2_lite_avx2.cpp src/cpu_features.cpp src/a2_dispatch.cpp
  src/model.cpp src/processor.cpp src/cpu.cpp src/resampler.cpp)
add_library(namturbo::namturbo ALIAS namturbo)
target_compile_features(namturbo PUBLIC cxx_std_20)
target_include_directories(namturbo PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>" "$<INSTALL_INTERFACE:include>"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
if(NAMTURBO_PROFILE)
  target_compile_definitions(namturbo PUBLIC NAMTURBO_PROFILE)
endif()
# JSON parser for the model files: nlohmann/json 3.12.0, single header, MIT licence (see NOTICE).
target_include_directories(namturbo SYSTEM PRIVATE "${CMAKE_CURRENT_LIST_DIR}/third_party/nlohmann")
if(MSVC)
  target_compile_options(namturbo PRIVATE -W4)
else()
  target_compile_options(namturbo PRIVATE -Wall -Wextra)
endif()

# The AVX2 files (the engines and the resampler) need AVX2/FMA code generation. MSVC accepts the intrinsics without flags; GCC and
# clang(-cl) need them. cpu_features.cpp is deliberately NOT given these flags: it must be safe to run on
# a CPU without AVX2 (callers check it before ever reaching AVX2 code).
if(NOT MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set_source_files_properties(src/a2_avx2.cpp src/a2_lite_avx2.cpp src/resampler.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Install: the library, the public headers, and a CMake package, so that a program can use
#   find_package(namturbo CONFIG REQUIRED)  and  target_link_libraries(app PRIVATE namturbo::namturbo)
# ---------------------------------------------------------------------------------------------------------------------
if(NAMTURBO_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(TARGETS namturbo EXPORT namturboTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
  install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  install(EXPORT namturboTargets NAMESPACE namturbo:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/namturbo)
  configure_package_config_file(cmake/namturboConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/namturboConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/namturbo)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/namturboConfigVersion.cmake" VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/namturboConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/namturboConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/namturbo)
endif()

if(NAMTURBO_BUILD_TESTS)
enable_testing()
# The tests, tools and benchmarks use internal headers (src/) and the JSON header. The public interface of the library is include/ only.
include_directories("${CMAKE_CURRENT_LIST_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}/tools" "${CMAKE_CURRENT_LIST_DIR}/benchmarks")
include_directories(SYSTEM "${CMAKE_CURRENT_LIST_DIR}/third_party/nlohmann")

# ---------------------------------------------------------------------------------------------------------------------
# Tests: need no NAM Core and no Python; the inputs and reference outputs are files in tests/data
# ---------------------------------------------------------------------------------------------------------------------
# Exit code 77 means "test material missing": reported as skipped, not passed.
function(namturbo_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE namturbo)
  add_test(NAME ${name} COMMAND ${name} "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}")
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
endfunction()

namturbo_test(test_weights)
namturbo_test(test_dispatch)
namturbo_test(test_api)
add_executable(test_adaptive tests/test_adaptive.cpp)
target_link_libraries(test_adaptive PRIVATE namturbo)
add_test(NAME test_adaptive COMMAND test_adaptive "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}")
set_tests_properties(test_adaptive PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
add_executable(test_swap tests/test_swap.cpp)
target_link_libraries(test_swap PRIVATE namturbo)
add_test(NAME test_swap COMMAND test_swap "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}")
set_tests_properties(test_swap PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
add_executable(test_monitor tests/test_monitor.cpp)
target_link_libraries(test_monitor PRIVATE namturbo)
add_test(NAME test_monitor COMMAND test_monitor)
set_tests_properties(test_monitor PROPERTIES TIMEOUT 120)
add_executable(test_resampler tests/test_resampler.cpp)
target_link_libraries(test_resampler PRIVATE namturbo)
add_test(NAME test_resampler COMMAND test_resampler)
set_tests_properties(test_resampler PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
namturbo_test(test_oversampling)
namturbo_test(test_ref_parity)
namturbo_test(test_avx2_parity)
namturbo_test(test_avx2_robust)
if(NOT NAMTURBO_SKIP_ALLOCATOR_TEST)
  namturbo_test(test_realtime)
endif()
namturbo_test(test_cpu_failure)
namturbo_test(test_fuzz_loader)
# The same suites against the A2-Lite engine (slim 0.0).
add_test(NAME test_avx2_parity_lite COMMAND test_avx2_parity "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}" 0.0)
add_test(NAME test_avx2_robust_lite COMMAND test_avx2_robust "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}" 0.0)
set_tests_properties(test_avx2_parity_lite test_avx2_robust_lite PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)
set_tests_properties(test_avx2_robust test_ref_parity PROPERTIES TIMEOUT 1200)

# Every public header must compile on its own with nothing but include/ on the include path (no internal header,
# no JSON library, no intrinsics).
set(NAMTURBO_PUBLIC_HEADERS cpu error model processor version namturbo)
set(NAMTURBO_HEADER_CHECK_SOURCES)
foreach(h ${NAMTURBO_PUBLIC_HEADERS})
  set(f "${CMAKE_BINARY_DIR}/header_check/check_${h}.cpp")
  file(WRITE "${f}" "#include \"namturbo/${h}.h\"\n")
  list(APPEND NAMTURBO_HEADER_CHECK_SOURCES "${f}")
endforeach()
add_library(namturbo_header_check OBJECT ${NAMTURBO_HEADER_CHECK_SOURCES})
set_property(TARGET namturbo_header_check PROPERTY INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
if(MSVC)
  target_compile_options(namturbo_header_check PRIVATE -W4)
else()
  target_compile_options(namturbo_header_check PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Tools and benchmarks
# ---------------------------------------------------------------------------------------------------------------------
add_executable(parity tools/parity.cpp)
target_link_libraries(parity PRIVATE namturbo)
add_executable(process_file tools/process_file.cpp)
target_link_libraries(process_file PRIVATE namturbo)

add_executable(bench benchmarks/bench.cpp)
target_link_libraries(bench PRIVATE namturbo)
add_executable(variant_compare benchmarks/variant_compare.cpp)
target_link_libraries(variant_compare PRIVATE namturbo)
add_executable(nam_bench benchmarks/nam_bench.cpp)
target_link_libraries(nam_bench PRIVATE namturbo)
add_executable(two_instance benchmarks/two_instance.cpp)
target_link_libraries(two_instance PRIVATE namturbo)

if(NAMTURBO_BASELINE)
  set(NAMCORE "${CMAKE_CURRENT_LIST_DIR}/external/NeuralAmpModelerCore" CACHE PATH "NAM Core checkout")
  if(NOT EXISTS "${NAMCORE}/NAM/dsp.h")
    message(FATAL_ERROR "NAM Core not found at ${NAMCORE}. Run: git submodule update --init --recursive")
  endif()
  find_program(NAMTURBO_PYTHON NAMES python python3)
  if(NOT NAMTURBO_PYTHON)
    message(FATAL_ERROR "NAMTURBO_BASELINE needs Python 3 with NumPy")
  endif()

  # NAM Core as a static library: the baseline and the reference outputs. Built like its own tools: A2 fast path on.
  file(GLOB NAM_SOURCES "${NAMCORE}/NAM/*.cpp" "${NAMCORE}/NAM/*/*.cpp")
  add_library(namcore STATIC ${NAM_SOURCES})
  target_include_directories(namcore SYSTEM PUBLIC "${NAMCORE}" "${NAMCORE}/Dependencies/eigen"
    "${NAMCORE}/Dependencies/nlohmann" "${NAMCORE}/Dependencies/AudioDSPTools/dsp")
  target_compile_definitions(namcore PUBLIC NAM_ENABLE_A2_FAST)
  # NAM Core uses Eigen, which uses AVX2 and FMA only when the compiler is told to generate them. A plug-in author who wants speed does
  # that, so the baseline is built that way by default; switch it off to compare with NAM Core as its own tools build it.
  option(NAMTURBO_BASELINE_AVX2 "Build NAM Core (the baseline) with AVX2 and FMA code generation" ON)
  if(NAMTURBO_BASELINE_AVX2)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(namcore PRIVATE /arch:AVX2)
    else()
      target_compile_options(namcore PRIVATE -mavx2 -mfma)
    endif()
  endif()
  if(MSVC)
    target_compile_definitions(namcore PUBLIC _USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(namcore PRIVATE -W0)  # third-party code: not ours to fix
  endif()

  # The benchmarks get the baseline rows.
  foreach(t bench nam_bench two_instance)
    target_link_libraries(${t} PRIVATE namcore)
    target_compile_definitions(${t} PRIVATE NAMTURBO_WITH_NAMCORE)
  endforeach()

  add_executable(golden benchmarks/golden.cpp)
  target_link_libraries(golden PRIVATE namcore)
  add_executable(accuracy benchmarks/accuracy.cpp)
  target_link_libraries(accuracy PRIVATE namturbo namcore)
  add_executable(precision_probe benchmarks/precision_probe.cpp)
  target_link_libraries(precision_probe PRIVATE namturbo namcore)

  # Makes tests/data/synth.raw, sweep10s.raw and the golden_*.raw reference outputs (NAM Core fast path, block 64) from the
  # test model. Run it only to change the test data; the result is committed (tests/data/README.md).
  add_custom_target(regenerate_test_data
    COMMAND ${NAMTURBO_PYTHON} "${CMAKE_CURRENT_LIST_DIR}/benchmarks/make_golden.py" $<TARGET_FILE:golden> "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}"
    DEPENDS golden VERBATIM)

  # NAM Core's fast and generic paths must agree, or nothing downstream can be trusted.
  add_test(NAME golden_selfcheck
    COMMAND ${NAMTURBO_PYTHON} "${CMAKE_CURRENT_LIST_DIR}/tests/golden_selfcheck.py" $<TARGET_FILE:golden> "${NAMTURBO_TEST_MODEL}" "${NAMTURBO_TEST_DATA}")
  # Our engine must be as precise as NAM Core's own on average over DC levels (the ill-conditioned inputs).
  add_test(NAME precision_vs_nam_full COMMAND precision_probe "${NAMTURBO_TEST_MODEL}" 1.0 --check)
  add_test(NAME precision_vs_nam_lite COMMAND precision_probe "${NAMTURBO_TEST_MODEL}" 0.0 --check)
  set_tests_properties(golden_selfcheck precision_vs_nam_full precision_vs_nam_lite PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
  set_tests_properties(precision_vs_nam_full precision_vs_nam_lite PROPERTIES PASS_REGULAR_EXPRESSION "PASSED")
  add_test(NAME bench_smoke COMMAND bench --model "${NAMTURBO_TEST_MODEL}" --slim 1.0 --backend scalar --seconds 0.2 --iters 2 --nam)
  set_tests_properties(bench_smoke PROPERTIES SKIP_RETURN_CODE 77 PASS_REGULAR_EXPRESSION "ours \\(Processor\\).*RTF.*\n.*NAM fast.*RTF.*\n.*NAM generic.*RTF")
else()
  add_test(NAME bench_smoke COMMAND bench --model "${NAMTURBO_TEST_MODEL}" --slim 1.0 --backend scalar --seconds 0.2 --iters 2)
  set_tests_properties(bench_smoke PROPERTIES SKIP_RETURN_CODE 77 PASS_REGULAR_EXPRESSION "ours \\(Processor\\).*RTF")
endif()

add_test(NAME nam_bench_smoke COMMAND nam_bench --model "${NAMTURBO_TEST_MODEL}" --buffers 32,64 --runs 1 --seconds 0.3)
set_tests_properties(nam_bench_smoke PROPERTIES SKIP_RETURN_CODE 77 PASS_REGULAR_EXPRESSION "[|] (full|lite) [|] 64 [|]" FAIL_REGULAR_EXPRESSION "TOO LARGE")

# The comparison of A2-Lite with A2-Full runs on the model of this repository and a generated signal.
add_test(NAME variant_compare_smoke COMMAND variant_compare --model "${NAMTURBO_TEST_DATA}/test-amp-and-cab.nam" --input "${NAMTURBO_TEST_DATA}/synth.raw")
set_tests_properties(variant_compare_smoke PROPERTIES SKIP_RETURN_CODE 77 PASS_REGULAR_EXPRESSION "[|] test-amp-and-cab.nam [|] synth.raw [|]")

# The benchmark program runs on the model that belongs to this repository (the second model of the benchmark tables).
add_test(NAME nam_bench_repo_model COMMAND nam_bench --model "${NAMTURBO_TEST_DATA}/test-amp-and-cab.nam" --buffers 64 --runs 1 --seconds 0.3)
set_tests_properties(nam_bench_repo_model PROPERTIES SKIP_RETURN_CODE 77 PASS_REGULAR_EXPRESSION "[|] test-amp-and-cab.nam [|] (full|lite) [|] 64 [|]" FAIL_REGULAR_EXPRESSION "TOO LARGE")

# The code fragments of the README and the integration guide must compile against the real interface (compiled only, never run).
add_library(docs_snippets OBJECT tests/docs_snippets.cpp)
target_link_libraries(docs_snippets PRIVATE namturbo)

# The example that shows which thread does what (examples/realtime_host): it runs for 2 seconds as a test.
add_executable(example_realtime_host examples/realtime_host/main.cpp)
target_link_libraries(example_realtime_host PRIVATE namturbo)
add_test(NAME example_realtime_host COMMAND example_realtime_host "${NAMTURBO_TEST_MODEL}" 2)
set_tests_properties(example_realtime_host PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)

# The installed library works for a separate project: install, build examples/consumer against the install tree, run it.
add_test(NAME install_consumer COMMAND ${CMAKE_COMMAND} "-DSOURCE_DIR=${CMAKE_CURRENT_LIST_DIR}" "-DBINARY_DIR=${CMAKE_BINARY_DIR}"
  "-DCONFIG=$<CONFIG>" "-DMODEL=${NAMTURBO_TEST_MODEL}" "-DGENERATOR=${CMAKE_GENERATOR}" "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
  "-DTOOLSET=${CMAKE_GENERATOR_TOOLSET}" -P "${CMAKE_CURRENT_LIST_DIR}/tests/install_check.cmake")
set_tests_properties(install_consumer PROPERTIES TIMEOUT 300)

# Short smoke run (the 60 s soak is run by hand): both pairings, free-running. They check that no block misses its real-time
# deadline, so they run alone (RUN_SERIAL): other tests running at the same time would take the processor away from them.
add_test(NAME two_instance_smoke COMMAND two_instance --model "${NAMTURBO_TEST_MODEL}" --seconds 3 --free-run --data "${NAMTURBO_TEST_DATA}")
add_test(NAME two_instance_smoke_mixed COMMAND two_instance --model "${NAMTURBO_TEST_MODEL}" --a full --b lite --seconds 3 --free-run --data "${NAMTURBO_TEST_DATA}")
set_tests_properties(two_instance_smoke two_instance_smoke_mixed PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120 RUN_SERIAL TRUE)

endif() # NAMTURBO_BUILD_TESTS

# ---------------------------------------------------------------------------------------------------------------------
# Warnings: every target of this project (NAM Core is not ours) builds with the strict warning level; NAMTURBO_WERROR makes them errors.
# ---------------------------------------------------------------------------------------------------------------------
get_property(namturbo_all_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(t IN LISTS namturbo_all_targets)
  if(t STREQUAL "namcore")
    continue()
  endif()
  get_target_property(kind ${t} TYPE)
  if(kind STREQUAL "UTILITY")
    continue()
  endif()
  if(MSVC)
    target_compile_options(${t} PRIVATE /W4 $<$<BOOL:${NAMTURBO_WERROR}>:/WX>)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra $<$<BOOL:${NAMTURBO_WERROR}>:-Werror>)
  endif()
endforeach()
