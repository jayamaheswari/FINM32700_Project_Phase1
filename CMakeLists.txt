cmake_minimum_required(VERSION 3.10)
project(HighPerformanceLA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(linalg STATIC
        src/linalg.cpp
)
target_include_directories(linalg PUBLIC include)

add_executable(test_linalg tests/test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE linalg)

add_executable(phase2_1_benchmark_linalg_test benchmarks/phase2_1_benchmark_linalg_test.cpp)
target_link_libraries(phase2_1_benchmark_linalg_test PRIVATE linalg)

add_executable(phase2_2_cache_test benchmarks/phase2_2_cache_test.cpp)
target_link_libraries(phase2_2_cache_test PRIVATE linalg)

add_executable(phase2_3_alignment_test benchmarks/phase2_3_alignment_test.cpp)
target_link_libraries(phase2_3_alignment_test PRIVATE linalg)

add_executable(phase2_3_forced_intrinsic_test benchmarks/phase2_3_forced_intrinsic_test.cpp)
target_link_libraries(phase2_3_forced_intrinsic_test PRIVATE linalg)

add_executable(phase2_4_inline_test benchmarks/phase2_4_inline_test.cpp)
target_link_libraries(phase2_4_inline_test PRIVATE linalg)

add_executable(phase2_6_optimization_test benchmarks/phase2_6_optimization_test.cpp)
target_link_libraries(phase2_6_optimization_test PRIVATE linalg)
