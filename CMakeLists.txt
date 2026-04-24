cmake_minimum_required(VERSION 3.16)
project(hogwild_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HOGWILD_ENABLE_OPENMP "Enable OpenMP parallelism" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-O3)
endif()

if(HOGWILD_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

add_library(hogwild_core
  src/types.cpp
  src/workloads.cpp
  src/schedules.cpp
  src/train_logistic.cpp
  src/train_mf.cpp
  src/io.cpp
  src/system_info.cpp
)

target_include_directories(hogwild_core PUBLIC include)

if(OpenMP_CXX_FOUND)
  target_link_libraries(hogwild_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(hogwild_core PUBLIC HOGWILD_HAS_OPENMP=1)
else()
  target_compile_definitions(hogwild_core PUBLIC HOGWILD_HAS_OPENMP=0)
endif()

add_executable(hogwild_bench src/main.cpp)
target_link_libraries(hogwild_bench PRIVATE hogwild_core)

add_executable(hogwild_tests tests/tests.cpp)
target_link_libraries(hogwild_tests PRIVATE hogwild_core)

enable_testing()
add_test(NAME hogwild_tests COMMAND hogwild_tests)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME csv_schema_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/check_csv_schema.py $<TARGET_FILE:hogwild_bench>
  )
endif()
