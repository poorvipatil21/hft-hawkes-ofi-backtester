cmake_minimum_required(VERSION 3.14)
project(hft_backtest_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra)

# Header-only INTERFACE library
add_library(backtest INTERFACE)
target_include_directories(backtest INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/strategies)

# Applications
add_executable(run_backtest     apps/run_backtest.cpp)
add_executable(benchmark        apps/benchmark.cpp)
add_executable(validate_fidelity apps/validate_fidelity.cpp)
add_executable(calibrate_hawkes  apps/calibrate_hawkes.cpp)
foreach(app run_backtest benchmark validate_fidelity calibrate_hawkes)
  target_link_libraries(${app} PRIVATE backtest)
endforeach()

# Tests
enable_testing()
foreach(t test_order_book test_matching test_friction test_hawkes)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE backtest)
  target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
