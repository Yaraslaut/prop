cmake_minimum_required(VERSION 3.14)

project(
    prop
    VERSION 0.0.1
    DESCRIPTION "Maxwell solver"
    HOMEPAGE_URL "  "
    LANGUAGES CXX
)


# ---- Some global variables ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_UNITY_BUILD OFF)
set(Kokkos_ARCH_NATIVE ON)
set(BUILD_SHARED_LIBS ON)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

include(FetchContent)
include(CTest)

# ---- Add pybind11 ----
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.10.3
)
FetchContent_MakeAvailable(pybind11)

# ---- Add Eigen ---- # talk   with python
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen)

# ---- Add Kokkos ----
# Kokkos::kokkos
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.0.00
)
FetchContent_MakeAvailable(Kokkos)

# ---- Add spdlog ----
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(spdlog)

# ---- Add Catch2 ----
# Catch2::Catch2WithMain
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(Catch2)

# ---- Add Units  ----
# mp-units::mp-units
FetchContent_Declare(
  gsl-lite
  GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
)
FetchContent_MakeAvailable(gsl-lite)
set(gsl-lite_DIR "${CMAKE_BINARY_DIR}/_deps/gsl-lite-build")
FetchContent_Populate(
  units
  GIT_REPOSITORY    https://github.com/mpusz/units.git
#  GIT_TAG           v0.7.0
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/_deps/units-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/_deps/units-build"
  SUBBUILD_DIR      "${CMAKE_BINARY_DIR}/_deps/units-subbuild"
)
set(UNITS_USE_LIBFMT OFF)
add_subdirectory("${CMAKE_BINARY_DIR}/_deps/units-src/src")


# ------------------------------------------------------
#
if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
endif()

# ---- Declare library ----

add_library(
    prop_lib
    src/prop.cpp
    src/system.cpp
)

target_link_libraries(prop_lib PUBLIC Kokkos::kokkos spdlog::spdlog mp-units::mp-units Eigen3::Eigen)

include_directories(
    prop_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
)

# ---- Declare python wrapper ----

pybind11_add_module(pyprop
    src/prop_generate.cpp)

TARGET_LINK_LIBRARIES(pyprop PRIVATE prop_lib)

# ---- Declare executable ----

add_executable(prop_exe src/main.cpp)
add_executable(prop::exe ALIAS prop_exe)
set_property(TARGET prop_exe PROPERTY OUTPUT_NAME prop)

# ---- Declare libraries that use for linking ----

set(Linked_libraries prop_lib)

# ---- Link libraries ----

target_link_libraries(prop_exe PRIVATE ${Linked_libraries})

# ---- Add tests ----

set(UNIT_TEST prop_test)
add_executable(${UNIT_TEST} src/test.cpp)
target_link_libraries(${UNIT_TEST} PRIVATE Catch2::Catch2WithMain ${Linked_libraries})
add_custom_command(
     TARGET ${UNIT_TEST}
     COMMENT "Run tests"
     POST_BUILD
     COMMAND ${UNIT_TEST}
)

add_custom_target(copy_python_test_file
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/test.py ${CMAKE_BINARY_DIR}
)



message(STATUS "==============================================================================")
message(STATUS "    Prop (${PROJECT_VERSION})")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "Build type:                                             ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "==============================================================================")
