cmake_minimum_required(VERSION 3.14)

project(
    prop
    VERSION 0.0.1
    DESCRIPTION "Maxwell solver"
    HOMEPAGE_URL "  "
    LANGUAGES CXX
)

## -------------------------

set(CMAKE_UNITY_BUILD OFF)
set(Kokkos_ARCH_NATIVE ON)
set(BUILD_SHARED_LIBS ON)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

include(FetchContent)
include(CTest)

# ---- Add Eigen ---- #
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen)

# ---- Add SpdLog ---- #
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG         v1.11.0
)
FetchContent_MakeAvailable(spdlog)

# ---- Add Kokkos ----
# Kokkos::kokkos
FetchContent_Declare(
  PyKokkosbase
  GIT_REPOSITORY https://github.com/Yaraslaut/pykokkos-base.git
  GIT_TAG        ec5944a136d607a4b6fec36bb90dfcd11b36f877
)
FetchContent_MakeAvailable(PyKokkosbase)
find_package(Python3 COMPONENTS Development)

# ---- Add Catch2 ----
# Catch2::Catch2WithMain
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2
)
FetchContent_MakeAvailable(Catch2)

# ---- Add SpdLog ---- #
FetchContent_Declare(
  boxed-cpp
  GIT_REPOSITORY https://github.com/contour-terminal/boxed-cpp.git
  GIT_TAG        6100165484b3d574b011b68df12e7b966f2759ae
)
FetchContent_MakeAvailable(boxed-cpp)

set(External_linked_libraries
  Kokkos::kokkos
  Eigen3::Eigen
  boxed-cpp
)

# ---- Declare library ----

add_library(
  prop_lib
  src/system.cpp
  src/prop.cpp
)


# ---- Handle some ycuda issues ----
if(NOT Kokkos_ENABLE_CUDA)
  # Compiler options
  set(External_linked_libraries
    ${External_linked_libraries}
    spdlog::spdlog_header_only
  )
  add_compile_options(
    $<$<CONFIG:DEBUG>:-g3>
    $<$<CONFIG:DEBUG>:-Og>
    $<$<CONFIG:RELEASE>:-O3>
  )
  target_compile_definitions(prop_lib PUBLIC
    USE_SPDLOG)
endif()



target_link_libraries(prop_lib PUBLIC
  ${External_linked_libraries}
)

include_directories(
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
)

# ---- Declare python wrapper ----

pybind11_add_module(pyprop
  src/prop_generate.cpp)

TARGET_LINK_LIBRARIES(pyprop PUBLIC
  prop_lib
)

# ---- Declare executable ----

# ---- Declare libraries that use for linking ----
# ---- Add tests ----


set(test_files
  test/system_test.cpp
  test/geometry_test.cpp
  test/main_test.cpp
)

set(UNIT_TEST prop_test)
add_executable(${UNIT_TEST}
  ${test_files}
)
target_link_libraries(${UNIT_TEST}
  Catch2::Catch2
  prop_lib)
add_test(${UNIT_TEST} ./${UNIT_TEST})

add_custom_command(
  TARGET ${UNIT_TEST}
  COMMENT "Run tests"
  POST_BUILD
  COMMAND ${UNIT_TEST}
)

#target_compile_features(prop_test PRIVATE cxx_std_20) TODO

# ---- CUSTOM COMMAND TO COPY PYTHON EXAMPLE ----
add_custom_target(copy_python_example_file
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/simple.py ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/plane_wave_with_pml_x.py ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/plane_wave_with_pml_y.py ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/point_source.py ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/logo/logo.py ${CMAKE_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/logo/processing.jl ${CMAKE_BINARY_DIR}
)



message(STATUS "==============================================================================")
message(STATUS "    Prop (${PROJECT_VERSION})")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "Build type:                                             ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "==============================================================================")
