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
  PyKokkosbase
  GIT_REPOSITORY https://github.com/Yaraslaut/pykokkos-base.git
  GIT_TAG        d61203ecc48ec389970ac25f8528fcdab7fd4486
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


set(External_linked_libraries
  Kokkos::kokkos
  Eigen3::Eigen
)


if(!Kokkos_ENABLE_CUDA)
# Compiler options
add_compile_options(
  $<$<CONFIG:DEBUG>:-g3>
  $<$<CONFIG:DEBUG>:-Og>
  $<$<CONFIG:RELEASE>:-O3>
)
endif()



# ---- Declare library ----

add_library(
    prop_lib
    src/system.cpp
)

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
  Eigen3::Eigen
)

# ---- Declare executable ----

# ---- Declare libraries that use for linking ----
set(Linked_libraries
  prop_lib
)

# ---- Add tests ----


if(${Prop_CatchKokkos})
set(test_files   test/system_test.cpp )
endif()


set(UNIT_TEST prop_test)
add_executable(${UNIT_TEST}
  ${test_files}
  test/geometry_test.cpp
)
target_link_libraries(${UNIT_TEST} PRIVATE Catch2::Catch2WithMain  ${Linked_libraries} )
add_custom_command(
     TARGET ${UNIT_TEST}
     COMMENT "Run tests"
     POST_BUILD
     COMMAND ${UNIT_TEST}
)
target_compile_features(prop_test PRIVATE cxx_std_20)

# ---- CUSTOM COMMAND TO COPY PYTHON EXAMPLE ----
add_custom_target(copy_python_test_file
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/examples/simple.py ${CMAKE_BINARY_DIR}
)



message(STATUS "==============================================================================")
message(STATUS "    Prop (${PROJECT_VERSION})")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "Build type:                                             ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------------------------")
message(STATUS "==============================================================================")
