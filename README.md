# prop

FDTD solver of Maxwell's equations with the use of different backends including CUDA and OpenMP.

# Build

You can see all available presets via `cmake --list-presets` command.
For OpenMP support use `prop-debug` preset, if you want to enable cuda use `prop-cuda-debug`.
Example command to build prop with openmp support

``` sh
cmake --preset=prop-debug
cmake --build build --target pyprop
```

include target `prop_test` if you want to run unit tests.


To execute simple example

``` sh
cd build
cmake --build . --target copy_python_test_file
python ./simple.py

```
# This project is still under heavy development


## known issues
https://github.com/pybind/pybind11/issues/4606


https://github.com/kokkos/pykokkos-base/issues/55


https://github.com/NVIDIA/thrust/issues/1703


https://forums.developer.nvidia.com/t/strange-errors-after-system-gcc-upgraded-to-13-1-1/252441
