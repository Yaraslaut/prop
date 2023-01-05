#include <iostream>
#include <string>

#include "field.h"
#include "geometry.h"

auto main(int argc, char** argv) -> int
{
    Kokkos::ScopeGuard _kokkos(argc,argv);
    spdlog::set_level(spdlog::level::info); // Set global log level to info

    return 0;
}
