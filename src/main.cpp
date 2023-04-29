import hello;


#include "field.h"
#include "geometry.h"
#include <iostream>
#include <string>


auto main(int argc, char** argv) -> int
{
    Kokkos::ScopeGuard _kokkos(argc, argv);
    spdlog::set_level(spdlog::level::info); // Set global log level to info
    hello();
    return 0;
}
