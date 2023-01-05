#include "Kokkos_Core.hpp"

#include <catch2/catch_session.hpp>

int main(int argc, char* argv[])
{

    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    int result = Catch::Session().run(argc, argv);

    Kokkos::finalize();

    return result;
}
