#include "field.h"
#include "geometry.h"
#include "system.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

using micro = Prop::basic_length_unit;

#define TEST_QUANT(arg1, arg2) REQUIRE_THAT(arg1.number(), Catch::Matchers::WithinAbs(arg2, 0.0000001))

TEST_CASE("Axis", "[geometry]")
{
    Prop::Axis t(-1.0, 1.0, 100);
    TEST_QUANT(t._min, -1.0);

    TEST_QUANT(t._max, 1.0);
    REQUIRE(t._N == 100);

    Prop::Axis t2(-1.0, 1.0);
    TEST_QUANT(t2._min, -1.0);
    TEST_QUANT(t2._max, 1.0);
    REQUIRE(t2._N == 1);

    Prop::Axis t3 {};
    TEST_QUANT(t3._min, 0.0);
    TEST_QUANT(t3._max, 0.0);
    REQUIRE(t3._N == 1);

    Prop::Axis t4 { -1.0, 1.0, 10 };
    TEST_QUANT(t4._min, -1.0);
    TEST_QUANT(t4._max, 1.0);
    REQUIRE(t4._N == 10);
}

TEST_CASE("Point", "[geometry]")
{
    Prop::Point3D t(11.0, 1.0, 2.0);
    TEST_QUANT(t._x, 11.0);
    TEST_QUANT(t._y, 1.0);
    TEST_QUANT(t._z, 2.0);
}

TEST_CASE("Dimensions 3D", "[geometry]")
{
    Prop::DimensionsUnits3D<Prop::basic_length_unit> size(1.0, 2.0, 3.0);
    TEST_QUANT(size._xdim, 1.0);
    TEST_QUANT(size._ydim, 2.0);
    TEST_QUANT(size._zdim, 3.0);
}

TEST_CASE("Dimensions 2D", "[geometry]")
{
    Prop::DimensionsUnits2D<Prop::basic_length_unit> size(1.0, 2.0);
    TEST_QUANT(size._xdim, 1.0);
    TEST_QUANT(size._ydim, 2.0);
}

TEST_CASE("Two dimensional system", "[TwoDimensionalSystem], [kokkos]")
{
    if (!Kokkos::is_initialized())
    {
        spdlog::debug("Initializing Kokkos...");
        Kokkos::initialize();
    }
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -0.1, 0.1, 10 };

    Prop::System2D system { ax, ay };
    auto& ext = system.getExternal(Prop::Components2DTM::Ez);
    std::ranges::for_each(std::views::iota(0, 100), [&](int i) {
        std::ranges::for_each(std::views::iota(0, 10), [&](int j) { ext(i, j) = i * 1.0; });
    });

    system.setExternal(Prop::Components2DTM::Ez);

    system.propagateCustom(1e-4);
}
