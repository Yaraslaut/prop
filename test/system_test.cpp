/**
 * This file is part of the "prop" project
 *   Copyright (c) 2023-2023 Yaraslau Tamashevich <yaraslau.tamashevich@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "geometry.h"
#include "medium.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <iomanip>

#define TEST_QUANT(arg1, arg2) REQUIRE_THAT(arg1, Catch::Matchers::WithinAbs(arg2, 0.0000001))
#include "system.h"

using namespace std::chrono;

TEST_CASE("getEpsilon", "[!throws][!mayfail]")
{
    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::Axis ax { -1.0, 1.0 };
    Prop::Axis ay { -0.1, 0.1 };

    Prop::System2D system { ax, ay };

    REQUIRE(system.getEpsilon().size() > 0);
}

TEST_CASE("Two dimensional system with source only", "[!throws][!mayfail]")
{
    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::Axis ax { -1.0, 1.0 };
    Prop::Axis ay { -0.1, 0.1 };

    Prop::System2D system { ax, ay };

    Prop::Point2D c { 0.0, 0.0 };
    Prop::PointSource source { 1.0, 1.0, c };
    system.addSourceEz(source);

    system.propagateCustom(1e-4);

    auto Ez = system.getExternal(Prop::Components2DTM::Ez);
    REQUIRE(std::abs(Ez(static_cast<int>(system.getNx() / 2.0))) > 0);
}

TEST_CASE("Two dimensional system with source and medium")
{
    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::Axis ax { -1.0, 1.0 };
    Prop::Axis ay { -0.1, 0.1 };

    Prop::System2D system { ax, ay };

    Prop::Point2D c { 0.0, 0.0 };
    Prop::PointSource source { 1.0, 1.0, c };
    system.addSourceEz(source);
    auto block = Prop::IsotropicMedium(Prop::Axis(-0.5, -0.4), Prop::Axis(0.0, 0.05), 1.0, 1.0, 1.0);
    system.addBlock(block);
    system.propagateCustom(1e-4);

    auto Ez = system.getExternal(Prop::Components2DTM::Ez);
    REQUIRE(std::abs(Ez(static_cast<int>(system.getNx() / 2.0))) > 0);
}

TEST_CASE("Two dimensional system with source and pml")
{
    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::Axis ax { -1.0, 1.0 };
    Prop::Axis ay { -0.1, 0.1 };

    Prop::System2D system { ax, ay };

    Prop::Point2D c { 0.0, 0.0 };
    Prop::PointSource source { 1.0, 1.0, c };
    system.addSourceEz(source);
    auto block = Prop::PMLRegionX(Prop::Axis(-0.5, -0.4), Prop::Axis(0.0, 0.05));
    system.addBlock(block);
    system.propagateCustom(1e-4);

    auto Ez = system.getExternal(Prop::Components2DTM::Ez);
    REQUIRE(std::abs(Ez(static_cast<int>(system.getNx() / 2.0))) > 0);
}

TEST_CASE("Two dimensional system benchmark ")
{

    Prop::Axis ax { -1.0, 1.0 };
    Prop::Axis ay { -0.1, 0.1 };

    Prop::System2D system { ax, ay };

    Prop::Point2D c { 0.0, 0.0 };
    Prop::PointSource source { 1.0, 1.0, c };
    system.addSourceEz(source);

    int total_time { 0 };
    int max_iters { 100 };

    for (int i = 0; i < max_iters; i++)
    {
        auto start = high_resolution_clock::now();
        system.propagateCustom(100.0);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(stop - start);
        total_time += duration.count();
    }
#ifdef USE_SPDLOG
    spdlog::info("Execution time in nanoseconds: **  {:f}  ** \n ",
                 static_cast<double>(total_time / max_iters));
#endif
}

TEST_CASE("Two dimensional system with source and propagation", "[!throws][!mayfail]")
{
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -0.1, 0.1, 10 };

    Prop::System2D system { ax, ay };
    system.propagateCustom(1e-4);
}

TEST_CASE("Create Box")
{
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -1.0, 1.0, 100 };
    Prop::Point2D c { 0.0, 0.0 };
    Prop::Box box(ax, ay);
    REQUIRE(true);
}

TEST_CASE("Create Point Source")
{
    Prop::Point2D c { 0.0, 0.0 };
    Prop::PointSource source { 1.0, 1.0, c };
    REQUIRE(true);
}

TEST_CASE("Create Plane Wave Source")
{
    Prop::Point2D c { 0.0, 0.0 };
    Prop::Point2D d { 1.0, 0.0 };
    Prop::PlaneWave source { 1.0, 1.0, c, d };
    REQUIRE(true);
}
