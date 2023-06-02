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

#include <system.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>


TEST_CASE("Two dimensional system")
{
    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -0.1, 0.1, 10 };

    Prop::System2D system { ax, ay };
    system.propagateCustom(1e-4);

    while(Kokkos::is_initialized())
        Kokkos::finalize();
}

TEST_CASE("Create Box")
{
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -1.0, 1.0, 100 };
    Prop::Point2D c { 0.0, 0.0 };
    Prop::Box box(ax, ay, c);
    REQUIRE(true);
}

TEST_CASE("Create Free Space Entity")
{
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -1.0, 1.0, 100 };
    Prop::Point2D c { 0.0, 0.0 };
    Prop::Box box(ax, ay, c);

    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::FreeSpaceEntity<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial> freespace(box);
    REQUIRE(true);
}

TEST_CASE("Create Source Entity")
{
    Prop::Axis ax { -1.0, 1.0, 100 };
    Prop::Axis ay { -1.0, 1.0, 100 };
    Prop::Point2D c { 0.0, 0.0 };
    Prop::Box box(ax, ay, c);

    if (!Kokkos::is_initialized())
    {
        Kokkos::initialize();
    }

    Prop::SourceEntity<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial> source(box);
    REQUIRE(true);
}

TEST_CASE("Last call to finalize")
{
    Kokkos::finalize();
    REQUIRE(true);
}
