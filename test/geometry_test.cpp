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

#include <field.h>
#include <geometry.h>
#include <system.h>
#include <types.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

using micro = Prop::basic_length_unit;

#define TEST_QUANT(arg1, arg2) REQUIRE_THAT(arg1, Catch::Matchers::WithinAbs(arg2, 0.0000001))

TEST_CASE("Axis")
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

TEST_CASE("Point")
{
    Prop::Point3D t(11.0, 1.0, 2.0);
    TEST_QUANT(t._x, 11.0);
    TEST_QUANT(t._y, 1.0);
    TEST_QUANT(t._z, 2.0);
}

TEST_CASE("Dimensions 3D")
{
    Prop::DimensionsUnits3D<Prop::basic_length_unit> size(1.0, 2.0, 3.0);
    TEST_QUANT(size._xdim, 1.0);
    TEST_QUANT(size._ydim, 2.0);
    TEST_QUANT(size._zdim, 3.0);
}

TEST_CASE("Dimensions 2D")
{
    Prop::DimensionsUnits2D<Prop::basic_length_unit> size(1.0, 2.0);
    TEST_QUANT(size._xdim, 1.0);
    TEST_QUANT(size._ydim, 2.0);
}
