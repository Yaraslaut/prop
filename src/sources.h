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

#pragma once

#include "field.h"
#include "geometry.h"
#include "prop.h"

namespace Prop
{

struct Source
{
};

struct PlaneWave2D: Source
{
    // Plane _plane;
    double _freq;
    double _amplitude;
    Point2D _center;
    Point2D _direction;

    PlaneWave2D(double f, double am, Point2D cen, Point2D dir):
        _freq(f), _amplitude(am), _center(cen), _direction(dir) {};

    double getField(double time, Point2D point)
    {

        auto res = _amplitude * std::cos(time * _freq);

        auto vecToCenter = point - _center;
        double scalarProd = dot(vecToCenter, _direction);
        if (abs(scalarProd) < 0.1) // TODO ARBITRARY NUMBER
            return res;
        return 0.0;
    }
};

} // namespace Prop
