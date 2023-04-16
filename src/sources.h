#pragma once

#include "field.h"
#include "geometry.h"
#include "prop.h"

#include <numbers>
#include <spdlog/spdlog.h>

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
    Point2D _direction; // TODO make it work

    PlaneWave2D(auto f, auto am, Point2D cen, Point2D dir):
        _freq(f), _amplitude(am), _center(cen), _direction(dir) {};

    double getField(double time, Point2D point)
    {

        auto res = _amplitude * std::cos(time * _freq);

        auto distToCenter = point._x - _center._x;
        res *= std::exp(-std::pow(distToCenter.number(), 2.0) / 0.5); // TODO fix usage of units

        return res;
    }
};

} // namespace Prop
