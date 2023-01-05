#pragma once

#include <numbers>

#include "geometry.h"
#include "field.h"
#include "prop.h"
#include <spdlog/spdlog.h>

namespace Prop
{

class Source{};

class PlaneWave: Source
{
    //Plane _plane;
    double _freq;  // TODO
};

// class GaussianSource: Source
// {
//   public:
//     Point _mu;
//     double _freq;
//     double _sigma;
//     GaussianSource(Point m, double f, double si): _mu(m), _freq(f), _sigma(si) {};
//     double get_field(Point x, double time)
//     {
//         //auto dist = x - _mu; TODO
//         auto exp_factor = 1.0; // TODO -std::pow(dist.number() / _sigma, 2);
//         auto norm = 1 / (_sigma * std::sqrt(2 * std::numbers::pi));
//         return norm * cos(_freq * time) * exp(0.5 * exp_factor) * norm;
//     }
// };

} // namespace Prop
