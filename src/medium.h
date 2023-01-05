#pragma once

#include <concepts>

namespace Prop
{

struct BaseMedium
{
};

struct IsotropicMedium: BaseMedium
{
    IsotropicMedium(double eps, double mu, double sigm): _epsilon(eps), _mu(mu), _sigma(sigm) {};
    double _epsilon;
    double _mu;
    double _sigma;
};

} // namespace Prop
