#pragma once

#include <concepts>


namespace Prop
{

struct Medium
{
    double _epsilon;
    double _mu;
    double _sigma;
    Medium(double eps, double mu, double sigm) : _epsilon(eps), _mu(mu),_sigma(sigm) {};
    virtual ~Medium() {};
};

struct IsotropicMedium: Medium
{
    IsotropicMedium(double eps, double mu, double sigm): Medium(eps,mu,sigm) {};
};

struct PML: Medium
{
    PML(): Medium(0,0,0) {};
};


} // namespace Prop
