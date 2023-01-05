#pragma once

#include "geometry.h"
#include "types.h"
#include <cmath>

namespace Prop
{
template <typename T>
double delta_function(T f)
{
    auto v = static_cast<double>(f);
    if (std::abs(v) < 0.02)
        return 1.;
    return 0.;
}
} // namespace Prop
