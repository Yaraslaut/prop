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
#include "types.h"

namespace Prop
{

class System2D;

struct PointSource
{

    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    double _freq;
    double _amplitude;
    Point2D _center;
    PointSource(double f, double am, Point2D center): _freq(f), _amplitude(am), _center(center) {};
    void Propagator(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto Ez = global_field._Ez.view<memory_space>();
        global_field._Ez.sync<memory_space>();
        auto x_index = global_geometry._x.getIndex(_center._x);
        auto y_index = global_geometry._y.getIndex(_center._y);
        Ez(x_index, y_index) += _amplitude * Kokkos::cos(_freq * time) * time_step;
        global_field._Ez.modify<memory_space>();
    };
};

struct PlaneWave
{
    // Plane _plane;
    double _freq;
    double _amplitude;
    Point2D _center;
    Point2D _direction;

    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    PlaneWave(double f, double am, Point2D cen, Point2D dir):
        _freq(f), _amplitude(am), _center(cen), _direction(dir) {};

    void Propagator(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto Ez = global_field._Ez.view<memory_space>();
        auto amplitude = _amplitude;
        auto freq = _freq;
        auto pos = global_geometry._x.getIndex(_center._x);
        Kokkos::parallel_for(
            global_field.getDevicePolicy(), KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                if (iinit == pos)
                    Ez(iinit, jinit) += amplitude * Kokkos::cos(freq * time) * time_step;
            });
    }
};

} // namespace Prop
