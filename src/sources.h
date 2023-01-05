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

#include <algorithm>

namespace Prop
{

class System2D;

struct PointSource
{

    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using memory_space = Types<ExecutionSpace>::memory_space;

    double _freq;
    double _amplitude;
    Point2D _center;
    PointSource(double f, double am, Point2D center): _freq(f), _amplitude(am), _center(center) {};
    void Propagator(double time,
                    double time_step,
                    Grid2DRectangular& global_field,
                    Geometry2D& global_geometry)
    {
        auto Ez = global_field._Ez.template view<memory_space>();
        global_field._Ez.template sync<memory_space>();
        auto space_step = global_geometry._x.dx;
        auto x_index = global_geometry._x.getIndex(_center._x);
        auto y_index = global_geometry._y.getIndex(_center._y);
        Ez(x_index, y_index) += _amplitude * Kokkos::cos(_freq * time) * time_step / space_step;
        global_field._Ez.template modify<memory_space>();
    };
};

struct PlaneWave
{
    double _freq;
    double _amplitude;
    Point2D _center;
    Point2D _direction;
    Kokkos::DualView<Kokkos::pair<int, int>*> _plane_points;

    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    DevicePolicy1D _policy;
    using view_type = Types<ExecutionSpace>::view_type;
    using memory_space = Types<ExecutionSpace>::memory_space;

    PlaneWave(double f, double am, Point2D cen, Point2D dir):
        _freq(f), _amplitude(am), _center(cen), _direction(dir)
    {
        normilize(_direction);
    };

    KOKKOS_INLINE_FUNCTION void normilize(Point2D& f)
    {
        auto norm = f._x * f._x + f._y * f._y;
        auto factor = Kokkos::sqrt(norm);
        f._x = f._x / factor;
        f._y = f._y / factor;
    }

    void Initialize(Grid2DRectangular& global_field, Geometry2D& global_geometry)
    {
        auto center_x = _center._x;
        auto center_y = _center._y;
        auto direction = _direction;
        double epsilon_for_scalar_product { 1 * global_geometry._x.dx }; // TODO

        int number_of_points { 0 };
        std::vector<std::pair<int, int>> points;

        int nx = global_geometry._x._N;
        int ny = global_geometry._y._N;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                auto pos_x = global_geometry._x.getCoord(i);
                auto pos_y = global_geometry._y.getCoord(j);
                auto vecFromCenter = Point2D(pos_x - center_x, pos_y - center_y);
                if (std::abs(dot(vecFromCenter, direction)) < epsilon_for_scalar_product)
                {

                    number_of_points++;
                    points.emplace_back(std::pair<int, int>(i, j));
                }
            }
        }

        _plane_points =
            Kokkos::DualView<Kokkos::pair<int, int>*>("points of plane wave source", number_of_points);
        _policy = DevicePolicy1D(0, number_of_points);
        auto host_plane_points = _plane_points.h_view;
        for (int i = 0; i < number_of_points; i++)
        {
            host_plane_points(i) = points[i];
        }
        _plane_points.sync<memory_space>();
#ifdef USE_SPDLOG
        spdlog::info("number of points for plane wave source is {} \n", number_of_points);
#endif
    }

    void Propagator(double time,
                    double time_step,
                    Grid2DRectangular& global_field,
                    Geometry2D& global_geometry)
    {
        auto Ez = global_field._Ez.view<memory_space>();
        auto amplitude = _amplitude;
        auto freq = _freq;
        auto plane_points = _plane_points.view_device();
        Kokkos::parallel_for(
            _policy, KOKKOS_LAMBDA(const int& index) {
                int i = plane_points(index).first;
                int j = plane_points(index).second;
                Ez(i, j) += amplitude * Kokkos::cos(freq * time) * time_step;
            });

        global_field._Ez.modify<memory_space>();
    }
};

} // namespace Prop
