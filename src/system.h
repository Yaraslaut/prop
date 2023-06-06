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

#include "Kokkos_Core_fwd.hpp"
#include "field.h"
#include "functors.h"
#include "geometry.h"
#include "pml.h"
#include "sources.h"
#include "types.h"

#include <functional>
#include <limits>
#include <memory>
namespace Prop
{

class System2D
{
    using GridData = GridData2D_dual;
    using External_data = External2D_data;
    using Components = Components2DTM;

  public:
    System2D(Axis x, Axis y, double pts_per_wavelength = 5)
    {
        double fmax { Const_c / Const_scaling_factor }; // TODO
        _resolution = pts_per_wavelength;
        _space_step = Const_c / (pts_per_wavelength * fmax);
        // lambda_characteristic / resolution
        std::cout << _space_step << std::endl;
        double factor { Const_c / Const_standard_courant_factor };
        _stable_time_step = _space_step / Const_standard_courant_factor;

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
        }

        x.calcN(_space_step);
        y.calcN(_space_step);
        _geometry = Geometry2D(x, y);
        _field = Grid2DRectangular(x._N, y._N);

        // auto freq = 2.0 * Kokkos::numbers::pi / 1e-6;
        // PointSource point_source(60, 60, freq);
        // _entities_point_source.push_back(std::make_unique<PointSource>(point_source));

        // freq = 1.0 * 2.0 * Kokkos::numbers::pi / 1e-6;
        // PlaneWave plane_wave(freq, 1.0, Point2D(0.0,0.0), Point2D(0.0,0.0));
        // _entities_plane_wave.push_back(std::make_unique<PlaneWave>(plane_wave));
    };

    const External_data& getExternal(Components comp) { return _field.getExternal(comp); }

    SimplePolicy2D getPolicy() { return _field.getPolicy(); }
    void addBlock(Block2D& block) { _geometry.addBlock(block); };
    void addSourceEz(PlaneWave& pw) {
        _entities_plane_wave.push_back(std::make_unique<PlaneWave>(pw)); }
    void addSourceEz(PointSource& pw) {
        _entities_point_source.push_back(std::make_unique<PointSource>(pw));
    }

    void propagateCustom(double total_time);
    void propagateFixedTime(double time_step);
    void propagate() { propagateCustom(_stable_time_step); }

    int getNx() { return static_cast<int>(_geometry._x._N); }
    int getNy() { return static_cast<int>(_geometry._y._N); }

  private:
    bool _first_time { true };
    double _time { 0.0 };
    double _stable_time_step = std::numeric_limits<double>::signaling_NaN();
    double _resolution = std::numeric_limits<double>::signaling_NaN();
    //    double _step_size_factor;
    double _space_step = std::numeric_limits<double>::signaling_NaN();

    Geometry2D _geometry;
    Grid2DRectangular _field;

    std::vector<std::unique_ptr<PointSource>> _entities_point_source;
    std::vector<std::unique_ptr<PlaneWave>> _entities_plane_wave;
    std::vector<std::unique_ptr<PMLregion>> _entities_pml_region;
};

} // namespace Prop
