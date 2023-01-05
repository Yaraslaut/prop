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
# */

#pragma once

#include "field.h"
#include "functors.h"
#include "geometry.h"
#include "medium.h"
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
        double factor { Const_c / Const_standard_courant_factor };
        _stable_time_step = 0.1 * _space_step / Const_standard_courant_factor;

#ifdef USE_SPDLOG
        spdlog::debug("stable time step: {} \n", _stable_time_step);
#endif
        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
        }
        x.calcN(_space_step);
        y.calcN(_space_step);
        _geometry = Geometry2D(x, y);
        _field = Grid2DRectangular(x._N, y._N);
    };

    const External_data& getExternal(Components comp) { return _field.getExternal(comp); }
    Eigen::MatrixXd getEpsilon();
    SimplePolicy2D getPolicy() { return _field.getPolicy(); }
    void addBlock(IsotropicMedium& block);
    void addBlock(PMLRegionX& pml_block);
    void addBlock(PMLRegionY& pml_block);

    void addSourceEz(PlaneWave& pw) { _entities_plane_wave.push_back(std::make_unique<PlaneWave>(pw)); }
    void addSourceEz(PointSource& pw) { _entities_point_source.push_back(std::make_unique<PointSource>(pw)); }

    void propagateCustom(double total_time);
    void propagateFixedTime(double time_step);
    void propagate() { propagateCustom(_stable_time_step); }

    int getNx() { return static_cast<int>(_geometry._x._N); }
    int getNy() { return static_cast<int>(_geometry._y._N); }

  private:
    int _max_entity_id { 0 };
    bool _first_time { true };
    double _time { 0.0 };
    double _stable_time_step = std::numeric_limits<double>::signaling_NaN();
    double _resolution = std::numeric_limits<double>::signaling_NaN();
    double _space_step = std::numeric_limits<double>::signaling_NaN();

    Geometry2D _geometry;
    Grid2DRectangular _field;

    template <typename T>
    using EntityVector = std::vector<std::unique_ptr<T>>;

    EntityVector<PointSource> _entities_point_source;
    EntityVector<PlaneWave> _entities_plane_wave;

    EntityVector<IsotropicMedium> _entities_material;
    EntityVector<PMLRegionX> _entities_pml_x;
    EntityVector<PMLRegionY> _entities_pml_y;
};

} // namespace Prop
