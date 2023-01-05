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
#include "sources.h"
#include "types.h"

#include <memory>
namespace Prop
{
class System2D
{
    using GridData = GridData2D_host;
    using External_data = External2D_data;
    using Components = Components2DTM;

  public:
    System2D(Axis x, Axis y): _geometry(x, y), _field(x._N, y._N)
    {
        _space_step = x.dx; // lambda_characteristic / resolution
        // x_step *0.5 / Const_c;
        //      _step_size_factor = 0.5; //_time_step * Const_c / _x_step;
        _resolution = 1.0; // lambda_characteristic / _x_step
        ///spdlog::info("[System] created with size : {:d} {:d}", x._N, y._N);
        if (!Kokkos::is_initialized())
        {
            // spdlog::debug("[TwoDimensionalSystem] Initializing Kokkos...");
            Kokkos::initialize();
        }
    };

    const External_data& getExternal(Components comp) { return _field.getExternal(comp); }

    SimplePolicy2D getPolicy() { return _field.getPolicy(); }
    void addBlock(Block2D& block) { _geometry.addBlock(block); };

    void addSourceEz(PlaneWave2D& pw) { _sources_Ez.push_back(std::make_unique<PlaneWave2D>(pw)); }

    void propagateCustom(double);
    void propagateFixedTime(double);
    void propagate() { propagateCustom(_courant * Const_c / std::sqrt(2.0 / _space_step / _space_step)); }

  private:
    bool _firstTime { true };
    double _time { 0.0 };
    double _courant { 0.5 };
    double _resolution = std::numeric_limits<double>::signaling_NaN();
    //    double _step_size_factor;
    double _space_step = std::numeric_limits<double>::signaling_NaN();

    Geometry2D _geometry;

    std::vector<std::unique_ptr<PlaneWave2D>> _sources_Ez;
    Grid2DRectangular _field;
};
} // namespace Prop
