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

#include "system.h"

#include "Kokkos_Core_fwd.hpp"
#include "functors.h"
#include "geometry.h"
#include "types.h"

// #include "Kokkos_Complex.hpp"
#include <cmath>
void Prop::System2D::propagateCustom(double totalTime)
{
    double max_time_step = _courant / Const_c / std::sqrt(2.0 / _space_step / _space_step);

    if (totalTime < max_time_step)
        return propagateFixedTime(totalTime);
    double accumulated_time_step { 0.0 };
    while (accumulated_time_step < totalTime)
    {
        accumulated_time_step += max_time_step;
        propagateFixedTime(max_time_step);
    }
}
void Prop::System2D::propagateFixedTime(double time_step)
{

    auto courant { Const_c * time_step * std::sqrt(2.0 / _space_step / _space_step) };

    auto update_from_source = [&]() {
        Kokkos::parallel_for(
            "update electric field", _field.getPolicy(), KOKKOS_LAMBDA(int i, int j) {
                auto x = _geometry._x.getCoord(i);
                auto y = _geometry._y.getCoord(j);
                auto Ez = _field._Ez.template view<typename GridData2D_dual::host_mirror_space>();
                for (auto& sourceEz: _sources_Ez)
                {
                    Ez(i, j) += sourceEz->getField(_time, Point2D { x, y }) * time_step;
                }
                _field._Ez.modify<Kokkos::DefaultHostExecutionSpace>();
            });
    };


    if (_firstTime)
    {
        _firstTime = false;
        this->_geometry.fillGeometryFromEntity();
    };

    //update_from_source();


    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateFromPlaneWave<GridData2D_dual::execution_space>(
                             _field._Ez, _field._Hx, _field._Hy, _time, time_step));


    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateMagneticFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field._Ez, _field._Hx, _field._Hy, time_step, _space_step));

    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateElectricFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field._Ez, _field._Hx, _field._Hy, time_step, _space_step));
    Kokkos::fence();


    _time += time_step;
    std::cout << "time is " << _time << '\n';
}
