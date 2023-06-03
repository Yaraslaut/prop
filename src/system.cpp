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
void Prop::System2D::propagateCustom(double total_time)
{
    if (total_time < _stable_time_step)
        return propagateFixedTime(total_time);
    double accumulated_time_step { 0.0 };
    while (accumulated_time_step < total_time)
    {
        accumulated_time_step += _stable_time_step;
        propagateFixedTime(_stable_time_step);
    }
}
void Prop::System2D::propagateFixedTime(double time_step)
{

    if (_first_time)
    {
        _first_time = false;
        this->_geometry.fillGeometryFromEntity();
    };

    for(auto& p : _entities_point_source)
    {
        p->Propagator(_time, time_step, _field);
    }

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
