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

#include <cmath>

Eigen::MatrixXd Prop::System2D::getEpsilon()
{
    auto nx = _geometry._x._N;
    auto ny = _geometry._y._N;
    auto epsilon = Eigen::MatrixXd(nx, ny);
    auto entity_ind = _field._which_entity.view_host();
#ifdef USE_SPDLOG
    spdlog::info(" {}x{} \n", nx, ny);
#endif
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            epsilon(i, j) = 1.0;
            for (auto& p: _entities_material)
            {
                if (entity_ind(i, j) == p->_entity_id)
                {
                    epsilon(i, j) = p->_epsilon;
#ifdef USE_SPDLOG
                    spdlog::debug(
                        "get from entity {}  in i:{} j:{} epsilon {} \n", p->_entity_id, i, j, p->_epsilon);
#endif
                }
            }
        }
    }
    return epsilon;
}

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
        for (auto& p: _entities_plane_wave)
        {
            p->Initialize(this->_field, this->_geometry);
        }
    };

    for (auto& p: _entities_pml_x)
    {
        p->updatePML(_time, time_step, this->_field, this->_geometry);
    }

    for (auto& p: _entities_pml_y)
    {
        p->updatePML(_time, time_step, this->_field, this->_geometry);
    }

    // START update of grid
    for (auto& p: _entities_point_source)
    {
        p->Propagator(_time, time_step, this->_field, this->_geometry);
    }

    for (auto& p: _entities_plane_wave)
    {
        p->Propagator(_time, time_step, this->_field, this->_geometry);
    }

    for (auto& p: _entities_material)
    {
        p->Propagator(_time, time_step, this->_field, this->_geometry);
    }

    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateMagneticFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field, time_step, _space_step, _geometry._x._N, _geometry._y._N));

    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateElectricFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field, time_step, _space_step, _geometry._x._N, _geometry._y._N));

    for (auto& p: _entities_pml_x)
    {
        p->applyPML(_time, time_step, this->_field, this->_geometry);
    }

    for (auto& p: _entities_pml_y)
    {
        p->applyPML(_time, time_step, this->_field, this->_geometry);
    }

    Kokkos::fence();

    _time += time_step;
#ifdef USE_SPDLOG
    spdlog::info("time is {} \n", _time);
#endif
}
