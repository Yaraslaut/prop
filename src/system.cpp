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

using namespace Prop;

Eigen::MatrixXd System2D::getEpsilon()
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

void System2D::addBlock(IsotropicMedium& block)
{
    _max_entity_id++;
#ifdef USE_SPDLOG
    spdlog::info("Add block to the system with index {} \n", _max_entity_id);
#endif
    block._entity_id = _max_entity_id;
    auto props = _geometry.getProperties(block._box);
    const int x_offset = std::get<0>(props);
    const int x_size = std::get<1>(props);
    const int y_offset = std::get<2>(props);
    const int y_size = std::get<3>(props);

    auto policy = SimplePolicy2D({ 0, 0 }, { x_size, y_size });
    auto entity_ind = _field._which_entity.view_host();
    auto max_entity_id = _max_entity_id;
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
            const int i = iinit + x_offset;
            const int j = jinit + y_offset;
            entity_ind(i, j) = max_entity_id;
        });
    _entities_material.push_back(std::make_unique<IsotropicMedium>(block));
    _field._which_entity.modify_host();
};

void System2D::addBlock(PMLRegionX& pml_block)
{

#ifdef USE_SPDLOG
    spdlog::debug("added pml X region \n");
#endif
    _max_entity_id++;

    pml_block._entity_id = _max_entity_id;
    auto props = _geometry.getProperties(pml_block._box);
    const int x_offset = std::get<0>(props);
    const int x_size = std::get<1>(props);
    const int y_offset = std::get<2>(props);
    const int y_size = std::get<3>(props);

    auto policy = SimplePolicy2D({ 0, 0 }, { x_size, y_size });
    auto entity_ind = _field._which_entity.view_host();
    auto max_entity_ind = _max_entity_id;
    _field._which_entity.sync_host();
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
            const int i = iinit + x_offset;
            const int j = jinit + y_offset;
            entity_ind(i, j) = max_entity_ind;
        });
    pml_block._box._x.calcN(_space_step);
    pml_block._box._y.calcN(_space_step);
    pml_block._Psi_Ez_x = GridData2D_dual("PML region aux field psi_ez_x", x_size, y_size);
    pml_block._Psi_Hy_x = GridData2D_dual("PML region aux field psi_hy_x", x_size, y_size);

    _entities_pml_x.push_back(std::make_unique<PMLRegionX>(pml_block));
    _field._which_entity.modify_host();
};

void System2D::addBlock(PMLRegionY& pml_block)
{

#ifdef USE_SPDLOG
    spdlog::debug("added pml Y region \n");
#endif
    _max_entity_id++;

    pml_block._entity_id = _max_entity_id;
    const auto [x_offset, x_size, y_offset, y_size] = _geometry.getProperties(pml_block._box);

    auto policy = SimplePolicy2D({ 0, 0 }, { x_size, y_size });
    auto entity_ind = _field._which_entity.view_host();
    auto max_entity_ind = _max_entity_id;
    _field._which_entity.sync_host();
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
            const int i = iinit + x_offset;
            const int j = jinit + y_offset;
            entity_ind(i, j) = max_entity_ind;
        });
    pml_block._box._x.calcN(_space_step);
    pml_block._box._y.calcN(_space_step);
    pml_block._Psi_Ez_y = GridData2D_dual("PML region aux field psi_ez_y", x_size, y_size);
    pml_block._Psi_Hx_y = GridData2D_dual("PML region aux field psi_hx_y", x_size, y_size);

    _entities_pml_y.push_back(std::make_unique<PMLRegionY>(pml_block));
    _field._which_entity.modify_host();
};

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

    auto prop = [&](auto&& ent) {
        for (auto& p: ent)
        {
            p->Propagator(_time, time_step, this->_field, this->_geometry);
        }
    };

    auto pml = [&]() {
        for (auto& p: _entities_pml_x)
        {
            p->updatePML(_time, time_step, this->_field, this->_geometry);
        }

        for (auto& p: _entities_pml_y)
        {
            p->updatePML(_time, time_step, this->_field, this->_geometry);
        }
    };

    // START update of grid
    pml();

    prop(_entities_point_source);
    prop(_entities_plane_wave);
    prop(_entities_material);

    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateMagneticFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field, time_step, _space_step, _geometry._x._N, _geometry._y._N));

    Kokkos::parallel_for(_field.getDevicePolicy(),
                         updateElectricFieldFreeSpace<GridData2D_dual::execution_space>(
                             _field, time_step, _space_step, _geometry._x._N, _geometry._y._N));

    pml();

    Kokkos::fence();

    _time += time_step;
#ifdef USE_SPDLOG
    spdlog::info("time is {} \n", _time);
#endif
}
