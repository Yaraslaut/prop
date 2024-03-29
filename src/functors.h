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
#include "types.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace Prop
{


template <class ExecutionSpace>
struct updateMagneticFieldFreeSpace
{

    using memory_space = typename Types<ExecutionSpace>::memory_space;
    using FieldViewType = typename Types<ExecutionSpace>::FieldViewType;
    using IndexViewType = typename Types<ExecutionSpace>::IndexViewType;

    FieldViewType Ez;
    FieldViewType Hx;
    FieldViewType Hy;
    IndexViewType entity_ind;

    int _nx;
    int _ny;

    double time_step;
    double space_step;

    updateMagneticFieldFreeSpace(Grid2DRectangular global_field, double dt, double dspace, int nx, int ny):
        time_step(dt), space_step(dspace), _nx(nx), _ny(ny)
    {

        Ez = global_field._Ez.view<memory_space>();
        Hx = global_field._Hx.view<memory_space>();
        Hy = global_field._Hy.view<memory_space>();
        entity_ind = global_field._which_entity.view<memory_space>();

        global_field._Ez.sync<memory_space>();
        global_field._Hx.sync<memory_space>();
        global_field._Hy.sync<memory_space>();
        global_field._which_entity.sync<memory_space>();

        // Mark Hx and Hy as modified
        global_field._Hx.modify<memory_space>();
        global_field._Hy.modify<memory_space>();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int iinit, const int jinit) const
    {

        auto getIndex = [](int ind, int N) {
            if (ind == -1)
                return N - 1;
            if (ind == N)
                return 0;
            return ind;
        };

        int i = getIndex(iinit, _nx);
        int ipo = getIndex(iinit + 1, _nx);

        int j = getIndex(jinit, _ny);
        int jpo = getIndex(jinit + 1, _ny);

        if (entity_ind(i, j) == 0)
        {
            auto mu { 1.0 };
            auto sigma { 0.0 };

            auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
            auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
            auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };
            auto C_hyh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
            auto C_hye { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };

            Hx(i, j) = C_hxh * Hx(i, j) - C_hxe * (Ez(i, jpo) - Ez(i, j));
            Hy(i, j) = C_hyh * Hy(i, j) + C_hye * (Ez(ipo, j) - Ez(i, j));
        }
    }
};

template <class ExecutionSpace>
struct updateElectricFieldFreeSpace
{

    using memory_space = typename Types<ExecutionSpace>::memory_space;
    using FieldViewType = typename Types<ExecutionSpace>::FieldViewType;
    using IndexViewType = typename Types<ExecutionSpace>::IndexViewType;

    FieldViewType Ez;
    FieldViewType Hx;
    FieldViewType Hy;
    IndexViewType entity_ind;

    int _nx;
    int _ny;
    double _time_step;
    double _space_step;

    updateElectricFieldFreeSpace(Grid2DRectangular global_field, double dt, double dspace, int nx, int ny):
        _time_step(dt), _space_step(dspace), _nx(nx), _ny(ny)
    {

        Ez = global_field._Ez.view<memory_space>();
        Hx = global_field._Hx.view<memory_space>();
        Hy = global_field._Hy.view<memory_space>();
        entity_ind = global_field._which_entity.view<memory_space>();

        global_field._Ez.sync<memory_space>();
        global_field._Hx.sync<memory_space>();
        global_field._Hy.sync<memory_space>();
        global_field._which_entity.sync<memory_space>();

        // Mark Hx and Hy as modified
        global_field._Hx.modify<memory_space>();
        global_field._Hy.modify<memory_space>();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int iinit, const int jinit) const
    {
        auto getIndex = [](int ind, int N) {
            if (ind == -1)
                return N - 1;
            if (ind == N)
                return 0;
            return ind;
        };

        int i = getIndex(iinit, _nx);
        int imo = getIndex(iinit - 1, _nx);

        int j = getIndex(jinit, _ny);
        int jmo = getIndex(jinit - 1, _ny);

        if (entity_ind(i, j) == 0)
        {
            auto mu { 1.0 };
            auto sigma { 0.0 };
            auto epsilon { 1.0 };
            auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * _time_step) / (2.0 * mu)) };
            auto C_eze { (1.0 - (sigma * _time_step) / (2.0 * epsilon))
                         / (1.0 + (sigma * _time_step) / (2.0 * epsilon)) };
            auto C_ezhx { _time_step / (1.0 + (sigma * _time_step) / (2.0 * epsilon))
                          / (epsilon * _space_step) };
            auto C_ezhy { _time_step / (1.0 + (sigma * _time_step) / (2.0 * epsilon))
                          / (epsilon * _space_step) };

            Ez(i, j) = C_eze * Ez(i, j) + C_ezhx * (Hy(i, j) - Hy(imo, j)) - C_ezhy * (Hx(i, j) - Hx(i, jmo));
        }
    }
};

} // namespace Prop
