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

#include "sources.h"
#include "types.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace Prop
{

template <class ExecutionSpace>
struct updateMagneticFieldFreeSpace
{

    using view_type = GridData2D_dual;
    typedef ExecutionSpace execution_space;

    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    Kokkos::View<Field_data_type**, memory_space> Ez;
    Kokkos::View<Field_data_type**, memory_space> Hx;
    Kokkos::View<Field_data_type**, memory_space> Hy;
    double time_step;
    double space_step;

    updateMagneticFieldFreeSpace(view_type dv_Ez, view_type dv_Hx, view_type dv_Hy, double dt, double dspace):
        time_step(dt), space_step(dspace)
    {

        Ez = dv_Ez.view<memory_space>();
        Hx = dv_Hx.view<memory_space>();
        Hy = dv_Hy.view<memory_space>();

        dv_Ez.sync<memory_space>();
        dv_Hx.sync<memory_space>();
        dv_Hy.sync<memory_space>();

        // Mark Hx and Hy as modified
        dv_Hx.modify<memory_space>();
        dv_Hy.modify<memory_space>();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int iinit, const int jinit) const
    {

        auto mu { 1.0 };
        auto sigma { 0.0 };

        auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
        auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
        auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };
        auto C_hyh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
        auto C_hye { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };

        int N = 100;
        auto getIndex = [](int ind, int N) {
            if (ind == -1)
                return N - 1;
            if (ind == N)
                return 0;
            return ind;
        };

        int i = getIndex(iinit, N);
        int ipo = getIndex(iinit + 1, N);

        int j = getIndex(jinit, N);
        int jpo = getIndex(jinit + 1, N);



        Hx(i, j) = C_hxh * Hx(i, j) - C_hxe * (Ez(i, jpo) - Ez(i, j));
        Hy(i, j) = C_hyh * Hy(i, j) + C_hye * (Ez(ipo, j) - Ez(i, j));
    }
};

template <class ExecutionSpace>
struct updateElectricFieldFreeSpace
{

    using view_type = GridData2D_dual;
    typedef ExecutionSpace execution_space;

    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    Kokkos::View<Field_data_type**, memory_space> Ez;
    Kokkos::View<Field_data_type**, memory_space> Hx;
    Kokkos::View<Field_data_type**, memory_space> Hy;
    double _time_step;
    double _space_step;

    updateElectricFieldFreeSpace(view_type dv_Ez, view_type dv_Hx, view_type dv_Hy, double dt, double dspace):
        _time_step(dt), _space_step(dspace)
    {

        Ez = dv_Ez.view<memory_space>();
        Hx = dv_Hx.view<memory_space>();
        Hy = dv_Hy.view<memory_space>();

        dv_Ez.sync<memory_space>();
        dv_Hx.sync<memory_space>();
        dv_Hy.sync<memory_space>();

        // Mark Hx and Hy as modified
        dv_Ez.modify<memory_space>();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int iinit, const int jinit) const
    {

        auto mu { 1.0 };
        auto sigma { 0.0 };
        auto epsilon { 1.0 };
        auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * _time_step) / (2.0 * mu)) };
        auto C_eze { (1.0 - (sigma * _time_step) / (2.0 * epsilon))
                     / (1.0 + (sigma * _time_step) / (2.0 * epsilon)) };
        auto C_ezhx { _time_step / (1.0 + (sigma * _time_step) / (2.0 * epsilon)) / (epsilon * _space_step) };
        auto C_ezhy { _time_step / (1.0 + (sigma * _time_step) / (2.0 * epsilon)) / (epsilon * _space_step) };

        int N = 100;
        auto getIndex = [](int ind, int N) {
            if (ind == -1)
                return N - 1;
            if (ind == N)
                return 0;
            return ind;
        };

        int i = getIndex(iinit, N);
        int imo = getIndex(iinit - 1, N);

        int j = getIndex(jinit, N);
        int jmo = getIndex(jinit - 1, N);

        Ez(i, j) = C_eze * Ez(i, j) + C_ezhx * (Hy(i, j) - Hy(imo, j)) - C_ezhy * (Hx(i, j) - Hx(i, jmo));
    }
};

template <class ExecutionSpace>
struct updateFromPlaneWave
{

    using view_type = GridData2D_dual;
    typedef ExecutionSpace execution_space;

    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    Kokkos::View<Field_data_type**, memory_space> Ez;
    Kokkos::View<Field_data_type**, memory_space> Hx;
    Kokkos::View<Field_data_type**, memory_space> Hy;
    double _time;
    double _time_step;

    updateFromPlaneWave(view_type dv_Ez, view_type dv_Hx, view_type dv_Hy, double time, double dt):
        _time(time), _time_step(dt)
    {

        Ez = dv_Ez.view<memory_space>();
        Hx = dv_Hx.view<memory_space>();
        Hy = dv_Hy.view<memory_space>();

        dv_Ez.sync<memory_space>();
        dv_Hx.sync<memory_space>();
        dv_Hy.sync<memory_space>();

        // Mark Hx and Hy as modified
        dv_Ez.modify<memory_space>();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j) const
    {

        auto some_function = [](int i, int j) -> double {
            auto exp = [](double x, double sigma) -> double {
                return Kokkos::exp(-0.5 * Kokkos::pow(x / sigma, 2.0))
                       / (sigma * Kokkos::sqrt(2 * Kokkos::numbers::pi));
            };
            return exp(i - 50, 1.0) * exp(j - 50, 1.0);
        };
        double freq = 2.0 * Kokkos::numbers::pi / 1e-6;
        Ez(i, j) += some_function(i, j) * Kokkos::cos(_time * freq) * _time_step;
    }
};

} // namespace Prop