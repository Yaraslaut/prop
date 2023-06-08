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
#include "types.h"

#include <iostream>

namespace Prop
{

struct IsotropicMedium
{

    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    IsotropicMedium(Axis x, Axis y, double eps, double sigm, double mu):
        _epsilon(eps), _mu(mu), _sigma(sigm), _box(x, y) {};

    Box _box;
    int _entity_id;

    double _epsilon;
    double _mu;
    double _sigma;

    void Propagator(double time,
                    double time_step,
                    Grid2DRectangular global_field,
                    Geometry2D& global_geometry)
    {
        auto props = global_geometry.getProperties(_box);
        const int x_offset = std::get<0>(props);
        const int x_size = std::get<1>(props);
        const int y_offset = std::get<2>(props);
        const int y_size = std::get<3>(props);

        auto Ez = global_field._Ez.view<memory_space>();
        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();
        auto entity_ind = global_field._which_entity.view<memory_space>();

        global_field._Ez.sync<memory_space>();
        global_field._Hx.sync<memory_space>();
        global_field._Hy.sync<memory_space>();
        global_field._which_entity.sync<memory_space>();

        // Mark Hx and Hy as modified
        global_field._Ez.modify<memory_space>();
        global_field._Hx.modify<memory_space>();
        global_field._Hy.modify<memory_space>();

        auto sigma { _sigma };
        auto mu { _mu };
        auto epsilon { _epsilon };

        auto space_step = global_geometry._x.dx;
        auto entity_id = _entity_id;

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });
        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = iinit + x_offset;
                const int j = jinit + y_offset;
                if (entity_ind(i, j) == entity_id)
                {
                    auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                    auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                    auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };
                    auto C_hyh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                    auto C_hye { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };

                    Hx(i, j) = C_hxh * Hx(i, j) - C_hxe * (Ez(i, j + 1) - Ez(i, j));
                    Hy(i, j) = C_hyh * Hy(i, j) + C_hye * (Ez(i + 1, j) - Ez(i, j));
                }
            });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = iinit + x_offset;
                const int j = jinit + y_offset;
                if (entity_ind(i, j) == entity_id)
                {
                    auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                    auto C_eze { (1.0 - (sigma * time_step) / (2.0 * epsilon))
                                 / (1.0 + (sigma * time_step) / (2.0 * epsilon)) };
                    auto C_ezhx { time_step / (1.0 + (sigma * time_step) / (2.0 * epsilon))
                                  / (epsilon * space_step) };
                    auto C_ezhy { time_step / (1.0 + (sigma * time_step) / (2.0 * epsilon))
                                  / (epsilon * space_step) };

                    Ez(i, j) = C_eze * Ez(i, j) + C_ezhx * (Hy(i, j) - Hy(i - 1, j))
                               - C_ezhy * (Hx(i, j) - Hx(i, j - 1));
                }
            });
    }
};

struct PMLRegion
{
    using GridData = GridData2D_dual;
    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;
    PMLRegion(Axis x, Axis y): _box(x, y) {};

    Box _box;
    int _entity_id;
    GridData _Psi_Ez_x;
    void updatePML(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto props = global_geometry.getProperties(_box);
        const int x_offset = std::get<0>(props);
        const int x_size = std::get<1>(props);
        const int y_offset = std::get<2>(props);
        const int y_size = std::get<3>(props);

        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();
        auto entity_ind = global_field._which_entity.view<memory_space>();

        auto space_step = global_geometry._x.dx;
        auto entity_id = _entity_id;

        auto Psi_Ez_x = _Psi_Ez_x.view<memory_space>();
        _Psi_Ez_x.sync<memory_space>();
        _Psi_Ez_x.modify<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });
        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = iinit + x_offset;
                const int j = jinit + y_offset;

                auto sigma { 1.1 };
                auto alpha { 1.0 };
                auto kappa { 1.5 };

                auto b_x = time_step * Kokkos::exp(sigma / kappa + alpha);
                auto c_x = sigma * ((b_x - 1) / (sigma + kappa * alpha) / kappa);
                // std::cout << c_x / space_step << std::endl;

                Psi_Ez_x(iinit, jinit) =
                    b_x * Psi_Ez_x(iinit, jinit) + c_x * (Hy(i, j) - Hy(i - 1, j)) / space_step;
            });
    }

    void applyPML(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto props = global_geometry.getProperties(_box);
        const int x_offset = std::get<0>(props);
        const int x_size = std::get<1>(props);
        const int y_offset = std::get<2>(props);
        const int y_size = std::get<3>(props);

        auto Ez = global_field._Ez.view<memory_space>();
        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();
        auto entity_ind = global_field._which_entity.view<memory_space>();

        global_field._Ez.sync<memory_space>();
        global_field._Hx.sync<memory_space>();
        global_field._Hy.sync<memory_space>();
        global_field._which_entity.sync<memory_space>();

        // Mark Hx and Hy as modified
        global_field._Ez.modify<memory_space>();
        global_field._Hx.modify<memory_space>();
        global_field._Hy.modify<memory_space>();

        auto space_step = global_geometry._x.dx;
        auto entity_id = _entity_id;

        auto Psi_Ez_x = _Psi_Ez_x.view<memory_space>();
        _Psi_Ez_x.sync<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });
        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = iinit + x_offset;
                const int j = jinit + y_offset;

                auto epsilon { 1.0 };
                auto mu { 1.0 };

                auto sigma { 1.1 };
                auto alpha { 1.0 };
                auto kappa { 1.5 };

                auto sigma_factor { (sigma * time_step) / (2.0 * epsilon) };
                auto C_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto C_b { time_step / epsilon / (1.0 + sigma_factor) };

                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };
                auto C_hyh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hye { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };

                Hx(i, j) = C_hxh * Hx(i, j) - C_hxe * (Ez(i, j + 1) - Ez(i, j));
                Hy(i, j) = C_hyh * Hy(i, j) + C_hye * (Ez(i + 1, j) - Ez(i, j));
            });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = iinit + x_offset;
                const int j = jinit + y_offset;

                auto epsilon { 1.0 };

                auto sigma { 1.1 };
                auto alpha { 1.0 };
                auto kappa { 1.5 };

                auto sigma_factor { (sigma * time_step) / (2.0 * epsilon) };
                auto C_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto C_b { time_step / epsilon / (1.0 + sigma_factor) };

                Ez(i, j) = C_a * Ez(i, j)
                           + C_b
                                 * ((Hy(i, j) - Hy(i - 1, j)) / (kappa * space_step)
                                    - (Hx(i, j) - Hx(i, j - 1)) / (kappa * space_step) + Psi_Ez_x(i, j));
            });
    }
};

} // namespace Prop

/*
** for(int i = 0; i < CPMLGrid; i++) {
        sigma_e[i] = sigma_max * pow( (double(i+0.5)/CPMLGrid) , m);
        sigma_h[i] = sigma_max * pow( (double(i+1.0)/CPMLGrid) , m);

        kappa_e[i] = 1 + (kappa_max-1) * pow( (double(i+0.5)/CPMLGrid) , m);
        kappa_h[i] = 1 + (kappa_max-1) * pow( (double(i+1.0)/CPMLGrid) , m);

        alpha_e[i] = alpha_max * pow( (double(CPMLGrid-(i+0.0))/CPMLGrid) , ma);
        alpha_h[i] = alpha_max * pow( (double(CPMLGrid-(i+0.5))/CPMLGrid) , ma);

        B_e[i] = exp( -(dt/eps0) * (sigma_e[i]/kappa_e[i] + alpha_e[i]) );
        B_h[i] = exp( -(dt/eps0) * (sigma_h[i]/kappa_h[i] + alpha_h[i]) );

        C_e[i] = sigma_e[i] / kappa_e[i] / (sigma_e[i] + kappa_e[i]*alpha_e[i])
                 * (B_e[i] - 1.0);
        C_h[i] = sigma_h[i] / kappa_h[i] / (sigma_h[i] + kappa_h[i]*alpha_h[i])
                 * (B_h[i] - 1.0);
    }
*/
