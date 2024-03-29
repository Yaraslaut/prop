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

namespace Prop
{

struct PMLRegionX
{
    using GridData = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      GridData::memory_space,
                                      GridData::host_mirror_space>::type memory_space;
    PMLRegionX(Axis x, Axis y): _box(x, y) {};

    Box _box;
    int _entity_id;
    GridData _Psi_Ez_x;
    GridData _Psi_Hy_x;
    void updatePML(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto props = global_geometry.getProperties(_box);
        const int x_offset = std::get<0>(props);
        const int x_size = std::get<1>(props);
        const int y_offset = std::get<2>(props);
        const int y_size = std::get<3>(props);

        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();
        auto Ez = global_field._Ez.view<memory_space>();
        auto entity_ind = global_field._which_entity.view<memory_space>();

        auto space_step = global_geometry._x.dx;
        auto entity_id = _entity_id;

        auto Psi_Ez_x = _Psi_Ez_x.view<memory_space>();
        _Psi_Ez_x.sync<memory_space>();
        _Psi_Ez_x.modify<memory_space>();

        auto Psi_Hy_x = _Psi_Hy_x.view<memory_space>();
        _Psi_Hy_x.sync<memory_space>();
        _Psi_Hy_x.modify<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });

        auto N_box = _box._x._N;
        auto nx = global_geometry._x._N;
        auto ny = global_geometry._y._N;

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int imo = getIndex(iinit + x_offset - 1, nx);
                const int ipo = getIndex(iinit + x_offset + 1, nx);

                const int j = getIndex(jinit + y_offset, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                double sigma = getSigma(i_inside, x_size);
                double alpha = getAlpha(i_inside, x_size);
                double kappa = getKappa(i_inside, x_size);

                auto b_x = time_step * Kokkos::exp(sigma / kappa + alpha);
                auto c_x = sigma * ((b_x - 1) / (sigma + kappa * alpha) / kappa);

                Psi_Ez_x(i_inside, j_inside) =
                    b_x * Psi_Ez_x(i_inside, j_inside) + c_x * (Hy(i, j) - Hy(imo, j)) / space_step;

                Psi_Hy_x(i_inside, j_inside) =
                    b_x * Psi_Hy_x(i_inside, j_inside) + c_x * (Ez(ipo, j) - Ez(i, j)) / space_step;
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

        auto Psi_Hy_x = _Psi_Hy_x.view<memory_space>();
        _Psi_Hy_x.sync<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });
        auto nx = global_geometry._x._N;
        auto ny = global_geometry._y._N;

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int ipo = getIndex(iinit + x_offset + 1, nx);
                const int j = getIndex(jinit + y_offset, ny);
                const int jpo = getIndex(jinit + y_offset + 1, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                auto epsilon { 1.0 };
                auto mu { 1.0 };

                double sigma = getSigma(i_inside, x_size);
                double alpha = getAlpha(i_inside, x_size);
                double kappa = getKappa(i_inside, x_size);

                auto sigma_factor { (sigma * time_step) / (2.0 * mu) };
                auto D_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto D_b { time_step / mu / (1.0 + sigma_factor) };

                Hx(i, j) = D_a * Hx(i, j) - D_b * (Ez(i, jpo) - Ez(i, j)) / (kappa * space_step);
                Hy(i, j) = D_a * Hy(i, j) + D_b * (Ez(ipo, j) - Ez(i, j)) / (kappa * space_step)
                           - D_b * Psi_Hy_x(i_inside, j_inside);
            });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int imo = getIndex(iinit + x_offset - 1, nx);
                const int j = getIndex(jinit + y_offset, ny);
                const int jmo = getIndex(jinit + y_offset - 1, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                auto epsilon { 1.0 };

                double sigma = getSigma(i_inside, x_size);
                double alpha = getAlpha(i_inside, x_size);
                double kappa = getKappa(i_inside, x_size);

                auto sigma_factor { (sigma * time_step) / (2.0 * epsilon) };
                auto C_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto C_b { time_step / epsilon / (1.0 + sigma_factor) };

                Ez(i, j) =
                    C_a * Ez(i, j)
                    + C_b
                          * ((Hy(i, j) - Hy(imo, j)) / (kappa * space_step)
                             - (Hx(i, j) - Hx(i, jmo)) / (kappa * space_step) + Psi_Ez_x(i_inside, j_inside));
            });
    }

  private:
    KOKKOS_INLINE_FUNCTION
    double getProperty(int ind, int N, double max, double min)
    {
        return min
               + (max - min) * Kokkos::abs(static_cast<double>(ind - N / 2.0)) / static_cast<double>(N / 2.0);
    }

    KOKKOS_INLINE_FUNCTION
    double getSigma(int ind, int N) { return getProperty(ind, N, 2.5, 0.1); }
    KOKKOS_INLINE_FUNCTION
    double getKappa(int ind, int N) { return getProperty(ind, N, 1.0, 2.0); }
    KOKKOS_INLINE_FUNCTION
    double getAlpha(int ind, int N) { return getProperty(ind, N, 1.0, 1.0); }

    KOKKOS_INLINE_FUNCTION
    int getIndex(int ind, int N)
    {
        if (ind == -1)
            return N - 1;
        if (ind == N)
            return 0;
        return ind;
    };
};

struct PMLRegionY
{
    using GridData = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      GridData::memory_space,
                                      GridData::host_mirror_space>::type memory_space;
    PMLRegionY(Axis x, Axis y): _box(x, y) {};

    Box _box;
    int _entity_id;
    GridData _Psi_Ez_y;
    GridData _Psi_Hx_y;
    void updatePML(double time, double time_step, Grid2DRectangular global_field, Geometry2D& global_geometry)
    {
        auto props = global_geometry.getProperties(_box);
        const int x_offset = std::get<0>(props);
        const int x_size = std::get<1>(props);
        const int y_offset = std::get<2>(props);
        const int y_size = std::get<3>(props);

        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();
        auto Ez = global_field._Ez.view<memory_space>();
        auto entity_ind = global_field._which_entity.view<memory_space>();

        auto space_step = global_geometry._x.dx;
        auto entity_id = _entity_id;

        auto Psi_Ez_y = _Psi_Ez_y.view<memory_space>();
        _Psi_Ez_y.sync<memory_space>();
        _Psi_Ez_y.modify<memory_space>();

        auto Psi_Hx_y = _Psi_Hx_y.view<memory_space>();
        _Psi_Hx_y.sync<memory_space>();
        _Psi_Hx_y.modify<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });

        auto N_box = _box._x._N;
        auto nx = global_geometry._x._N;
        auto ny = global_geometry._y._N;

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int imo = getIndex(iinit + x_offset - 1, nx);
                const int ipo = getIndex(iinit + x_offset + 1, nx);

                const int j = getIndex(jinit + y_offset, ny);
                const int jmo = getIndex(jinit + y_offset - 1, ny);
                const int jpo = getIndex(jinit + y_offset + 1, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                double sigma = getSigma(j_inside, x_size);
                double alpha = getAlpha(j_inside, x_size);
                double kappa = getKappa(j_inside, x_size);

                auto b_y = time_step * Kokkos::exp(sigma / kappa + alpha);
                auto c_y = sigma * ((b_y - 1) / (sigma + kappa * alpha) / kappa);

                Psi_Ez_y(i_inside, j_inside) =
                    b_y * Psi_Ez_y(i_inside, j_inside) + c_y * (Hx(i, jmo) - Hx(i, j)) / space_step;

                Psi_Hx_y(i_inside, j_inside) =
                    b_y * Psi_Hx_y(i_inside, j_inside) + c_y * (Ez(i, jpo) - Ez(i, j)) / space_step;
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

        auto Psi_Ez_y = _Psi_Ez_y.view<memory_space>();
        _Psi_Ez_y.sync<memory_space>();

        auto Psi_Hx_y = _Psi_Hx_y.view<memory_space>();
        _Psi_Hx_y.sync<memory_space>();

        auto policy = DevicePolicy2D({ 0, 0 }, { x_size, y_size });
        auto nx = global_geometry._x._N;
        auto ny = global_geometry._y._N;

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int ipo = getIndex(iinit + x_offset + 1, nx);
                const int j = getIndex(jinit + y_offset, ny);
                const int jpo = getIndex(jinit + y_offset + 1, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                auto epsilon { 1.0 };
                auto mu { 1.0 };

                double sigma = getSigma(j_inside, x_size);
                double alpha = getAlpha(j_inside, x_size);
                double kappa = getKappa(j_inside, x_size);

                auto sigma_factor { (sigma * time_step) / (2.0 * mu) };
                auto D_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto D_b { time_step / mu / (1.0 + sigma_factor) };

                Hx(i, j) = D_a * Hx(i, j) - D_b * (Ez(i, jpo) - Ez(i, j)) / (kappa * space_step)
                           + D_b * Psi_Hx_y(i_inside, j_inside);
                Hy(i, j) = D_a * Hy(i, j) + D_b * (Ez(ipo, j) - Ez(i, j)) / (kappa * space_step);
            });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                const int i = getIndex(iinit + x_offset, nx);
                const int imo = getIndex(iinit + x_offset - 1, nx);
                const int j = getIndex(jinit + y_offset, ny);
                const int jmo = getIndex(jinit + y_offset - 1, ny);

                const int i_inside = getIndex(iinit, x_size);
                const int j_inside = getIndex(jinit, y_size);

                auto epsilon { 1.0 };

                double sigma = getSigma(j_inside, x_size);
                double alpha = getAlpha(j_inside, x_size);
                double kappa = getKappa(j_inside, x_size);

                auto sigma_factor { (sigma * time_step) / (2.0 * epsilon) };
                auto C_a { (1.0 - sigma_factor) / (1.0 + sigma_factor) };
                auto C_b { time_step / epsilon / (1.0 + sigma_factor) };

                Ez(i, j) =
                    C_a * Ez(i, j)
                    + C_b
                          * ((Hy(i, j) - Hy(imo, j)) / (kappa * space_step)
                             - (Hx(i, j) - Hx(i, jmo)) / (kappa * space_step) - Psi_Ez_y(i_inside, j_inside));
            });
    }

  private:
    KOKKOS_INLINE_FUNCTION
    double getProperty(int ind, int N, double max, double min)
    {
        return min
               + (max - min) * Kokkos::abs(static_cast<double>(ind - N / 2.0)) / static_cast<double>(N / 2.0);
    }

    KOKKOS_INLINE_FUNCTION
    double getSigma(int ind, int N) { return getProperty(ind, N, 2.5, 0.1); }
    KOKKOS_INLINE_FUNCTION
    double getKappa(int ind, int N) { return getProperty(ind, N, 1.0, 2.0); }
    KOKKOS_INLINE_FUNCTION
    double getAlpha(int ind, int N) { return getProperty(ind, N, 1.0, 1.0); }

    KOKKOS_INLINE_FUNCTION
    int getIndex(int ind, int N)
    {
        if (ind == -1)
            return N - 1;
        if (ind == N)
            return 0;
        return ind;
    };
};

} // namespace Prop
