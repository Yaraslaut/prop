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

#include "Kokkos_Core_fwd.hpp"
#include "field.h"
#include "functors.h"
#include "geometry.h"
#include "sources.h"
#include "types.h"

#include <functional>
#include <limits>
#include <memory>
namespace Prop
{

struct PointSource
{

    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;

    int _x_coord;
    int _y_coord;
    double _freq;
    PointSource(int x, int y, double freq): _x_coord(x), _y_coord(y), _freq(freq) {};
    void Propagator(double time, double time_step, Grid2DRectangular global_field)
    {
        auto Ez = global_field._Ez.view<memory_space>();
        global_field._Ez.sync<memory_space>();
        Ez(_x_coord, _y_coord) += Kokkos::cos(_freq * time) * time_step;
        global_field._Ez.modify<memory_space>();
    };
};

// https://empossible.net/wp-content/uploads/2020/09/Lecture-CPML.pdf
struct PMLregion
{
    using view_type = GridData2D_dual;
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    typedef typename std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                      view_type::memory_space,
                                      view_type::host_mirror_space>::type memory_space;
    using GridData = view_type;
    GridData _Phi_Dz_x;
    GridData _Phi_Dz_y;
    GridData _Phi_By_x;
    Box _box;

    PMLregion(Box box): _box(box)
    {
        auto nx = box._x._N;
        auto ny = box._y._N;
        _Phi_Dz_x = GridData("[PML] Phi Ez x", nx, ny);
        _Phi_By_x = GridData("[PML] Phi Hy x", nx, ny);
    }

    void updatePML(double time, double time_step, Grid2DRectangular global_field)
    {

        auto dev_Phi_By_x = _Phi_By_x.view<memory_space>();

        auto dev_Phi_Dz_x = _Phi_Dz_x.view<memory_space>();

        auto Ez = global_field._Ez.view<memory_space>();
        auto Hx = global_field._Hx.view<memory_space>();
        auto Hy = global_field._Hy.view<memory_space>();

        auto nx = _box._x._N;
        auto ny = _box._y._N;
        auto policy = DevicePolicy2D({ 0, 0 }, { nx, ny });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                auto getIndex = [](int ind, int N) {
                    if (ind == -1)
                        return N - 1;
                    if (ind == N)
                        return 0;
                    return ind;
                };

                int i = getIndex(iinit, nx);
                int ipo = getIndex(iinit + 1, nx);

                int j = getIndex(jinit, ny);

                auto scaling = [](int index, double ampl, double factor) {
                    auto displ = Kokkos::abs((index - 12.5) / 12.5);

                    return ampl * Kokkos::pow(displ, factor);
                };

                double normilized_dist_to_center_square =
                    Kokkos::pow(Kokkos::abs(i - 12.5), 2.0) / Kokkos::pow(Kokkos::abs(12.5), 2.0);
                double factor_to_zero = 1.0 / (1.0 + normilized_dist_to_center_square) - 0.5; // TODO
                int m { 2 };
                double sigma_max = 0.8 * (m + 1) / 376.0;
                double sigma_x = 0.1 + sigma_max * factor_to_zero;
                double k_x = 1.0 + (1.0 - 1.0) * factor_to_zero;
                double a_x = 0.0 * normilized_dist_to_center_square;

                auto b_d_y_x = Kokkos::exp((sigma_x / k_x + a_x) * time_step);
                auto b_d_z_x = Kokkos::exp((sigma_x / k_x + a_x) * time_step);

                auto c_b_y_x = sigma_x / (sigma_x * k_x + a_x * k_x * k_x) * (b_d_y_x - 1);
                auto c_d_z_x = sigma_x / (sigma_x * k_x + a_x * k_x * k_x) * (b_d_z_x - 1);

                auto d_E_z_x = (Ez(ipo, j) - Ez(i, j)) / _box._x.dx;
                auto d_H_y_x = (Hy(ipo, j) - Hy(i, j)) / _box._x.dx;

                dev_Phi_By_x(i, j) = b_d_y_x * dev_Phi_By_x(i, j) + c_b_y_x * d_E_z_x;

                dev_Phi_Dz_x(i, j) = b_d_z_x * dev_Phi_Dz_x(i, j) + c_d_z_x * d_H_y_x;
            });
    }

    void updateMagneticField(double time, double time_step, Grid2DRectangular global_field)
    {

        auto dev_Phi_By_x = _Phi_By_x.view<memory_space>();

        auto Hy = global_field._Hy.view<memory_space>();

        auto nx = _box._x._N;
        auto ny = _box._y._N;
        auto policy = DevicePolicy2D({ 0, 0 }, { nx, ny });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                auto getIndex = [](int ind, int N) {
                    if (ind == -1)
                        return N - 1;
                    if (ind == N)
                        return 0;
                    return ind;
                };

                int i = getIndex(iinit, nx);
                int j = getIndex(jinit, ny);

                auto mu { 1.0 };
                double normilized_dist_to_center_square =
                    Kokkos::pow(Kokkos::abs(i - 12.5), 2.0) / Kokkos::pow(Kokkos::abs(12.5), 2.0);
                double factor_to_zero = 1.0 / (1.0 + normilized_dist_to_center_square) - 0.5; // TODO
                int m { 2 };
                double sigma_max = 0.8 * (m + 1) / 376.0;
                double sigma_x = 0.1 + sigma_max * factor_to_zero;
                double sigma = sigma_x;

                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * _box._x.dx) };

                Hy(i, j) = Hy(i, j) + C_hxe * dev_Phi_By_x(i, j);
            });

        global_field._Hx.modify<memory_space>();
        global_field._Hy.modify<memory_space>();
    }

    void updateElectricField(double time, double time_step, Grid2DRectangular global_field)
    {

        auto dev_Phi_Dz_x = _Phi_Dz_x.view<memory_space>();
        auto Ez = global_field._Ez.view<memory_space>();

        auto nx = _box._x._N;
        auto ny = _box._y._N;
        auto policy = DevicePolicy2D({ 0, 0 }, { nx, ny });

        Kokkos::parallel_for(
            policy, KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                auto getIndex = [](int ind, int N) {
                    if (ind == -1)
                        return N - 1;
                    if (ind == N)
                        return 0;
                    return ind;
                };

                int i = getIndex(iinit, nx);
                int j = getIndex(jinit, ny);

                auto mu { 1.0 };

                double normilized_dist_to_center_square =
                    Kokkos::pow(Kokkos::abs(i - 12.5), 2.0) / Kokkos::pow(Kokkos::abs(12.5), 2.0);
                double factor_to_zero = 1.0 / (1.0 + normilized_dist_to_center_square) - 0.5; // TODO
                int m { 2 };
                double sigma_max = 0.8 * (m + 1) / 376.0;
                double sigma_x = 0.1 + sigma_max * factor_to_zero;
                double sigma = sigma_x;

                auto epsilon { 1.0 };
                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_eze { (1.0 - (sigma * time_step) / (2.0 * epsilon))
                             / (1.0 + (sigma * time_step) / (2.0 * epsilon)) };

                Ez(i, j) = Ez(i, j) + C_eze * dev_Phi_Dz_x(i, j);
            });

        global_field._Ez.modify<memory_space>();
    }
};

class System2D
{
    using GridData = GridData2D_dual;
    using External_data = External2D_data;
    using Components = Components2DTM;

  public:
    System2D(Axis x, Axis y)
    {
        double fmax { Const_c / Const_scaling_factor }; // TODO
        double pts_per_wavelength { 5.0 };
        _resolution = pts_per_wavelength;
        _space_step = Const_c / (pts_per_wavelength * fmax); // lambda_characteristic / resolution
        std::cout << _space_step << std::endl;
        double factor { Const_c / Const_standard_courant_factor };
        _stable_time_step = _space_step / Const_standard_courant_factor;

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
        }

        x.calcN(_space_step);
        y.calcN(_space_step);
        _geometry = Geometry2D(x, y);
        _field = Grid2DRectangular(x._N, y._N);

        auto freq = 2.0 * Kokkos::numbers::pi / 1e-6;
        PointSource point_source(60, 60, freq);
        _entities_point_source.push_back(std::make_unique<PointSource>(point_source));

        Axis axis_x { 0.0, 0.1, 10 };
        Axis axis_y { -1.0, 1.0, 100 };
        Box box(axis_x, axis_y, Point2D(0.5, 0.5));
        PMLregion pml(box);
        _entities_pml_region.push_back(std::make_unique<PMLregion>(pml));
    };

    const External_data& getExternal(Components comp) { return _field.getExternal(comp); }

    SimplePolicy2D getPolicy() { return _field.getPolicy(); }
    void addBlock(Block2D& block) { _geometry.addBlock(block); };
    void addSourceEz(PlaneWave2D& pw) { _sources_Ez.push_back(std::make_unique<PlaneWave2D>(pw)); }

    void propagateCustom(double total_time);
    void propagateFixedTime(double time_step);
    void propagate() { propagateCustom(_stable_time_step); }

    int getNx() { return static_cast<int>(_geometry._x._N); }
    int getNy() { return static_cast<int>(_geometry._y._N); }

  private:
    bool _first_time { true };
    double _time { 0.0 };
    double _stable_time_step = std::numeric_limits<double>::signaling_NaN();
    double _resolution = std::numeric_limits<double>::signaling_NaN();
    //    double _step_size_factor;
    double _space_step = std::numeric_limits<double>::signaling_NaN();

    Geometry2D _geometry;

    std::vector<std::unique_ptr<PlaneWave2D>> _sources_Ez;

    std::vector<std::unique_ptr<PointSource>> _entities_point_source;
    std::vector<std::unique_ptr<PMLregion>> _entities_pml_region;
    Grid2DRectangular _field;
};

} // namespace Prop
