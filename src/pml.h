#pragma once

#include "Kokkos_Macros.hpp"
#include "types.h"
#include "geometry.h"
#include "field.h"


namespace Prop
{

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

    KOKKOS_INLINE_FUNCTION
    double getSigma(const int i)
    {
        double normilized_dist_to_center_square =
            Kokkos::pow(Kokkos::abs(i - 12.5), 2.0) / Kokkos::pow(Kokkos::abs(12.5), 2.0);
        double factor_to_zero = 1.0 / (1.0 + normilized_dist_to_center_square) - 0.5; // TODO
        int m { 2 };
        double cpmp_factor { 2.0};
        double cpml_order = 4.0;
        return cpmp_factor * Kokkos::pow(normilized_dist_to_center_square , cpml_order);
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

        auto space_step = _box._x.dx;
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
                int imo = getIndex(iinit - 1, nx);

                int j = getIndex(jinit, ny);

                auto scaling = [](int index, double ampl, double factor) {
                    auto displ = Kokkos::abs((index - 12.5) / 12.5);

                    return ampl * Kokkos::pow(displ, factor);
                };

                double normilized_dist_to_center_square =
                    Kokkos::pow(Kokkos::abs(i - 12.5), 2.0) / Kokkos::pow(Kokkos::abs(12.5), 2.0);
                double factor_to_zero = 1.0 / (1.0 + normilized_dist_to_center_square) - 0.5; // TODO
                int m { 2 };
                double cpmp_factor { 1.0};
                double cpml_order = 4.0;
                double sigma_x = - cpmp_factor * Kokkos::pow(factor_to_zero , cpml_order);
                double k_x = 1 + factor_to_zero * (cpmp_factor - 1);
                double a_x = 0.0;//sigma_x * time_step / (2 * k_x );

                auto b_d_y_x = Kokkos::exp((sigma_x / k_x + a_x) * time_step );
                auto b_d_z_x = Kokkos::exp((sigma_x / k_x + a_x) * time_step );
                auto c_b_y_x = sigma_x / (sigma_x * k_x + a_x * k_x * k_x) * (b_d_y_x - 1);
                auto c_d_z_x = sigma_x / (sigma_x * k_x + a_x * k_x * k_x) * (b_d_z_x - 1);

                auto d_E_z_x = (Ez(i, j) - Ez(imo, j)) / space_step;
                auto d_H_y_x = (Hy(i, j) - Hy(imo, j)) / space_step;

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
        auto space_step = _box._x.dx;
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
                auto sigma = 1.0;
                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * space_step) };

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
                auto sigma = 1.0;
                auto epsilon { 1.0 };
                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_eze { (1.0 - (sigma * time_step) / (2.0 * epsilon))
                             / (1.0 + (sigma * time_step) / (2.0 * epsilon)) };

                Ez(i, j) = Ez(i, j) + C_eze * dev_Phi_Dz_x(i, j);
            });

        global_field._Ez.modify<memory_space>();
    }

};


} // namespace Prop
