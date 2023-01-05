#pragma once

#include "types.h"

#include <KokkosExp_InterOp.hpp>
#include <cstdint>
#include <ranges>
#include <spdlog/spdlog.h>

namespace Prop
{

enum class Components3D
{
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz
};

enum class Components2DTM
{
    Ez,
    Hx,
    Hy
};

struct Grid2DRectangular
{
    using GridData = GridData2D;
    using External_data = External2D_data;
    using Components = Components2DTM;

    Grid2DRectangular(index nx, index ny): _nx(nx), _ny(ny)
    {
        // electric field
        _Ez = GridData("field_Ez", _nx, _ny);
        // magnetic field
        _Hx = GridData("field_Hx", _nx, _ny);
        _Hy = GridData("field_Hy", _nx, _ny);

        _external = Eigen::Tensor<External_data_type, 2>(_nx, _ny);
        _policy = SimplePolicy2D({ 0, 0 }, { _nx, _ny });

        Kokkos::parallel_for(
            "Initializing field", _policy, KOKKOS_LAMBDA(const int& i, const int& j) {
                _Ez(i, j) = 0.0;
                _Hx(i, j) = 0.0;
                _Hy(i, j) = 0.0;
            });
    };

    External_data& getExternal(Components comp)
    {
        auto& internal = _Ez;
        switch (comp)
        {
            case Components::Ez: internal = _Ez; break;
            case Components::Hx: internal = _Hx; break;
            case Components::Hy: internal = _Hy; break;
        }

        Kokkos::parallel_for(
            "update external data", _policy, KOKKOS_LAMBDA(const int& i, const int& j) {
                _external(i, j) = internal(i, j);
            });

        return _external;
    };

    void setExternal(Components comp)
    {
        auto& internal = _Ez;
        switch (comp)
        {
            case Components::Ez: internal = _Ez; break;
            case Components::Hx: internal = _Hx; break;
            case Components::Hy: internal = _Hy; break;
        }

        Kokkos::parallel_for(
            "update external data", _policy, KOKKOS_LAMBDA(const int& i, const int& j) {
                internal(i, j) = _external(i, j);
            });
    };

    SimplePolicy2D getPolicy() { return _policy; }
    External_data _external;

    GridData _Ez;
    GridData _Hx;
    GridData _Hy;
    SimplePolicy2D _policy;
    index _nx;
    index _ny;
};

} // namespace Prop
