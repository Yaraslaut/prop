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

#include "types.h"

#include <KokkosExp_InterOp.hpp>
#include <cstdint>

namespace Prop
{

enum class Components2DTM
{
    Ez,
    Hx,
    Hy
};

struct Grid2DRectangular
{
    using GridData = GridData2D_dual;
    using External_data = External2D_data;
    using Components = Components2DTM;

    Grid2DRectangular() {};
    Grid2DRectangular(index nx, index ny): _nx(nx), _ny(ny)
    {
        // electric field
        _Ez = GridData("field_Ez", _nx, _ny);
        // magnetic field
        _Hx = GridData("field_Hx", _nx, _ny);
        _Hy = GridData("field_Hy", _nx, _ny);

        _external = External_data(_nx, _ny);

        _policy = SimplePolicy2D({ 0, 0 }, { _nx, _ny });
        _device_policy = DevicePolicy2D({ 0, 0 }, { _nx, _ny });

        auto host_Ez = _Ez.h_view;
        auto host_Hx = _Hx.h_view;
        auto host_Hy = _Hy.h_view;

        for (int i = 0; i < _nx; i++)
        {
            for (int j = 0; j < _ny; j++)
            {
                host_Ez(i, j) = 0.0;
                host_Hx(i, j) = 0.0;
                host_Hy(i, j) = 0.0;
            }
        }
    };

    const External_data& getExternal(Components comp)
    {
        auto& internal = _Ez;
        switch (comp)
        {
            case Components::Ez: internal = _Ez; break;
            case Components::Hx: internal = _Hx; break;
            case Components::Hy: internal = _Hy; break;
        }

        _external = External_data(_nx, _ny);

        internal.sync<GridData::host_mirror_space>();
        auto host_internal = internal.template view<typename GridData2D_dual::host_mirror_space>();
        for (int i = 0; i < _nx; i++)
            for (int j = 0; j < _ny; j++)
                _external(i, j) = host_internal(i, j);

        return _external;
    };

    SimplePolicy2D getPolicy() { return _policy; }
    DevicePolicy2D getDevicePolicy() { return _device_policy; }

    External_data _external;

    GridData _Ez;
    GridData _Hx;
    GridData _Hy;
    SimplePolicy2D _policy;
    DevicePolicy2D _device_policy;
    index _nx;
    index _ny;
};

struct GridSubView
{
    using GridData = GridData2D_dual;
    using External_data = External2D_data;
    using Components = Components2DTM;
    using Span = Kokkos::pair<int, int>;

    using SubviewType = Kokkos::Subview<GridData2D_dual, std::pair<int, int>, std::pair<int,int>>;

    GridSubView() = default;
    GridSubView(Span x, Span y)
    {
        _policy = SimplePolicy2D({ 0, 0 }, { x.second - x.first, y.second - y.first });
        _device_policy = DevicePolicy2D({ 0, 0 }, { x.second - x.first, y.second - y.first });
    };

    SubviewType _Ez;
    SubviewType _Hx;
    SubviewType _Hy;
    SimplePolicy2D _policy;
    DevicePolicy2D _device_policy;
};

} // namespace Prop
