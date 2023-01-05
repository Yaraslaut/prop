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

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <boxed-cpp/boxed.hpp>
#include <sys/types.h>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef USE_SPDLOG
    #include <spdlog/spdlog.h>
#endif

namespace Prop
{

constexpr double Const_epsilon0 = 8.85418782 * 1e-12;
constexpr double Const_mu0 = 4.0 * Kokkos::numbers::pi * 1e-7;
constexpr double Const_c = 1.0;                       // 299792457.95971;
constexpr double Const_standard_courant_factor = 1.0; // 84853;
constexpr double Const_scaling_factor = 1.0;          // 1e-6;         // TODO

using index = int;

using basic_length_unit = double;
using l_unit = double;

using Complex = Kokkos::complex<double>;
using Field_data_type = double;

using GridData2D_host = Kokkos::View<Field_data_type**>;
using GridData2D_dual = Kokkos::DualView<Field_data_type**>;
using GridData2DIndex_dual = Kokkos::DualView<index**>;

using DevicePolicy1D = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>;
using SimplePolicy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultHostExecutionSpace>;
using DevicePolicy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>;

using External_data_type = Field_data_type;

using External2D_data = Eigen::Matrix<External_data_type, Eigen::Dynamic, Eigen::Dynamic>;

template <class ExecutionSpace>
struct Types
{
    using view_type = GridData2D_dual;

    using memory_space = std::conditional<std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace>::value,
                                          view_type::t_dev::memory_space,
                                          view_type::t_host::memory_space>::type;

    using FieldViewType = Kokkos::View<Field_data_type**, memory_space>;
    using IndexViewType = Kokkos::View<index**, memory_space>;
};

using Vec = Eigen::Vector3d;
namespace tags
{
    struct Conductivity
    {
    };
} // namespace tags
using Conductivity = boxed::boxed<double, tags::Conductivity>;
} // namespace Prop
