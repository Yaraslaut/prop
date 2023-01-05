#pragma once
#include <sys/types.h>

#include <Kokkos_Core.hpp>
#include <Eigen/Dense>
#include <units/isq/si/length.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <units/isq/si/length.h>


namespace Prop
{
  constexpr double Const_c = 1.0;
  constexpr double Const_epsilon = 1.0;
  using index = int;//uint_fast16_t;

  using basic_length_unit = units::isq::si::micrometre;
  using l_unit = units::isq::si::length<basic_length_unit>;


// 3D aray for the field component in the space
  using Complex = Kokkos::complex<double>;
  using Field_data_type = double;
  using GridData2D =
      Kokkos::View<Field_data_type**>;

  using SimplePolicy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
  using SimplePolicy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

  using External_data_type = Field_data_type;
  using External3D_data    = Eigen::Tensor<External_data_type, 3>;
  using External2D_data    = Eigen::Tensor<External_data_type, 2>;
  //using exec_space = typename Data::traits::execution_space;

  using Vec = Eigen::Vector3d;
}; // namespace Prop
