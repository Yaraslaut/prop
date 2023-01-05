#include "system.h"

#include "geometry.h"

// #include "Kokkos_Complex.hpp"
#include <cmath>
void Prop::System2D::propagateCustom(double totalTime)
{
    double max_time_step = _courant / Const_c / std::sqrt(2.0 / _space_step / _space_step);
    spdlog::info("max_time_step is {:f}", max_time_step);
    if (totalTime < max_time_step)
        return propagateFixedTime(totalTime);
    double accumulated_time_step { 0.0 };
    while (accumulated_time_step < totalTime)
    {
        accumulated_time_step += max_time_step;
        propagateFixedTime(max_time_step);
    }
}
void Prop::System2D::propagateFixedTime(double time_step)
{

    auto courant { Const_c * time_step * std::sqrt(2.0 / _space_step / _space_step) };
    spdlog::debug("[System] Courant factor is   : {:f} ", courant);
    spdlog::debug("[System] Propagate. Current time is  : {:f} ", _time);

    auto update_from_source = [&]() {
        Kokkos::parallel_for(
            "update electric field", _field.getPolicy(), KOKKOS_LAMBDA(int i, int j) {
                auto x = _geometry._x.getCoord(i);
                auto y = _geometry._y.getCoord(j);
                for (auto& sourceEz: _sources_Ez)
                {
                    _field._Ez(i, j) += sourceEz->getField(_time, Point2D { x, y }) * time_step;
                }
            });
    };

    auto update_magnetic_field = [&]() {
        Kokkos::parallel_for(
            "update magnetic field", _field.getPolicy(), KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                auto ipo = _geometry._x.getIndex(iinit + 1);
                auto i = _geometry._x.getIndex(iinit);
                auto imo = _geometry._x.getIndex(iinit - 1);

                auto jpo = _geometry._y.getIndex(jinit + 1);
                auto j = _geometry._y.getIndex(jinit);
                auto jmo = _geometry._y.getIndex(jinit - 1);

                auto mu { _geometry.getMu(i, j) };
                auto sigma { _geometry.getSigma(i, j) };

                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_hxh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hxe { one_over_one_plus_sigma_mu * time_step / (mu * _space_step) };
                auto C_hyh { one_over_one_plus_sigma_mu * (1.0 - (sigma * time_step) / (2.0 * mu)) };
                auto C_hye { one_over_one_plus_sigma_mu * time_step / (mu * _space_step) };
                _field._Hx(i, j) = C_hxh * _field._Hx(i, j) - C_hxe * (_field._Ez(i, jpo) - _field._Ez(i, j));
                _field._Hy(i, j) = C_hyh * _field._Hy(i, j) + C_hye * (_field._Ez(ipo, j) - _field._Ez(i, j));
            });
    };

    auto update_electric_field = [&]() {
        Kokkos::parallel_for(
            "update electric field", _field.getPolicy(), KOKKOS_LAMBDA(const int& iinit, const int& jinit) {
                auto i = _geometry._x.getIndex(iinit);
                auto imo = _geometry._x.getIndex(iinit - 1);

                auto j = _geometry._y.getIndex(jinit);
                auto jmo = _geometry._y.getIndex(jinit - 1);

                auto mu { _geometry.getMu(i, j) };
                auto sigma { _geometry.getSigma(i, j) };
                auto epsilon { _geometry.getEpsilon(i, j) };
                auto one_over_one_plus_sigma_mu { 1.0 / (1.0 + (sigma * time_step) / (2.0 * mu)) };
                auto C_eze { (1.0 - (sigma * time_step) / (2.0 * epsilon))
                             / (1.0 + (sigma * time_step) / (2.0 * epsilon)) };
                auto C_ezhx { time_step / (1.0 + (sigma * time_step) / (2.0 * epsilon))
                              / (epsilon * _space_step) };
                auto C_ezhy { time_step / (1.0 + (sigma * time_step) / (2.0 * epsilon))
                              / (epsilon * _space_step) };

                _field._Ez(i, j) = C_eze * _field._Ez(i, j) + C_ezhx * (_field._Hy(i, j) - _field._Hy(imo, j))
                                   - C_ezhy * (_field._Hx(i, j) - _field._Hx(i, jmo));
            });
    };

    if (_firstTime)
    {
        _firstTime = false;
        this->_geometry.fillGeometryFromEntity();
    };

    update_from_source();
    update_magnetic_field();
    update_electric_field();

    _time += time_step;
    spdlog::info("time is {:f}", _time);
}
