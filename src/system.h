#pragma once

#include "field.h"
#include "geometry.h"
#include "types.h"
namespace Prop
{
class System2D
{
    using GridData = GridData2D;
    using External_data = External2D_data;
    using Components = Components2DTM;

  public:
    System2D(Axis x, Axis y): _geometry(x, y), _field(x._N, y._N)
    {
        _space_step = x.dx; // lambda_characteristic / resolution
        // x_step *0.5 / Const_c;
        //      _step_size_factor = 0.5; //_time_step * Const_c / _x_step;
        _resolution = 1.0; // lambda_characteristic / _x_step
        spdlog::info("[System] created with size : {:d} {:d}", x._N, y._N);
        if (!Kokkos::is_initialized())
        {
            spdlog::debug("[TwoDimensionalSystem] Initializing Kokkos...");
            Kokkos::initialize();
        }

    };

    External_data& getExternal(Components comp) { return _field.getExternal(comp); }

    void setExternal(Components comp) { _field.setExternal(comp); }

    SimplePolicy2D getPolicy() { return _field.getPolicy(); }
    void addBlock(Block2D& block) {
        _geometry.addBlock(block);
    };
    void propagateCustom(double);
    void propagateFixedTime(double);
    void propagate() { propagateCustom(_courant * Const_c / std::sqrt(2.0 / _space_step / _space_step)); }
    GridData& getEz() { return _field._Ez; };
    GridData& getHx() { return _field._Hx; };
    GridData& getHy() { return _field._Hy; };

  private:
    bool _firstTime {true};
    double _time { 0.0 };
    double _courant { 0.5 };
    double _resolution = std::numeric_limits<double>::signaling_NaN();
    //    double _step_size_factor;
    double _space_step = std::numeric_limits<double>::signaling_NaN();

    Geometry2D _geometry;

    Grid2DRectangular _field;
};
} // namespace Prop
