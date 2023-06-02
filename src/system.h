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
#include "functors.h"
#include "geometry.h"
#include "sources.h"
#include "types.h"

#include <functional>
#include <limits>
#include <memory>
namespace Prop
{
class System2D
{
    using GridData = GridData2D_host;
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
    Grid2DRectangular _field;
};

template <class ExecutionSpace, typename ExecutionPolicy>
struct Entity
{
    typedef ExecutionSpace _execution_space;
    typedef ExecutionPolicy _execution_policy;

    virtual void propagateFixedTime(double);

    bool _first_time { true };
    GeometryInBox _geometry;
    GridSubView _local_field;
    std::function<void(double)> _local_propagator;
    Entity(Box box, double space_step): _geometry(box, space_step) {};
    virtual ~Entity();
};

template <class ExecutionSpace, class ExecutionPolicy>
struct FreeSpaceEntity: public Entity<ExecutionSpace, ExecutionPolicy>
{
    using BaseEntity = Entity<ExecutionSpace, ExecutionPolicy>;
    FreeSpaceEntity(Box box): BaseEntity(box)
    {
        this->_local_propagator = [this](double time_step) {
            Kokkos::parallel_for(BaseEntity::_execution_policy(),
                                 updateMagneticFieldFreeSpace<typename BaseEntity::_execution_space>(
                                     this->_local_field._Ez,
                                     this->_local_field._Hx,
                                     this->_field._Hy,
                                     time_step,
                                     this->_geometry->_space_step));
            Kokkos::parallel_for(BaseEntity::_execution_policy(),
                                 updateElectricFieldFreeSpace<typename BaseEntity::_execution_space>(
                                     this->_local_field._Ez,
                                     this->_local_field._Hx,
                                     this->_field._Hy,
                                     time_step,
                                     this->_geometry->_space_step));

            Kokkos::fence();
        };
    };
};

template <class ExecutionSpace, class ExecutionPolicy>
struct PointSourceEntity: public Entity<ExecutionSpace, ExecutionPolicy>
{
    using BaseEntity = Entity<ExecutionSpace, ExecutionPolicy>;
    double freq;
    PointSourceEntity(Box box): BaseEntity(box)
    {

        this->_local_propagator = [this](double time_step) {
            Kokkos::parallel_for(
                BaseEntity::_execution_policy(),
                updateFromPlaneWave<typename BaseEntity::_execution_space>(this->_local_field._Ez,
                                                                           this->_local_field._Hx,
                                                                           this->_field._Hy,
                                                                           time_step,
                                                                           this->_geometry->_space_step));
        };
    };
};

} // namespace Prop
