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
#include "Kokkos_Macros.hpp"
#include "medium.h"
#include "types.h"

#include <memory>
#include <type_traits>

namespace Prop
{

/*
** main structure for Axis
** used to create basis system
** axis has finite length and
** defined via _min and _max coordinates
** with N points in this range
** template is used to add dimensional units
** to the structure maybe
*/
template <typename T>
struct AxisUnits
{
    l_unit _min;
    l_unit _max;
    index _N = 1;
    double dx = std::numeric_limits<double>::signaling_NaN();
    AxisUnits(): _min(0.0), _max(0.0) {};
    AxisUnits(double min, double max):
        _min(l_unit(min * Const_scaling_factor)), _max(l_unit(max * Const_scaling_factor)) {};

    AxisUnits(double min, double max, index N):
        _min(l_unit(min * Const_scaling_factor)), _max(l_unit(max * Const_scaling_factor)), _N(N)
    {
        dx = (_max - _min) / static_cast<double>(N);
    };

    void calcN(double space_step) { _N = static_cast<int>((_max - _min) / space_step); }

    KOKKOS_INLINE_FUNCTION
    l_unit getCoord(index i) { return _min + l_unit(dx * i); }
    KOKKOS_INLINE_FUNCTION
    index getIndex(index i)
    {
        if (i == -1)
            return _N - 1;
        if (i == _N)
            return 0;
        return i;
    }
};

using Axis = AxisUnits<basic_length_unit>;

/*
** bounding box of entity for 2D space
*/
template <typename T>
struct DimensionsUnits2D
{
    l_unit _xdim;
    l_unit _ydim;
    DimensionsUnits2D(): _xdim(0.0), _ydim(0.0) {};
    DimensionsUnits2D(double x, double y): _xdim(l_unit(x)), _ydim(l_unit(y)) {};
};

using Dimensions2D = DimensionsUnits2D<basic_length_unit>;

/*
** point in 2d space with
** x,y cordinates
** tempalte argument from units::isq::si::
** represents one unit of length
*/
template <typename T>
struct Point2DUnits
{
    l_unit _x;
    l_unit _y;
    Point2DUnits(double x, double y): _x(l_unit(x)), _y(l_unit(y)) {};
    Point2DUnits(): _x(0.0), _y(0.0) {};

    Point2DUnits<T> operator-(Point2DUnits<T>& other)
    {
        return Point2DUnits<T>(_x - other._x, _y - other._y);
    }

    bool operator<(Point2DUnits<T>& other) { return (_x < other.x) && (_y < other.y); }
    bool operator<(DimensionsUnits2D<T>& other) { return (_x < other._xdim) && (_y < other._ydim); }
};

using Point2D = Point2DUnits<basic_length_unit>;

/*
 * scalar product of vectors
 */

template <typename T>
double dot(Point2DUnits<T>& f, Point2DUnits<T>& s)
{
    return f._x * s._x + f._y * s._y;
}

struct Block2D
{
    Block2D(Point2D cent, Dimensions2D size): _center(cent), _size(size), _medium(0.0, 0.0, 0.0) {};
    Block2D(Point2D cent, Dimensions2D size, IsotropicMedium medium):
        _center(cent), _size(size), _medium(medium) {};
    Block2D(Point2D cent, Dimensions2D size, double eps, double mu, double sigma):
        _center(cent), _size(size), _medium(eps, mu, sigma) {};

    Point2D _center;
    Dimensions2D _size;
    IsotropicMedium _medium;

    bool isInside(Point2D p)
    {
        return (abs((p._x - _center._x)) < (_size._xdim) * 0.5)
               && (abs((p._y - _center._y)) < (_size._ydim) * 0.5);
    }
    IsotropicMedium getCharacteristics(Point2D p)
    {
        if (isInside(p))
            return this->_medium;
        return IsotropicMedium(0, 0, 0);
    };
};

// template <typename T>
// struct Block2DTestunits
// {
//     Point2DUnits<T> _center;
//     DimensionsUnits2D<T> _size;
//     IsotropicMedium _medium;
//     Block2DTestunits(auto cent, auto size): _center(cent), _size(size), _medium { 0.0, 0.0, 0.0 } {};
//     Block2DTestunits(auto cent, auto size, auto eps, auto mu, auto sigma):
//         _center(cent), _size(size), _medium { eps, mu, sigma } {};
//     Block2DTestunits(auto cent, auto size, auto med): _center(cent), _size(size), _medium(med) {};

// };

// using Block2D = Block2DTestunits<basic_length_unit>;

class Geometry2D
{
    using GridData = GridData2D_host;

  public:
    Geometry2D(): _x(), _y() {};
    Geometry2D(Axis x, Axis y): _x(x), _y(y)
    {
        auto n = _x._N;
        _sigma = GridData("[geometry] sigma", _x._N, _y._N);
        _epsilon = GridData("[geometry] epsilon", _x._N, _y._N);
        _mu = GridData("[geometry] mu", _x._N, _y._N);

        // fill geoemtry with default values
        for (int i = 0; i < _x._N; i++)
            for (int j = 0; j < _y._N; j++)
            {
                _sigma(i, j) = 0.0;
                _epsilon(i, j) = 1.0;
                _mu(i, j) = 1.0;
            }
    };

    void fillGeometryFromEntity() {}

    double get_step() { return std::min(_x.dx, _y.dx); };
    Axis _x;
    Axis _y;

    void addBlock(Block2D& b)
    {
        _items.push_back(std::make_unique<Block2D>(b));
        ;
    };

    std::vector<std::unique_ptr<Block2D>> _items;

    Field_data_type getSigma(index i, index j) { return _sigma(i, j); };
    Field_data_type getEpsilon(index i, index j) { return _epsilon(i, j); };
    Field_data_type getMu(index i, index j) { return _mu(i, j); };

  private:
    GridData _sigma;
    GridData _epsilon;
    GridData _mu;
};

struct Box
{
    Axis _x;
    Axis _y;
    Point2D _center;

    Box(Axis x, Axis y, Point2D c): _x(x), _y(y), _center(c) {};
};

struct GeometryInBox
{
    Box _box;
    double _space_step;

    GeometryInBox(Box box, double sp): _box(box), _space_step(sp) {};
};

} // namespace Prop
