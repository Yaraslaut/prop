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
#include "types.h"

#include <memory>
#include <tuple>
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
    double dx = std::numeric_limits<double>::signaling_NaN(); // TODO
    AxisUnits(): _min(0.0), _max(0.0) {};
    AxisUnits(double min, double max):
        _min(l_unit(min * Const_scaling_factor)), _max(l_unit(max * Const_scaling_factor)) {};

    AxisUnits(double min, double max, index N):
        _min(l_unit(min * Const_scaling_factor)), _max(l_unit(max * Const_scaling_factor)), _N(N)
    {
        dx = (_max - _min) / static_cast<double>(N);
    };

    void calcN(double space_step)
    {
        dx = space_step;
        _N = static_cast<int>((_max - _min) / space_step);
    }

    KOKKOS_INLINE_FUNCTION
    l_unit getCoord(index i) { return _min + l_unit(dx * i); }
    index getIndex(l_unit x) { return static_cast<index>((x - _min) / dx); }
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

std::tuple<int,int> getOffsetAndSize(Axis big, Axis small)
{
    int offset = big.getIndex(small._min);
    int size = big.getIndex(small._max) - offset;
    return std::make_tuple(offset,size);
}

struct Box
{
    Axis _x;
    Axis _y;
    Point2D _center;

    //Box(Axis x, Axis y, Point2D c): _x(x), _y(y), _center(c) {};
    Box(Axis x, Axis y) : _x(x) , _y(y), _center(0.0,0.0){};
};

struct GeometryInBox
{
    Box _box;
    GeometryInBox(Box box, double sp): _box(box) {};
};

class Geometry2D
{
    using GridData = GridData2D_host;

  public:
    Geometry2D(): _x(), _y() {};
    Geometry2D(Axis x, Axis y): _x(x), _y(y) {};

    void fillGeometryFromEntity() {}

    std::tuple<int, int, int, int> getProperties(Box box) {

        auto xProps = getOffsetAndSize(this->_x, box._x);
        int x_offset = std::get<0>(xProps);
        int x_size = std::get<1>(xProps);

        auto yProps = getOffsetAndSize(this->_y, box._y);
        int y_offset = std::get<0>(yProps);
        int y_size = std::get<1>(yProps);

        return std::make_tuple( x_offset, x_size, y_offset, y_size );
    }
    // double get_step() { return std::min(_x.dx, _y.dx); };
    Axis _x;
    Axis _y;
};

} // namespace Prop
