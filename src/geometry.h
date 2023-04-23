#pragma once
#include "medium.h"
#include "types.h"
#include "units/magnitude.h"

#include <concepts>
#include <memory>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace Prop
{

/*
** Axis in one dimension
** does not have any orientation
** and defines length via _min and _max coordinates
** with N points in this range
*/
template <typename T>
struct AxisUnits
{
    l_unit _min;
    l_unit _max;
    index _N = 1;
    double dx = 0.;
    AxisUnits(): _min(0.0), _max(0.0) {};
    AxisUnits(auto min, auto max, index n): _min(l_unit(min)), _max(l_unit(max)), _N(n)
    {
        dx = (_max - _min).number() / static_cast<double>(_N);
    };
    AxisUnits(auto min, auto max): _min(l_unit(min)), _max(l_unit(max))
    {
        dx = (_max - _min).number() / static_cast<double>(_N);
    };
    l_unit getCoord(index i) { return _min + l_unit(dx * i); }
    index getIndex(index i) { return (i + 10) % _N; }
};

using Axis = AxisUnits<basic_length_unit>;

/*
** bounding box of entity for 3D space
*/
template <typename T>
struct DimensionsUnits3D
{
    l_unit _xdim;
    l_unit _ydim;
    l_unit _zdim;
    DimensionsUnits3D(auto x, auto y, auto z): _xdim(l_unit(x)), _ydim(l_unit(y)), _zdim(l_unit(z)) {};
};

/*
** bounding box of entity for 2D space
*/
template <typename T>
struct DimensionsUnits2D
{
    l_unit _xdim;
    l_unit _ydim;
    DimensionsUnits2D(): _xdim(0.0), _ydim(0.0) {};
    DimensionsUnits2D(auto x, auto y): _xdim(l_unit(x)), _ydim(l_unit(y)) {};
};

using Dimensions2D = DimensionsUnits2D<basic_length_unit>;

/*
** point in 3d space with
** x,y,z cordinates
** tempalte argument from units::isq::si::
** represents one unit of length
*/
template <typename T>
struct Point3DUnits
{
    l_unit _x;
    l_unit _y;
    l_unit _z;
    Point3DUnits(auto x, auto y, auto z): _x(l_unit(x)), _y(l_unit(y)), _z(l_unit(z)) {};
    Point3DUnits(): _x(0.0), _y(0.0), _z(0.0) {};
};

using Point3D = Point3DUnits<basic_length_unit>;

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
    Point2DUnits(auto x, auto y): _x(l_unit(x)), _y(l_unit(y)) {};
    Point2DUnits(): _x(0.0), _y(0.0) {};

    Point2DUnits<T> operator-(Point2DUnits<T>& other)
    {
        return Point2DUnits<T>(_x - other._x, _y - other._y);
    }

    bool operator<(Point2DUnits<T>& other) { return (_x < other.x) && (_y < other.y); }
    bool operator<(DimensionsUnits2D<T>& other) { return (_x < other._xdim) && (_y < other._ydim); }
};

using Point3D = Point3DUnits<basic_length_unit>;
using Point2D = Point2DUnits<basic_length_unit>;
template <typename T>
concept concept_axis = std::derived_from<T, Axis>;

/*
 * scalar product funciton for 2d vectors
 */
template<typename T>
double dot(Point2DUnits<T>& f, Point2DUnits<T>& s)
{
    return (f._x * s._x + f._y * s._y).number(); // TODO fix units
}


/*
** base class for entity in 2D space
*/
// struct Entity2D
// {
//     Entity2D(): _medium { 0.0, 0.0, 0.0 } {};
//     Entity2D(auto medium): _medium(medium) {};
//     Entity2D(auto eps, auto mu, auto sigma): _medium { eps, mu, sigma } {};

//     virtual IsotropicMedium getCharacteristics(Point2D p) = 0;
//     virtual ~Entity2D();
// };

struct Block2D
{
    Block2D(Point2D cent, Dimensions2D size): _center(cent), _size(size), _medium(0.0, 0.0, 0.0) {};
    Block2D(Point2D cent, Dimensions2D size, IsotropicMedium medium):
        _center(cent), _size(size), _medium(medium) {};
    Block2D(Point2D cent, Dimensions2D size, auto eps, auto mu, auto sigma):
        _center(cent), _size(size), _medium(eps, mu, sigma) {};

    Point2D _center;
    Dimensions2D _size;
    IsotropicMedium _medium;

    bool isInside(Point2D p)
    {
        return (abs((p._x - _center._x).number()) < (_size._xdim).number() * 0.5)
               && (abs((p._y - _center._y).number()) < (_size._ydim).number() * 0.5);
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
    using GridData = GridData2D;

  public:
    Geometry2D(Axis x, Axis y): _x(x), _y(y)
    {
        auto n = _x._N;
        _sigma = GridData("[geometry] sigma", _x._N, _y._N);
        _epsilon = GridData("[geometry] epsilon", _x._N, _y._N);
        _mu = GridData("[geometry] mu", _x._N, _y._N);

        // fill geoemtry with default values
        Kokkos::parallel_for(
            "[geometry] fill with initial values",
            SimplePolicy2D({ 0, 0 }, { _x._N, _y._N }),
            KOKKOS_LAMBDA(const int& i, const int& j) {
                _sigma(i, j) = 0.0;
                _epsilon(i, j) = 1.0;
                _mu(i, j) = 1.0;
            });
    };

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

    void fillGeometryFromEntity()
    {
        spdlog::info("Filling geometry");
        Kokkos::parallel_for(
            "[geometry] fill with initial values",
            SimplePolicy2D({ 0, 0 }, { _x._N, _y._N }),
            KOKKOS_LAMBDA(const int& i, const int& j) {
                auto x = _x.getCoord(i);
                auto y = _y.getCoord(j);
                for (auto& block: _items)
                {
                    auto mat = block->getCharacteristics(Point2D(x, y));
                    spdlog::info("[Geometry] Filling geometry: {} {} {:f} {:f} {:f} ",
                                 i,
                                 j,
                                 mat._sigma,
                                 mat._epsilon,
                                 mat._mu);

                    _sigma(i, j) += mat._sigma;
                    _epsilon(i, j) += mat._epsilon;
                    _mu(i, j) += mat._mu;
                }
            });
    }

    void getPMLFactor(index i ,index j)
    {
        auto x = _x.getCoord(i);
        auto y = _y.getCoord(j);
        double sigma {0.2};
        double aw{0.1};
        double epsilon{1.0};
        double omega{1.0};
        auto c = 1.0 +  sigma / (aw + std::complex<double>(0.0,1.0) * epsilon * omega  );
    }
  private:
    GridData _sigma;
    GridData _epsilon;
    GridData _mu;
};

} // namespace Prop
