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

#include "field.h"
#include "geometry.h"
#include "medium.h"
#include "prop.cpp"
#include "sources.h"
#include "system.h"
#include "types.h"

#include <KokkosExp_InterOp.hpp>
#include <Kokkos_Core.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pyprop, m)
{

    py::class_<Prop::Axis>(m, "Axis").def(py::init<double, double>());
    py::class_<Prop::Point2D>(m, "Point2D").def(py::init<double, double>());
    py::class_<Prop::Dimensions2D>(m, "Dimensions2D").def(py::init<double, double>());
    py::class_<Prop::PlaneWave>(m, "PlaneWave")
        .def(py::init<double, double, Prop::Point2D, Prop::Point2D>());
    py::class_<Prop::PointSource>(m, "PointSource")
        .def(py::init<double, double, Prop::Point2D>());

    py::enum_<Prop::Components2DTM>(m, "Component2D")
        .value("Ez", Prop::Components2DTM::Ez)
        .value("Hx", Prop::Components2DTM::Hx)
        .value("Hy", Prop::Components2DTM::Hy)
        .export_values();

    py::class_<Prop::IsotropicMedium>(m, "Block_IsotropicMedium").def(py::init<Prop::Axis,Prop::Axis,double, double, double>());
    py::class_<Prop::PMLRegion>(m, "Block_PMLRegion").def(py::init<Prop::Axis, Prop::Axis>());

    py::class_<Prop::System2D>(m, "System2D")
        .def(py::init<Prop::Axis, Prop::Axis>())
        .def(py::init<Prop::Axis, Prop::Axis, int>())
        .def("get", &Prop::System2D::getExternal, py::return_value_policy::reference_internal)
        .def("propagate", &Prop::System2D::propagate)
        .def("propagate", &Prop::System2D::propagateCustom)
        .def("addBlock", py::overload_cast<Prop::IsotropicMedium&>(&Prop::System2D::addBlock))
        .def("addBlock", py::overload_cast<Prop::PMLRegion&>(&Prop::System2D::addBlock))
        .def("nx", &Prop::System2D::getNx)
        .def("ny", &Prop::System2D::getNy)
        .def("addSourceEz", py::overload_cast<Prop::PlaneWave&>(&Prop::System2D::addSourceEz))
        .def("addSourceEz", py::overload_cast<Prop::PointSource&>(&Prop::System2D::addSourceEz));

    m.def("initialize", []() {
        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
        }
    });

    m.def("debug_output", []() {

    });

    m.def("info_output", []() {

    });

    static auto _atexit = []() {
        if (Kokkos::is_initialized())
            Kokkos::finalize();
    };

    atexit(_atexit);
}
