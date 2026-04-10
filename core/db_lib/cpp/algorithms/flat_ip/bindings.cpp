#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "flat_ip.h"

namespace py = pybind11;
using namespace vectordb::algorithms;

PYBIND11_MODULE(_flat_ip, m) {
    m.doc() = "VectorDB FLAT-IP Index Module";

    py::class_<IndexFlatIP>(m, "IndexFlatIP")
        .def(py::init<size_t>())
        .def("add", [](IndexFlatIP &self, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be 2D array");
            }
            size_t n = buf.shape[0];
            size_t d = buf.shape[1];
            self.add(n, static_cast<float*>(buf.ptr));
        })
        .def("search", [](IndexFlatIP &self, py::array_t<float> x, size_t k) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be 2D array");
            }
            size_t n = buf.shape[0];
            py::array_t<float> distances({n, k});
            py::array_t<size_t> labels({n, k});
            py::buffer_info dist_buf = distances.request();
            py::buffer_info label_buf = labels.request();
            self.search(n, static_cast<float*>(buf.ptr), k, 
                       static_cast<float*>(dist_buf.ptr), 
                       static_cast<size_t*>(label_buf.ptr));
            return std::make_tuple(distances, labels);
        })
        .def_property_readonly("ntotal", &IndexFlatIP::get_ntotal)
        .def_property_readonly("dimension", &IndexFlatIP::get_dimension);
}
