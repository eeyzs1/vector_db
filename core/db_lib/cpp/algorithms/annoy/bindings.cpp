#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "annoy.h"

namespace py = pybind11;
using namespace vectordb::algorithms;

PYBIND11_MODULE(vectordb_annoy, m) {
    m.doc() = "VectorDB Annoy Index Module";

    py::class_<IndexAnnoy>(m, "IndexAnnoy")
        .def(py::init<size_t, size_t>(),
             py::arg("dimension"), py::arg("n_trees") = 10)
        .def("add", [](IndexAnnoy &self, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be 2D array");
            }
            size_t n = buf.shape[0];
            self.add(n, static_cast<float*>(buf.ptr));
        })
        .def("search", [](IndexAnnoy &self, py::array_t<float> x, size_t k) {
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
        .def_property_readonly("ntotal", &IndexAnnoy::get_ntotal)
        .def_property_readonly("dimension", &IndexAnnoy::get_dimension)
        .def_property_readonly("n_trees", &IndexAnnoy::get_n_trees);
}
