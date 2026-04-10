#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "lsh.h"

namespace py = pybind11;
using namespace vectordb::algorithms;

PYBIND11_MODULE(_lsh, m) {
    m.doc() = "VectorDB LSH Index Module";

    py::class_<IndexLSH>(m, "IndexLSH")
        .def(py::init<size_t, size_t, size_t, float>(),
             py::arg("dimension"), py::arg("num_hash_tables") = 8,
             py::arg("num_hash_functions") = 4, py::arg("r") = 1.0)
        .def("add", [](IndexLSH &self, py::array_t<float> x) {
            py::buffer_info buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be 2D array");
            }
            size_t n = buf.shape[0];
            self.add(n, static_cast<float*>(buf.ptr));
        })
        .def("search", [](IndexLSH &self, py::array_t<float> x, size_t k) {
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
        .def_property_readonly("ntotal", &IndexLSH::get_ntotal)
        .def_property_readonly("dimension", &IndexLSH::get_dimension)
        .def_property_readonly("num_hash_tables", &IndexLSH::get_num_hash_tables)
        .def_property_readonly("num_hash_functions", &IndexLSH::get_num_hash_functions)
        .def_property_readonly("num_probes", &IndexLSH::get_num_probes)
        .def("set_num_probes", &IndexLSH::set_num_probes, py::arg("n"),
             "Set number of probes per hash table for multi-probe search");
}
