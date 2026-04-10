#include "ivf.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace vectordb::algorithms;

PYBIND11_MODULE(vectordb_ivf, m) {
    m.doc() = "IVF (Inverted File) index implementation";

    py::class_<IndexIVF>(m, "IndexIVF")
        .def(py::init<size_t, size_t>(),
             py::arg("dimension"),
             py::arg("nlist") = 100)
        .def("train", [](IndexIVF& self, py::array_t<float, py::array::c_style | py::array::forcecast> x) {
            auto buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be a 2D array");
            }
            size_t n = buf.shape[0];
            size_t dim = buf.shape[1];
            if (dim != self.get_dimension()) {
                throw std::invalid_argument("Dimension mismatch");
            }
            self.train(n, static_cast<const float*>(buf.ptr));
        }, py::arg("x"), "Train the index with vectors")
        .def("add", [](IndexIVF& self, py::array_t<float, py::array::c_style | py::array::forcecast> x) {
            auto buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be a 2D array");
            }
            size_t n = buf.shape[0];
            size_t dim = buf.shape[1];
            if (dim != self.get_dimension()) {
                throw std::invalid_argument("Dimension mismatch");
            }
            self.add(n, static_cast<const float*>(buf.ptr));
        }, py::arg("x"), "Add vectors to the index")
        .def("search", [](IndexIVF& self, py::array_t<float, py::array::c_style | py::array::forcecast> x, size_t k) {
            auto buf = x.request();
            if (buf.ndim != 2) {
                throw std::invalid_argument("Input must be a 2D array");
            }
            size_t n = buf.shape[0];
            size_t dim = buf.shape[1];
            if (dim != self.get_dimension()) {
                throw std::invalid_argument("Dimension mismatch");
            }

            auto dist_arr = py::array_t<float>({n, k});
            auto label_arr = py::array_t<size_t>({n, k});

            auto dist_buf = dist_arr.request();
            auto label_buf = label_arr.request();

            self.search(n, static_cast<const float*>(buf.ptr), k,
                        static_cast<float*>(dist_buf.ptr),
                        static_cast<size_t*>(label_buf.ptr));

            auto label_int64 = py::array_t<int64_t>({n, k});
            auto label_int64_buf = label_int64.request();
            auto src_labels = static_cast<size_t*>(label_buf.ptr);
            auto dst_labels = static_cast<int64_t*>(label_int64_buf.ptr);
            for (size_t i = 0; i < n * k; ++i) {
                dst_labels[i] = static_cast<int64_t>(src_labels[i]);
            }

            return py::make_tuple(dist_arr, label_int64);
        }, py::arg("x"), py::arg("k"), "Search for k nearest neighbors")
        .def("set_nprobe", &IndexIVF::set_nprobe, py::arg("nprobe"), "Set number of probes")
        .def_property_readonly("ntotal", [](IndexIVF& self) { return self.get_ntotal(); }, "Get the number of vectors")
        .def("get_dimension", &IndexIVF::get_dimension, "Get the dimension of the vectors")
        .def("get_nlist", &IndexIVF::get_nlist, "Get the number of lists")
        .def("get_nprobe", &IndexIVF::get_nprobe, "Get the number of probes");
}
