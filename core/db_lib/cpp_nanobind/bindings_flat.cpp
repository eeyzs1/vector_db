
#include &lt;nanobind/nanobind.h&gt;
#include &lt;nanobind/ndarray.h&gt;
#include "../cpp/algorithms/flat/flat.h"

namespace nb = nanobind;
using namespace vectordb::algorithms;

NB_MODULE(_flat_nanobind, m) {
    m.doc() = "Flat L2 index algorithm for vector database (nanobind version)";
    m.attr("__version__") = "1.0.0";

    nb::class_&lt;IndexFlatL2&gt;(m, "IndexFlatL2")
        .def(nb::init&lt;size_t&gt;(), nb::arg("dimension"), "Create an IndexFlatL2 with given dimension")
        .def("add", [](IndexFlatL2&amp; self, nb::ndarray&lt;float, nb::shape&lt;nb::any, nb::any&gt;, nb::c_contig&gt; x) {
            if (x.ndim() != 2) {
                throw std::runtime_error("Input must be 2D array");
            }
            size_t n = x.shape(0);
            size_t d = x.shape(1);
            if (d != self.get_dimension()) {
                throw std::runtime_error("Dimension mismatch");
            }
            self.add(n, static_cast&lt;float*&gt;(x.data()));
        }, nb::arg("x"), "Add vectors to the index")
        .def("search", [](IndexFlatL2&amp; self, nb::ndarray&lt;float, nb::shape&lt;nb::any, nb::any&gt;, nb::c_contig&gt; x, size_t k) {
            if (x.ndim() != 2) {
                throw std::runtime_error("Input must be 2D array");
            }
            size_t n = x.shape(0);
            size_t d = x.shape(1);
            if (d != self.get_dimension()) {
                throw std::runtime_error("Dimension mismatch");
            }

            nb::ndarray&lt;float, nb::shape&lt;nb::any, nb::any&gt;&gt; distances({n, k});
            nb::ndarray&lt;size_t, nb::shape&lt;nb::any, nb::any&gt;&gt; labels({n, k});

            self.search(n, static_cast&lt;float*&gt;(x.data()), k,
                       static_cast&lt;float*&gt;(distances.data()),
                       static_cast&lt;size_t*&gt;(labels.data()));

            return std::make_tuple(distances, labels);
        }, nb::arg("x"), nb::arg("k"), "Search for nearest neighbors")
        .def("size", &amp;IndexFlatL2::size, "Get the number of vectors in the index")
        .def("get_dimension", &amp;IndexFlatL2::get_dimension, "Get the dimension of the vectors");
}
