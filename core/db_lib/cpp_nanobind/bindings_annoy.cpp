
#include &lt;nanobind/nanobind.h&gt;
#include &lt;nanobind/ndarray.h&gt;
#include "../cpp/algorithms/annoy/annoy.h"

namespace nb = nanobind;
using namespace vectordb::algorithms;

NB_MODULE(_annoy_nanobind, m) {
    m.doc() = "Annoy index algorithm for vector database (nanobind version)";
    m.attr("__version__") = "1.0.0";

    nb::class_&lt;IndexAnnoy&gt;(m, "IndexAnnoy")
        .def(nb::init&lt;size_t, size_t&gt;(), nb::arg("dimension"), nb::arg("n_trees") = 10,
             "Create an IndexAnnoy with given dimension and number of trees")
        .def("add", [](IndexAnnoy&amp; self, nb::ndarray&lt;float, nb::shape&lt;nb::any, nb::any&gt;, nb::c_contig&gt; x) {
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
        .def("search", [](IndexAnnoy&amp; self, nb::ndarray&lt;float, nb::shape&lt;nb::any, nb::any&gt;, nb::c_contig&gt; x, size_t k) {
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
        .def_prop_ro("ntotal", &amp;IndexAnnoy::get_ntotal)
        .def_prop_ro("dimension", &amp;IndexAnnoy::get_dimension)
        .def_prop_ro("n_trees", &amp;IndexAnnoy::get_n_trees)
        .def("size", &amp;IndexAnnoy::get_ntotal, "Get the number of vectors in the index")
        .def("get_dimension", &amp;IndexAnnoy::get_dimension, "Get the dimension of the vectors");
}

