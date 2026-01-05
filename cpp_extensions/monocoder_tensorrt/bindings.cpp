/**
 * Python bindings for MonoCoder TensorRT
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "monocoder_trt.hpp"

namespace py = pybind11;

PYBIND11_MODULE(monocoder_trt_cpp, m) {
    m.doc() = "MonoCoder TensorRT Optimized Inference Module (C++ Hybrid)";
    
    py::class_<ompar::MonoCoderTRT>(m, "MonoCoderTRT")
        .def(py::init<const std::string&>(), "Initialize with path to .engine file")
        .def("generate", &ompar::MonoCoderTRT::generate, 
             py::arg("start_ids"), py::arg("max_length") = 64, py::arg("stream_ptr") = 0,
             "Generate tokens from start_ids using greedy decoding")
        .def("forward", &ompar::MonoCoderTRT::forward,
             py::arg("input_ids"), py::arg("stream_ptr") = 0,
             "Run forward pass (returns logits for the last token)");
}
