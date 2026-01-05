/**
 * Python Bindings para DFG Extractor - VERSIÓN SIMPLIFICADA
 * Paso 1.3: Interfaz Python usando pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dfg_extractor.hpp"

namespace py = pybind11;
using namespace ompar;
using namespace pybind11::literals;  // Para usar "_a"

PYBIND11_MODULE(dfg_extractor_cpp, m) {
    m.doc() = "DFG Extractor optimizado en C++ para OMPar";
    
    // Estructura DFGNode
    py::class_<DFGNode>(m, "DFGNode")
        .def(py::init<>())
        .def(py::init<const std::string&, int, const std::string&>())
        .def_readwrite("name", &DFGNode::name)
        .def_readwrite("position", &DFGNode::position)
        .def_readwrite("type", &DFGNode::type)
        .def_readwrite("dependencies", &DFGNode::dependencies)
        .def("__repr__", [](const DFGNode& node) {
            return "<DFGNode name='" + node.name + "' pos=" + std::to_string(node.position) + ">";
        });
    
    // Estructura CodeToken
    py::class_<CodeToken>(m, "CodeToken")
        .def(py::init<>())
        .def_readwrite("text", &CodeToken::text)
        .def_readwrite("start_line", &CodeToken::start_line)
        .def_readwrite("start_col", &CodeToken::start_col)
        .def_readwrite("end_line", &CodeToken::end_line)
        .def_readwrite("end_col", &CodeToken::end_col)
        .def("__repr__", [](const CodeToken& token) {
            return "<CodeToken '" + token.text + "'>";
        });
    
    // Estructura DFGResult
    py::class_<DFGResult>(m, "DFGResult")
        .def(py::init<>())
        .def_readwrite("code_tokens", &DFGResult::code_tokens)
        .def_readwrite("dfg_nodes", &DFGResult::dfg_nodes)
        .def_readwrite("success", &DFGResult::success)
        .def_readwrite("error_message", &DFGResult::error_message)
        .def("__repr__", [](const DFGResult& result) {
            return "<DFGResult success=" + std::string(result.success ? "True" : "False") +
                   " tokens=" + std::to_string(result.code_tokens.size()) +
                   " nodes=" + std::to_string(result.dfg_nodes.size()) + ">";
        });
    
    // Estructura Stats
    py::class_<DFGExtractor::Stats>(m, "Stats")
        .def(py::init<>())
        .def_readwrite("total_extractions", &DFGExtractor::Stats::total_extractions)
        .def_readwrite("successful_extractions", &DFGExtractor::Stats::successful_extractions)
        .def_readwrite("failed_extractions", &DFGExtractor::Stats::failed_extractions)
        .def_readwrite("total_time_ms", &DFGExtractor::Stats::total_time_ms)
        .def_readwrite("avg_time_ms", &DFGExtractor::Stats::avg_time_ms)
        .def("__repr__", [](const DFGExtractor::Stats& stats) {
            return "<Stats total=" + std::to_string(stats.total_extractions) +
                   " avg_time=" + std::to_string(stats.avg_time_ms) + "ms>";
        });
    
    // Clase DFGExtractor
    py::class_<DFGExtractor>(m, "DFGExtractor")
        .def(py::init<>())
        .def("extract", &DFGExtractor::extract,
             "Extraer DFG de código fuente",
             py::arg("source_code"))
        .def("extract_batch", &DFGExtractor::extract_batch,
             "Extraer DFG de múltiples códigos (paralelizado con OpenMP)",
             py::arg("source_codes"))
        .def("get_stats", &DFGExtractor::get_stats,
             "Obtener estadísticas de rendimiento")
        .def("reset_stats", &DFGExtractor::reset_stats,
             "Resetear estadísticas")
        .def("__repr__", [](const DFGExtractor&) {
            return "<DFGExtractor (C++)>";
        });
    
    // Función de utilidad para benchmarking
    m.def("benchmark_extraction", 
        [](const std::string& code, int iterations) -> py::dict {
            DFGExtractor extractor;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < iterations; i++) {
                extractor.extract(code);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double total_ms = duration.count() / 1000.0;
            double avg_ms = total_ms / iterations;
            
            py::dict result;
            result["total_time_ms"] = total_ms;
            result["avg_time_ms"] = avg_ms;
            result["iterations"] = iterations;
            result["throughput_per_sec"] = (iterations / total_ms) * 1000.0;
            
            return result;
        }, 
        "Benchmark de extracción de DFG",
        "code"_a, "iterations"_a = 100
    );
}
