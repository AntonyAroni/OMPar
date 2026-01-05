/**
 * DFG Extractor - C++ Implementation
 * Versión COMPLETA con tree-sitter
 */

#include "dfg_extractor.hpp"
#include <chrono>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace ompar {

DFGExtractor::DFGExtractor() {
    // Crear parser
    parser_ = ts_parser_new();
    
    // Configurar lenguaje C#/C
    const TSLanguage* lang = tree_sitter_c_sharp();
    if (!ts_parser_set_language(parser_, lang)) {
        std::cerr << "⚠️  Error configurando lenguaje tree-sitter" << std::endl;
    }
    
    // Inicializar estadísticas
    reset_stats();
    
    std::cout << "✅ DFG Extractor inicializado (C++ con tree-sitter)" << std::endl;
}

DFGExtractor::~DFGExtractor() {
    if (parser_) {
        ts_parser_delete(parser_);
        parser_ = nullptr;
    }
}

void DFGExtractor::reset_stats() {
    stats_ = Stats{0, 0, 0, 0.0, 0.0};
}

std::string DFGExtractor::get_node_text(TSNode node, const std::string& source_code) {
    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);
    
    if (start >= source_code.length() || end > source_code.length() || start >= end) {
        return "";
    }
    
    return source_code.substr(start, end - start);
}

bool DFGExtractor::is_identifier(TSNode node) {
    const char* type = ts_node_type(node);
    
    return (
        strcmp(type, "identifier") == 0 ||
        strcmp(type, "variable_declarator") == 0 ||
        strcmp(type, "parameter") == 0
    );
}

void DFGExtractor::tokenize_node(TSNode node, const std::string& source_code,
                                  std::vector<CodeToken>& tokens) {
    // Si es un nodo terminal (sin hijos), es un token
    uint32_t child_count = ts_node_child_count(node);
    
    if (child_count == 0) {
        std::string text = get_node_text(node, source_code);
        if (!text.empty() && text != " " && text != "\n" && text != "\t") {
            CodeToken token;
            token.text = text;
            
            TSPoint start = ts_node_start_point(node);
            TSPoint end = ts_node_end_point(node);
            
            token.start_line = start.row;
            token.start_col = start.column;
            token.end_line = end.row;
            token.end_col = end.column;
            
            tokens.push_back(token);
        }
    } else {
        // Recorrer hijos recursivamente
        for (uint32_t i = 0; i < child_count; i++) {
            TSNode child = ts_node_child(node, i);
            tokenize_node(child, source_code, tokens);
        }
    }
}

void DFGExtractor::analyze_node(TSNode node, const std::string& source_code,
                                 std::vector<DFGNode>& dfg,
                                 std::unordered_map<std::string, int>& var_to_index) {
    const char* node_type = ts_node_type(node);
    
    // Detectar asignaciones
    if (strcmp(node_type, "assignment_expression") == 0 ||
        strcmp(node_type, "simple_assignment_expression") == 0) {
        
        // Obtener el lado izquierdo (variable asignada)
        uint32_t child_count = ts_node_child_count(node);
        if (child_count >= 1) {
            TSNode left = ts_node_child(node, 0);
            std::string var_name = get_node_text(left, source_code);
            
            if (!var_name.empty()) {
                DFGNode dfg_node;
                dfg_node.name = var_name;
                dfg_node.position = static_cast<int>(dfg.size());
                dfg_node.type = "computedFrom";
                
                // Buscar dependencias en el lado derecho
                if (child_count >= 3) {
                    TSNode right = ts_node_child(node, 2);
                    
                    // Función para encontrar identificadores
                    std::function<void(TSNode)> find_deps = [&](TSNode n) {
                        if (is_identifier(n)) {
                            std::string dep_name = get_node_text(n, source_code);
                            auto it = var_to_index.find(dep_name);
                            if (it != var_to_index.end() && dep_name != var_name) {
                                dfg_node.dependencies.push_back(it->second);
                            }
                        }
                        
                        uint32_t count = ts_node_child_count(n);
                        for (uint32_t i = 0; i < count; i++) {
                            find_deps(ts_node_child(n, i));
                        }
                    };
                    
                    find_deps(right);
                }
                
                // Agregar al DFG
                var_to_index[var_name] = static_cast<int>(dfg.size());
                dfg.push_back(dfg_node);
            }
        }
    }
    
    // Detectar declaraciones de variables
    if (strcmp(node_type, "variable_declaration") == 0 ||
        strcmp(node_type, "local_declaration_statement") == 0) {
        
        uint32_t child_count = ts_node_child_count(node);
        for (uint32_t i = 0; i < child_count; i++) {
            TSNode child = ts_node_child(node, i);
            const char* child_type = ts_node_type(child);
            
            if (strcmp(child_type, "variable_declarator") == 0) {
                std::string var_name = get_node_text(child, source_code);
                
                // Limpiar el nombre (puede tener inicialización)
                size_t eq_pos = var_name.find('=');
                if (eq_pos != std::string::npos) {
                    var_name = var_name.substr(0, eq_pos);
                }
                // Trim
                var_name.erase(0, var_name.find_first_not_of(" \t\n"));
                var_name.erase(var_name.find_last_not_of(" \t\n") + 1);
                
                if (!var_name.empty()) {
                    DFGNode dfg_node;
                    dfg_node.name = var_name;
                    dfg_node.position = static_cast<int>(dfg.size());
                    dfg_node.type = "declaration";
                    
                    var_to_index[var_name] = static_cast<int>(dfg.size());
                    dfg.push_back(dfg_node);
                }
            }
        }
    }
    
    // Recorrer hijos recursivamente
    uint32_t child_count = ts_node_child_count(node);
    for (uint32_t i = 0; i < child_count; i++) {
        TSNode child = ts_node_child(node, i);
        analyze_node(child, source_code, dfg, var_to_index);
    }
}

DFGResult DFGExtractor::extract(const std::string& source_code) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DFGResult result;
    stats_.total_extractions++;
    
    try {
        // Parsear con tree-sitter
        TSTree* tree = ts_parser_parse_string(
            parser_,
            nullptr,
            source_code.c_str(),
            source_code.length()
        );
        
        if (!tree) {
            result.error_message = "Failed to parse code";
            result.success = false;
            stats_.failed_extractions++;
            return result;
        }
        
        // Obtener nodo raíz
        TSNode root_node = ts_tree_root_node(tree);
        
        // Tokenizar
        tokenize_node(root_node, source_code, result.code_tokens);
        
        // Extraer DFG
        std::unordered_map<std::string, int> var_to_index;
        analyze_node(root_node, source_code, result.dfg_nodes, var_to_index);
        
        // Limpiar
        ts_tree_delete(tree);
        
        result.success = true;
        stats_.successful_extractions++;
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        result.success = false;
        stats_.failed_extractions++;
    }
    
    // Actualizar estadísticas de tiempo
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;
    
    stats_.total_time_ms += time_ms;
    stats_.avg_time_ms = stats_.total_time_ms / stats_.total_extractions;
    
    return result;
}

std::vector<DFGResult> DFGExtractor::extract_batch(const std::vector<std::string>& source_codes) {
    std::vector<DFGResult> results(source_codes.size());
    
    // Paralelizar con OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < source_codes.size(); i++) {
        // Cada thread necesita su propio extractor para thread-safety
        DFGExtractor local_extractor;
        results[i] = local_extractor.extract(source_codes[i]);
    }
    
    return results;
}

} // namespace ompar
