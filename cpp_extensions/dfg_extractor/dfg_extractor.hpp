/**
 * DFG Extractor - C++ Implementation
 * Versión COMPLETA con tree-sitter
 * 
 * Extrae Data Flow Graph de código C/C++ usando tree-sitter
 * Optimizado para velocidad con C++17 y OpenMP
 */

#ifndef DFG_EXTRACTOR_HPP
#define DFG_EXTRACTOR_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <functional>

// Incluir tree-sitter
#include <tree_sitter/api.h>

// Declaración externa del lenguaje C#
extern "C" {
    const TSLanguage *tree_sitter_c_sharp(void);
}

namespace ompar {

/**
 * Representa un nodo en el Data Flow Graph
 */
struct DFGNode {
    std::string name;           // Nombre de la variable
    int position;               // Posición en el código (índice)
    std::string type;           // Tipo de operación (comesFrom, computedFrom, etc.)
    std::vector<int> dependencies; // Índices de nodos de los que depende
    
    DFGNode() : position(-1) {}
    
    DFGNode(const std::string& n, int pos, const std::string& t)
        : name(n), position(pos), type(t) {}
};

/**
 * Representa un token de código
 */
struct CodeToken {
    std::string text;
    int start_line;
    int start_col;
    int end_line;
    int end_col;
    
    CodeToken() : start_line(0), start_col(0), end_line(0), end_col(0) {}
};

/**
 * Resultado de la extracción de DFG
 */
struct DFGResult {
    std::vector<CodeToken> code_tokens;
    std::vector<DFGNode> dfg_nodes;
    bool success;
    std::string error_message;
    
    DFGResult() : success(false) {}
};

/**
 * Clase principal para extraer DFG de código C/C++
 */
class DFGExtractor {
public:
    /**
     * Constructor
     * Inicializa el parser de tree-sitter para C#/C
     */
    DFGExtractor();
    
    /**
     * Destructor
     * Libera recursos del parser
     */
    ~DFGExtractor();
    
    /**
     * Extraer DFG de código fuente
     * 
     * @param source_code Código fuente en C/C++
     * @return DFGResult con tokens y nodos del DFG
     */
    DFGResult extract(const std::string& source_code);
    
    /**
     * Extraer DFG de múltiples códigos (batch processing)
     * Usa OpenMP para paralelizar
     * 
     * @param source_codes Vector de códigos fuente
     * @return Vector de DFGResult
     */
    std::vector<DFGResult> extract_batch(const std::vector<std::string>& source_codes);
    
    /**
     * Obtener estadísticas de rendimiento
     */
    struct Stats {
        size_t total_extractions;
        size_t successful_extractions;
        size_t failed_extractions;
        double total_time_ms;
        double avg_time_ms;
    };
    
    Stats get_stats() const { return stats_; }
    void reset_stats();

private:
    // Parser de tree-sitter
    TSParser* parser_;
    
    // Estadísticas
    Stats stats_;
    
    // Métodos privados
    
    /**
     * Tokenizar el código usando tree-sitter
     */
    void tokenize_node(TSNode node, const std::string& source_code, 
                      std::vector<CodeToken>& tokens);
    
    /**
     * Extraer texto de un nodo
     */
    std::string get_node_text(TSNode node, const std::string& source_code);
    
    /**
     * Analizar dependencias de datos
     */
    void analyze_node(TSNode node, const std::string& source_code,
                     std::vector<DFGNode>& dfg,
                     std::unordered_map<std::string, int>& var_to_index);
    
    /**
     * Verificar si es un nodo de identificador
     */
    bool is_identifier(TSNode node);
};

} // namespace ompar

#endif // DFG_EXTRACTOR_HPP
