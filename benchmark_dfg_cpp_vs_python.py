#!/usr/bin/env python3
"""
Benchmark comparativo: DFG Extractor C++ vs Python
Compara el rendimiento del parsing/DFG extraction
"""

import sys
import time
import statistics

# Importar módulo C++
print("📦 Cargando módulos...")
try:
    import dfg_extractor_cpp
    print("   ✅ dfg_extractor_cpp cargado")
except ImportError as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Importar módulo Python original
sys.path.insert(0, 'OMPify')
sys.path.insert(0, 'parser')
try:
    from parser import DFG_csharp
    from parser import (remove_comments_and_docstrings,
                        tree_to_token_index,
                        index_to_code_token)
    from tree_sitter import Language, Parser
    
    # Configurar parser Python
    LANGUAGE = Language('parser/my-languages.so', 'c_sharp')
    py_parser = Parser()
    py_parser.set_language(LANGUAGE)
    print("   ✅ Parser Python cargado")
except Exception as e:
    print(f"   ❌ Error cargando parser Python: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Casos de prueba
test_codes = [
    ("Simple loop", """for (int i = 0; i < n; i++) {
    arr[i] = 0;
}"""),
    ("Reduction", """for (int i = 0; i < n; i++) {
    total += arr[i];
}"""),
    ("Array copy", """for (int i = 0; i < n; i++) {
    dest[i] = src[i];
}"""),
    ("Element-wise", """for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
}"""),
    ("Nested loop", """for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}"""),
]

ITERATIONS = 100

def benchmark_cpp(code: str, iterations: int) -> dict:
    """Benchmark del extractor C++"""
    extractor = dfg_extractor_cpp.DFGExtractor()
    
    # Warm-up
    extractor.extract(code)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = extractor.extract(code)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        'mean_ms': statistics.mean(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'tokens': len(result.code_tokens),
        'dfg_nodes': len(result.dfg_nodes),
    }

def benchmark_python(code: str, iterations: int) -> dict:
    """Benchmark del extractor Python original"""
    
    def extract_dataflow_python(code_str):
        try:
            # Parsear
            tree = py_parser.parse(bytes(code_str, 'utf8'))
            root_node = tree.root_node
            tokens_index = tree_to_token_index(root_node)
            code_lines = code_str.split('\n')
            code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
            index_to_code = {}
            for idx, (index, tok) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, tok)
            try:
                DFG, _ = DFG_csharp(root_node, index_to_code, {})
            except:
                DFG = []
            return code_tokens, DFG
        except:
            return [], []
    
    # Warm-up
    extract_dataflow_python(code)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        tokens, dfg = extract_dataflow_python(code)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        'mean_ms': statistics.mean(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'tokens': len(tokens),
        'dfg_nodes': len(dfg),
    }

# Ejecutar benchmarks
print("\n" + "="*80)
print("📊 BENCHMARK: DFG Extractor C++ vs Python")
print("="*80)
print(f"Iteraciones por test: {ITERATIONS}")
print("="*80)

results = []

for name, code in test_codes:
    print(f"\n📝 Test: {name}")
    print("-"*60)
    
    # C++
    print("   🔧 Ejecutando C++...", end=' ', flush=True)
    cpp_result = benchmark_cpp(code, ITERATIONS)
    print(f"✓ {cpp_result['mean_ms']:.4f} ms")
    
    # Python
    print("   🐍 Ejecutando Python...", end=' ', flush=True)
    py_result = benchmark_python(code, ITERATIONS)
    print(f"✓ {py_result['mean_ms']:.4f} ms")
    
    # Calcular speedup
    speedup = py_result['mean_ms'] / cpp_result['mean_ms']
    improvement = ((py_result['mean_ms'] - cpp_result['mean_ms']) / py_result['mean_ms']) * 100
    
    results.append({
        'name': name,
        'cpp': cpp_result,
        'python': py_result,
        'speedup': speedup,
        'improvement': improvement
    })
    
    print(f"\n   ⚡ Speedup: {speedup:.2f}x ({improvement:.1f}% más rápido)")
    print(f"   📊 Tokens: C++={cpp_result['tokens']}, Python={py_result['tokens']}")
    print(f"   📊 DFG Nodes: C++={cpp_result['dfg_nodes']}, Python={py_result['dfg_nodes']}")

# Resumen
print("\n" + "="*80)
print("📊 RESUMEN")
print("="*80)
print(f"\n{'Test':<20} {'Python (ms)':>12} {'C++ (ms)':>12} {'Speedup':>10}")
print("-"*60)

total_speedups = []
for r in results:
    print(f"{r['name']:<20} {r['python']['mean_ms']:>12.4f} {r['cpp']['mean_ms']:>12.4f} {r['speedup']:>9.2f}x")
    total_speedups.append(r['speedup'])

avg_speedup = statistics.mean(total_speedups)
print("-"*60)
print(f"{'PROMEDIO':<20} {'':<12} {'':<12} {avg_speedup:>9.2f}x")
print("="*80)

print(f"\n✅ El DFG Extractor C++ es {avg_speedup:.2f}x más rápido que Python!")
print("="*80)
