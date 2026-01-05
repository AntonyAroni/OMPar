#!/usr/bin/env python3
"""
Test del módulo DFG Extractor C++
Fase 1.6: Verificación del módulo compilado
"""

import sys
import time

print("="*60)
print("🧪 TEST: DFG Extractor C++")
print("="*60)

# Intentar importar el módulo
print("\n📦 Importando módulo C++...")
try:
    import dfg_extractor_cpp
    print("✅ Módulo importado exitosamente!")
    print(f"   Ubicación: {dfg_extractor_cpp.__file__}")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    sys.exit(1)

# Mostrar documentación
print(f"\n📖 Documentación:")
print(f"   {dfg_extractor_cpp.__doc__}")

# Crear instancia del extractor
print("\n🔧 Creando DFGExtractor...")
try:
    extractor = dfg_extractor_cpp.DFGExtractor()
    print("✅ Extractor creado!")
except Exception as e:
    print(f"❌ Error creando extractor: {e}")
    sys.exit(1)

# Probar extracción
print("\n🔬 Probando extracción de DFG...")
test_code = """for (int i = 0; i < n; i++) {
    arr[i] = 0;
}"""

try:
    start = time.perf_counter()
    result = extractor.extract(test_code)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"✅ Extracción completada en {elapsed:.2f} ms")
    print(f"   Success: {result.success}")
    print(f"   Tokens: {len(result.code_tokens)}")
    print(f"   DFG Nodes: {len(result.dfg_nodes)}")
    
    if result.code_tokens:
        print(f"   Primer token: {result.code_tokens[0]}")
    if result.dfg_nodes:
        print(f"   Primer nodo: {result.dfg_nodes[0]}")
        
except Exception as e:
    print(f"❌ Error en extracción: {e}")
    import traceback
    traceback.print_exc()

# Probar estadísticas
print("\n📊 Estadísticas:")
try:
    stats = extractor.get_stats()
    print(f"   Total extracciones: {stats.total_extractions}")
    print(f"   Exitosas: {stats.successful_extractions}")
    print(f"   Fallidas: {stats.failed_extractions}")
    print(f"   Tiempo promedio: {stats.avg_time_ms:.4f} ms")
except Exception as e:
    print(f"❌ Error obteniendo stats: {e}")

# Probar benchmark
print("\n⏱️  Benchmark (100 iteraciones)...")
try:
    bench = dfg_extractor_cpp.benchmark_extraction(test_code, 100)
    print(f"   Tiempo total: {bench['total_time_ms']:.2f} ms")
    print(f"   Tiempo promedio: {bench['avg_time_ms']:.4f} ms")
    print(f"   Throughput: {bench['throughput_per_sec']:.0f} extracciones/segundo")
except Exception as e:
    print(f"❌ Error en benchmark: {e}")
    import traceback
    traceback.print_exc()

# Probar batch
print("\n📦 Probando batch processing...")
try:
    codes = [test_code] * 10
    start = time.perf_counter()
    results = extractor.extract_batch(codes)
    elapsed = (time.perf_counter() - start) * 1000
    
    successful = sum(1 for r in results if r.success)
    print(f"✅ Batch completado en {elapsed:.2f} ms")
    print(f"   Items procesados: {len(results)}")
    print(f"   Exitosos: {successful}")
    print(f"   Tiempo por item: {elapsed/len(results):.2f} ms")
except Exception as e:
    print(f"❌ Error en batch: {e}")

print("\n" + "="*60)
print("✅ TEST COMPLETADO")
print("="*60)
