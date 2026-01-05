# 🚀 Guía de Optimización C++/CUDA - OMPar

Esta guía documenta el proceso **paso a paso** de optimización de OMPar usando C++/CUDA.

---

## 📊 Estado Actual (Baseline)

| Componente | Tiempo | Porcentaje |
|------------|--------|------------|
| **MonoCoder (generación)** | 461.89 ms | 95.2% |
| **OMPify (clasificación)** | 23.49 ms | 4.8% |
| **Total** | 485.01 ms | 100% |

**Throughput actual**: 2.06 inferencias/segundo

---

## 🎯 Plan de Optimización

### Fase 1: DFG Extractor en C++ ✅ (EN PROGRESO)
- **Objetivo**: Acelerar parsing y extracción de Data Flow Graph
- **Speedup esperado**: 10-50x en parsing
- **Impacto en total**: ~5-10% mejora (parsing es pequeña parte)
- **Dificultad**: Moderada

### Fase 2: Inferencia con TensorRT/CUDA
- **Objetivo**: Acelerar MonoCoder (el cuello de botella principal)
- **Speedup esperado**: 5-10x
- **Impacto en total**: ~80-90% mejora
- **Dificultad**: Alta

### Fase 3: Pipeline Paralelo
- **Objetivo**: Procesar múltiples archivos en paralelo
- **Speedup esperado**: Nx (N = número de archivos)
- **Impacto**: Throughput masivo
- **Dificultad**: Moderada

---

## 📁 Estructura del Proyecto C++

```
OMPar/
├── cpp_extensions/          # Extensiones C++/CUDA
│   └── dfg_extractor/       # DFG Extractor en C++
│       ├── dfg_extractor.hpp    # Header
│       ├── dfg_extractor.cpp    # Implementación
│       ├── bindings.cpp         # Python bindings (pybind11)
│       ├── CMakeLists.txt       # Configuración de compilación
│       └── build.sh             # Script de compilación
│
├── parser/                  # Parser Python original (a reemplazar)
│   ├── DFG.py              # ← Será reemplazado por C++
│   └── my-languages.so     # Tree-sitter compilado
│
└── OMPify/
    └── model.py            # Usará DFG Extractor C++
```

---

## 🔨 FASE 1: DFG Extractor en C++

### Paso 1.1: Estructura Básica ✅

**Archivos creados:**
- `dfg_extractor.hpp` - Definiciones de clases y estructuras
- `dfg_extractor.cpp` - Implementación en C++
- `bindings.cpp` - Interfaz Python con pybind11

**Características:**
- ✅ Usa tree-sitter (mismo que Python)
- ✅ Paralelización con OpenMP para batch processing
- ✅ Optimizado con `-O3 -march=native`
- ✅ Estadísticas de rendimiento integradas

### Paso 1.2: Compilación

**Requisitos:**
```bash
# Instalar dependencias
sudo apt-get install cmake g++ libomp-dev

# En el entorno virtual
pip install pybind11
```

**Compilar:**
```bash
cd cpp_extensions/dfg_extractor
./build.sh
```

**Salida esperada:**
```
🔨 Compilando DFG Extractor C++
✅ Compilación exitosa!
El módulo 'dfg_extractor_cpp.so' está disponible
```

### Paso 1.3: Prueba del Módulo C++

**Test básico:**
```python
import dfg_extractor_cpp

# Crear extractor
extractor = dfg_extractor_cpp.DFGExtractor()

# Código de prueba
code = """
for (int i = 0; i < n; i++) {
    arr[i] = arr[i] + 1;
}
"""

# Extraer DFG
result = extractor.extract(code)

print(f"Success: {result.success}")
print(f"Tokens: {len(result.code_tokens)}")
print(f"DFG Nodes: {len(result.dfg_nodes)}")

# Ver estadísticas
stats = extractor.get_stats()
print(f"Avg time: {stats.avg_time_ms:.2f} ms")
```

### Paso 1.4: Benchmark C++ vs Python

**Script de comparación:**
```python
import time
from parser.DFG import DFG_csharp  # Python original
import dfg_extractor_cpp           # C++ nuevo

code = "for (int i = 0; i < n; i++) { arr[i] = 0; }"

# Benchmark Python
start = time.perf_counter()
for _ in range(100):
    # Llamar DFG Python original
    pass
python_time = (time.perf_counter() - start) * 1000

# Benchmark C++
cpp_result = dfg_extractor_cpp.benchmark_extraction(code, 100)

print(f"Python: {python_time:.2f} ms")
print(f"C++:    {cpp_result['total_time_ms']:.2f} ms")
print(f"Speedup: {python_time / cpp_result['total_time_ms']:.2f}x")
```

### Paso 1.5: Integración con OMPify

**Modificar `OMPify/model.py`:**

```python
# ANTES (Python):
from parser import DFG_csharp

def extract_dataflow(self, code, parser, lang):
    # ... código Python lento ...
    DFG, _ = parser[1](root_node, index_to_code, {})
    return code_tokens, dfg

# DESPUÉS (C++):
try:
    import dfg_extractor_cpp
    USE_CPP_DFG = True
except ImportError:
    USE_CPP_DFG = False
    from parser import DFG_csharp

def extract_dataflow(self, code, parser, lang):
    if USE_CPP_DFG:
        # Usar versión C++ (mucho más rápida)
        extractor = dfg_extractor_cpp.DFGExtractor()
        result = extractor.extract(code)
        return result.code_tokens, result.dfg_nodes
    else:
        # Fallback a Python
        # ... código original ...
```

### Paso 1.6: Benchmark Completo

```bash
# Ejecutar benchmark con DFG C++
python benchmark_simple.py \
    --baseline benchmark_baseline.json \
    --output benchmark_with_cpp_dfg.json \
    --iterations 20
```

**Resultado esperado:**
```
📊 COMPARACIÓN CON BASELINE
================================================================================
Test                           Baseline       Actual    Speedup     Mejora
--------------------------------------------------------------------------------
Simple loop                     25.47 ms     15.23 ms      1.67x    +40.2%
Reduction                      701.03 ms    695.12 ms      1.01x     +0.8%
...
```

**Nota**: El speedup será pequeño porque el parsing es solo ~5% del tiempo total.

---

## 🔥 FASE 2: TensorRT para MonoCoder (PRÓXIMO)

### Paso 2.1: Exportar Modelo a ONNX

**Objetivo**: Convertir MonoCoder (GPTNeoX) a formato ONNX

```python
import torch
from transformers import GPTNeoXForCausalLM

# Cargar modelo
model = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP')
model.eval()

# Exportar a ONNX
dummy_input = torch.randint(0, 50000, (1, 64))
torch.onnx.export(
    model,
    dummy_input,
    "monocoder_omp.onnx",
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'logits': {0: 'batch', 1: 'sequence'}
    }
)
```

### Paso 2.2: Optimizar con TensorRT

**C++ con TensorRT:**
```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>

class MonoCoderTensorRT {
private:
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    
public:
    void loadONNX(const char* onnx_path);
    std::vector<int> generate(const std::vector<int>& input_ids);
};
```

**Speedup esperado**: 5-10x (de 462ms a 50-90ms)

---

## 📊 Proyección de Mejoras

| Fase | Componente | Tiempo Actual | Tiempo Optimizado | Speedup |
|------|------------|---------------|-------------------|---------|
| Baseline | Total | 485.01 ms | - | 1.0x |
| **Fase 1** | DFG Parsing | ~10 ms | ~1 ms | 10x |
| | **Total** | **485 ms** | **~476 ms** | **1.02x** |
| **Fase 2** | MonoCoder | 462 ms | 50 ms | 9.2x |
| | **Total** | **485 ms** | **~73 ms** | **6.6x** |
| **Fase 3** | Pipeline (batch=10) | 485 ms/item | 73 ms/batch | 66x |
| | **Throughput** | **2.06/s** | **137/s** | **66x** |

---

## ✅ Checklist de Progreso

### Fase 1: DFG Extractor C++
- [x] Crear estructura de archivos C++
- [x] Implementar DFGExtractor básico
- [x] Crear Python bindings con pybind11
- [x] Configurar CMake
- [ ] **Compilar módulo** ← SIGUIENTE PASO
- [ ] Probar módulo standalone
- [ ] Integrar con OMPify
- [ ] Benchmark y comparar
- [ ] Validar correctitud

### Fase 2: TensorRT (Pendiente)
- [ ] Exportar MonoCoder a ONNX
- [ ] Crear wrapper C++ con TensorRT
- [ ] Python bindings
- [ ] Integrar con compAI.py
- [ ] Benchmark y comparar

### Fase 3: Pipeline Paralelo (Pendiente)
- [ ] Implementar pipeline en C++
- [ ] Batch processing
- [ ] Benchmark

---

## 🚀 Próximos Pasos Inmediatos

### 1. Compilar DFG Extractor C++

```bash
cd cpp_extensions/dfg_extractor
./build.sh
```

### 2. Si hay errores de compilación

**Posibles problemas:**
- **pybind11 no encontrado**: `pip install pybind11`
- **tree-sitter no encontrado**: Verificar que `parser/my-languages.so` existe
- **OpenMP no encontrado**: `sudo apt-get install libomp-dev`

### 3. Probar el módulo

```bash
cd ../..
python3 -c "import dfg_extractor_cpp; print('✅ Módulo cargado!')"
```

### 4. Benchmark

```bash
python3 test_cpp_dfg.py  # Script de prueba (crear)
```

---

## 📝 Notas Importantes

1. **Cambios incrementales**: Cada fase es independiente y verificable
2. **Fallback a Python**: Si C++ falla, el código Python original sigue funcionando
3. **Benchmark después de cada cambio**: Siempre medir el impacto
4. **Validar correctitud**: Ejecutar `simple_tests.py` después de cada cambio

---

¿Listo para compilar? Ejecuta:

```bash
cd cpp_extensions/dfg_extractor
./build.sh
```
