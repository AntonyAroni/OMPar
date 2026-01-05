# 📊 Guía de Benchmarking - OMPar

Esta guía explica cómo medir y comparar el rendimiento de OMPar antes y después de realizar optimizaciones.

---

## 🎯 Objetivo

Medir de forma precisa:
- ⏱️ **Tiempos de ejecución** de cada componente
- 💾 **Uso de memoria** durante la inferencia
- 🚀 **Throughput** (inferencias por segundo)
- 📊 **Comparaciones** antes/después de optimizaciones

---

## 📋 Requisitos

Instalar dependencias adicionales para benchmarking:

```bash
pip install psutil gputil
```

---

## 🚀 Uso Básico

### 1️⃣ Crear Baseline (ANTES de optimizaciones)

**IMPORTANTE**: Ejecuta esto ANTES de hacer cualquier cambio al código.

```bash
python benchmark_performance.py \
    --model_weights model \
    --iterations 20 \
    --save-baseline
```

Esto creará el archivo `benchmark_baseline.json` con las métricas actuales.

**Salida esperada:**
```
🚀 BENCHMARK DE RENDIMIENTO - OMPar
================================================================================
Dispositivo: cuda
Iteraciones: 20
================================================================================

⏱️  Midiendo tiempo de inicialización del modelo...
✅ Inicialización: 2847.32 ms

📊 BENCHMARKS INDIVIDUALES
================================================================================

📊 Benchmarking: Simple loop
   Iteraciones: 20
   🔥 Warm-up... ✓
   ⏱️  Ejecutando iteraciones... 5 10 15 20 ✓
   ✅ Media: 45.23 ms
   📈 Min/Max: 42.11 / 51.34 ms
   📊 Desv. Est.: 2.45 ms
   🚀 Throughput: 22.11 iter/s

...
```

### 2️⃣ Realizar Optimizaciones

Implementa tus mejoras en C++/CUDA siguiendo las recomendaciones.

### 3️⃣ Comparar con Baseline (DESPUÉS de optimizaciones)

```bash
python benchmark_performance.py \
    --model_weights model \
    --iterations 20 \
    --baseline benchmark_baseline.json \
    --output benchmark_optimized.json
```

**Salida esperada:**
```
📊 COMPARACIÓN CON BASELINE
================================================================================

Simple loop:
  Baseline:  45.23 ms
  Actual:    12.34 ms
  Speedup:   3.67x
  Mejora:    +72.7%

Reduction:
  Baseline:  48.91 ms
  Actual:    13.21 ms
  Speedup:   3.70x
  Mejora:    +73.0%

...
```

---

## 📊 Métricas Medidas

### Tiempos de Ejecución

| Métrica | Descripción |
|---------|-------------|
| **mean_ms** | Tiempo promedio de ejecución |
| **median_ms** | Mediana (más robusta a outliers) |
| **min_ms** | Mejor caso |
| **max_ms** | Peor caso |
| **stdev_ms** | Desviación estándar (variabilidad) |
| **p95_ms** | Percentil 95 (95% de casos son más rápidos) |
| **p99_ms** | Percentil 99 (99% de casos son más rápidos) |

### Componentes del Pipeline

El benchmark mide cada etapa por separado:

1. **classification_ms**: Tiempo de OMPify (detectar si es paralelizable)
2. **generation_ms**: Tiempo de MonoCoder (generar pragma)
3. **formatting_ms**: Tiempo de formateo del pragma
4. **total_ms**: Tiempo total end-to-end

### Memoria

- **memory_before_mb**: Memoria antes de inferencia
- **memory_after_mb**: Memoria después de inferencia
- **memory_peak_mb**: Pico de memoria durante inferencia
- **memory_increase_mb**: Incremento de memoria

### Throughput

- **iterations_per_second**: Cuántas inferencias por segundo
- **ms_per_iteration**: Milisegundos por inferencia

---

## 🔬 Casos de Prueba

El benchmark incluye 5 casos de prueba representativos:

1. **Simple loop**: Inicialización básica
2. **Reduction**: Suma acumulativa
3. **Array copy**: Copia de arrays
4. **Element-wise operation**: Operación elemento a elemento
5. **Complex operation**: Bucle anidado con operaciones complejas

---

## 📈 Interpretación de Resultados

### Speedup

```
Speedup = Tiempo_Baseline / Tiempo_Actual
```

- **Speedup > 1.0**: Mejora (más rápido)
- **Speedup = 1.0**: Sin cambios
- **Speedup < 1.0**: Regresión (más lento)

### Ejemplos

| Speedup | Interpretación |
|---------|----------------|
| 2.0x | 2 veces más rápido (50% del tiempo original) |
| 3.0x | 3 veces más rápido (33% del tiempo original) |
| 10.0x | 10 veces más rápido (10% del tiempo original) |

### Mejora Porcentual

```
Mejora% = ((Tiempo_Baseline - Tiempo_Actual) / Tiempo_Baseline) × 100
```

- **+50%**: Reducción del 50% en tiempo (2x más rápido)
- **+75%**: Reducción del 75% en tiempo (4x más rápido)
- **+90%**: Reducción del 90% en tiempo (10x más rápido)

---

## 🎯 Objetivos de Optimización

### Fase 1: Quick Wins
- **Objetivo**: 3-5x speedup
- **Tiempo**: 1-2 semanas
- **Implementaciones**: DFG Extractor C++, Cache, ONNX Runtime

### Fase 2: Optimizaciones Medias
- **Objetivo**: 8-12x speedup
- **Tiempo**: 2-4 semanas
- **Implementaciones**: Static Analyzer, Pipeline Paralelo, TensorRT

### Fase 3: Producción
- **Objetivo**: 10-15x speedup
- **Tiempo**: 4-6 semanas
- **Implementaciones**: CLI standalone, Memory optimizations

---

## 📝 Ejemplo Completo

### Paso 1: Baseline

```bash
# Crear baseline ANTES de optimizaciones
python benchmark_performance.py --save-baseline --iterations 50

# Resultado: benchmark_baseline.json creado
```

### Paso 2: Implementar Optimización

Por ejemplo, implementar DFG Extractor en C++:

```bash
# Compilar módulo C++
cd cpp_extensions
mkdir build && cd build
cmake ..
make -j8
cd ../..
```

### Paso 3: Comparar

```bash
# Ejecutar benchmark con optimización
python benchmark_performance.py \
    --baseline benchmark_baseline.json \
    --iterations 50 \
    --output benchmark_with_cpp_dfg.json

# Ver comparación automática
```

### Paso 4: Analizar Resultados

```bash
# Ver archivo JSON con resultados detallados
cat benchmark_with_cpp_dfg.json | jq '.benchmarks[0]'
```

---

## 🔍 Análisis Avanzado

### Comparar Múltiples Versiones

```python
import json
import pandas as pd

# Cargar resultados
baseline = json.load(open('benchmark_baseline.json'))
opt1 = json.load(open('benchmark_opt1.json'))
opt2 = json.load(open('benchmark_opt2.json'))

# Crear tabla comparativa
data = []
for b, o1, o2 in zip(baseline['benchmarks'], opt1['benchmarks'], opt2['benchmarks']):
    data.append({
        'Test': b['name'],
        'Baseline (ms)': b['times']['mean_ms'],
        'Opt1 (ms)': o1['times']['mean_ms'],
        'Opt2 (ms)': o2['times']['mean_ms'],
        'Speedup Opt1': b['times']['mean_ms'] / o1['times']['mean_ms'],
        'Speedup Opt2': b['times']['mean_ms'] / o2['times']['mean_ms']
    })

df = pd.DataFrame(data)
print(df.to_markdown(index=False))
```

### Visualizar Resultados

```python
import matplotlib.pyplot as plt

# Gráfico de speedup
tests = [b['name'] for b in baseline['benchmarks']]
speedups = [
    baseline['benchmarks'][i]['times']['mean_ms'] / 
    opt1['benchmarks'][i]['times']['mean_ms']
    for i in range(len(tests))
]

plt.figure(figsize=(10, 6))
plt.bar(tests, speedups)
plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
plt.ylabel('Speedup (x)')
plt.title('Speedup por Optimización')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('speedup_comparison.png')
```

---

## ⚠️ Consideraciones Importantes

### 1. Warm-up

El benchmark hace un "warm-up" antes de medir:
- Carga modelos en memoria
- Inicializa CUDA kernels
- Llena caches

**No omitas el warm-up** o tendrás mediciones incorrectas.

### 2. Número de Iteraciones

- **Mínimo recomendado**: 10 iteraciones
- **Recomendado**: 20-50 iteraciones
- **Para paper/publicación**: 100+ iteraciones

Más iteraciones = resultados más confiables pero más tiempo.

### 3. Variabilidad

Si `stdev_ms` es muy alto (>10% de `mean_ms`):
- Aumentar número de iteraciones
- Cerrar otros programas
- Verificar throttling de GPU/CPU

### 4. GPU vs CPU

Los resultados varían significativamente:
- **GPU**: Mejor para batches grandes
- **CPU**: Mejor para latencia baja

Siempre especifica qué dispositivo usaste.

---

## 📊 Formato de Resultados JSON

```json
{
  "timestamp": "2026-01-04T20:59:00",
  "device": "cuda",
  "initialization_time_ms": 2847.32,
  "system_info": {
    "cpu": {...},
    "memory": {...},
    "gpu": {...}
  },
  "benchmarks": [
    {
      "name": "Simple loop",
      "iterations": 20,
      "code_length": 45,
      "times": {
        "mean_ms": 45.23,
        "median_ms": 44.87,
        "min_ms": 42.11,
        "max_ms": 51.34,
        "stdev_ms": 2.45,
        "p95_ms": 49.12,
        "p99_ms": 50.87
      },
      "components": {
        "classification_ms": {...},
        "generation_ms": {...},
        "formatting_ms": {...},
        "total_ms": {...}
      },
      "throughput": {
        "iterations_per_second": 22.11,
        "ms_per_iteration": 45.23
      }
    }
  ],
  "batch_benchmark": {...},
  "memory_benchmark": {...}
}
```

---

## 🎓 Tips y Mejores Prácticas

1. **Siempre crea baseline ANTES** de hacer cambios
2. **Usa mismo hardware** para comparaciones justas
3. **Cierra otros programas** durante benchmarking
4. **Ejecuta múltiples veces** y promedia
5. **Documenta cambios** entre versiones
6. **Guarda todos los JSONs** para referencia futura
7. **Verifica que optimizaciones no rompan correctitud**

---

## 🔗 Archivos Relacionados

- Script de benchmark: [`benchmark_performance.py`](benchmark_performance.py)
- Pruebas de correctitud: [`simple_tests.py`](simple_tests.py)
- Resultados de pruebas: [`SIMPLE_TESTS_RESULTS.md`](SIMPLE_TESTS_RESULTS.md)

---

## 📞 Troubleshooting

### Error: "No module named 'psutil'"

```bash
pip install psutil gputil
```

### Error: "CUDA out of memory"

Reduce el número de iteraciones o usa CPU:

```bash
CUDA_VISIBLE_DEVICES="" python benchmark_performance.py ...
```

### Resultados inconsistentes

1. Cerrar otros programas
2. Aumentar iteraciones
3. Verificar temperatura de GPU/CPU
4. Deshabilitar turbo boost si es necesario

---

## ✅ Checklist Pre-Optimización

- [ ] Ejecutar `benchmark_performance.py --save-baseline`
- [ ] Verificar que `benchmark_baseline.json` existe
- [ ] Documentar versión actual del código
- [ ] Hacer commit en git antes de cambios
- [ ] Ejecutar `simple_tests.py` para verificar correctitud

## ✅ Checklist Post-Optimización

- [ ] Ejecutar `benchmark_performance.py --baseline benchmark_baseline.json`
- [ ] Verificar speedup obtenido
- [ ] Ejecutar `simple_tests.py` para verificar correctitud
- [ ] Documentar cambios realizados
- [ ] Guardar resultados con nombre descriptivo
- [ ] Hacer commit con resultados

---

**¡Listo para empezar a optimizar! 🚀**
