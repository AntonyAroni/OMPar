#!/usr/bin/env python3
"""
Sistema de Benchmarking para OMPar
Mide tiempos de ejecución de cada componente del pipeline
"""

import torch
import argparse
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import GPUtil
from compAI import OMPAR

class PerformanceBenchmark:
    def __init__(self, model_path: str, device: str, args):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'system_info': self._get_system_info(),
            'benchmarks': []
        }
        
        # Medir tiempo de inicialización
        print("⏱️  Midiendo tiempo de inicialización del modelo...")
        start = time.perf_counter()
        self.ompar = OMPAR(model_path=model_path, device=device, args=args)
        init_time = time.perf_counter() - start
        
        self.results['initialization_time_ms'] = init_time * 1000
        print(f"✅ Inicialización: {init_time*1000:.2f} ms\n")
    
    def _get_system_info(self) -> Dict:
        """Obtener información del sistema"""
        info = {
            'cpu': {
                'model': 'Unknown',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            }
        }
        
        # Información de GPU si está disponible
        if self.device == 'cuda':
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info['gpu'] = {
                        'name': gpu.name,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_free_mb': gpu.memoryFree,
                        'driver': gpu.driver,
                        'cuda_version': torch.version.cuda
                    }
            except:
                info['gpu'] = {'name': 'CUDA available but details unavailable'}
        
        return info
    
    def _measure_component_times(self, code: str) -> Dict:
        """Medir tiempos de cada componente del pipeline"""
        times = {}
        
        # 1. Tiempo de clasificación (OMPify)
        start = time.perf_counter()
        pragma_cls = self.ompar.cls_par(code)
        times['classification_ms'] = (time.perf_counter() - start) * 1000
        
        # 2. Si es paralelizable, medir generación de pragma
        if pragma_cls:
            start = time.perf_counter()
            pragma = self.ompar.gen_par(code)
            times['generation_ms'] = (time.perf_counter() - start) * 1000
            
            # 3. Tiempo de formateo
            start = time.perf_counter()
            formatted = self.ompar.pragma_format(pragma)
            times['formatting_ms'] = (time.perf_counter() - start) * 1000
        else:
            times['generation_ms'] = 0
            times['formatting_ms'] = 0
        
        # Tiempo total
        times['total_ms'] = sum(times.values())
        
        return times
    
    def benchmark_single(self, code: str, name: str, iterations: int = 10) -> Dict:
        """Benchmark de un solo snippet de código"""
        print(f"📊 Benchmarking: {name}")
        print(f"   Iteraciones: {iterations}")
        
        all_times = []
        component_times = {
            'classification_ms': [],
            'generation_ms': [],
            'formatting_ms': [],
            'total_ms': []
        }
        
        # Warm-up (no contar la primera ejecución)
        print("   🔥 Warm-up...", end='', flush=True)
        self.ompar.auto_comp(code)
        print(" ✓")
        
        # Ejecutar múltiples iteraciones
        print("   ⏱️  Ejecutando iteraciones...", end='', flush=True)
        for i in range(iterations):
            start = time.perf_counter()
            result = self.ompar.auto_comp(code)
            elapsed = (time.perf_counter() - start) * 1000
            all_times.append(elapsed)
            
            # Medir componentes individuales
            comp_times = self._measure_component_times(code)
            for key, value in comp_times.items():
                component_times[key].append(value)
            
            if (i + 1) % 5 == 0:
                print(f" {i+1}", end='', flush=True)
        
        print(" ✓")
        
        # Calcular estadísticas
        stats = {
            'name': name,
            'iterations': iterations,
            'code_length': len(code),
            'times': {
                'mean_ms': statistics.mean(all_times),
                'median_ms': statistics.median(all_times),
                'min_ms': min(all_times),
                'max_ms': max(all_times),
                'stdev_ms': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                'p95_ms': sorted(all_times)[int(len(all_times) * 0.95)],
                'p99_ms': sorted(all_times)[int(len(all_times) * 0.99)]
            },
            'components': {
                key: {
                    'mean_ms': statistics.mean(values),
                    'min_ms': min(values),
                    'max_ms': max(values)
                }
                for key, values in component_times.items()
            },
            'throughput': {
                'iterations_per_second': 1000 / statistics.mean(all_times),
                'ms_per_iteration': statistics.mean(all_times)
            }
        }
        
        # Mostrar resultados
        print(f"   ✅ Media: {stats['times']['mean_ms']:.2f} ms")
        print(f"   📈 Min/Max: {stats['times']['min_ms']:.2f} / {stats['times']['max_ms']:.2f} ms")
        print(f"   📊 Desv. Est.: {stats['times']['stdev_ms']:.2f} ms")
        print(f"   🚀 Throughput: {stats['throughput']['iterations_per_second']:.2f} iter/s\n")
        
        return stats
    
    def benchmark_batch(self, codes: List[Tuple[str, str]], batch_size: int = 10) -> Dict:
        """Benchmark de procesamiento por lotes"""
        print(f"📦 Benchmarking procesamiento por lotes")
        print(f"   Tamaño del lote: {batch_size}")
        print(f"   Total de snippets: {len(codes)}")
        
        start = time.perf_counter()
        results = []
        
        for i, (code, name) in enumerate(codes[:batch_size]):
            pragma = self.ompar.auto_comp(code)
            results.append(pragma)
            
            if (i + 1) % 5 == 0:
                print(f"   Procesados: {i+1}/{batch_size}", end='\r', flush=True)
        
        total_time = (time.perf_counter() - start) * 1000
        
        stats = {
            'batch_size': batch_size,
            'total_time_ms': total_time,
            'avg_time_per_item_ms': total_time / batch_size,
            'throughput_items_per_second': (batch_size / total_time) * 1000
        }
        
        print(f"\n   ✅ Tiempo total: {stats['total_time_ms']:.2f} ms")
        print(f"   📊 Promedio por item: {stats['avg_time_per_item_ms']:.2f} ms")
        print(f"   🚀 Throughput: {stats['throughput_items_per_second']:.2f} items/s\n")
        
        return stats
    
    def benchmark_memory(self, code: str) -> Dict:
        """Medir uso de memoria durante la inferencia"""
        print("💾 Benchmarking uso de memoria...")
        
        # Memoria antes
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            mem_before = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Ejecutar inferencia
        _ = self.ompar.auto_comp(code)
        
        # Memoria después
        if self.device == 'cuda':
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            mem_after = psutil.Process().memory_info().rss / (1024**2)
            mem_peak = mem_after
        
        stats = {
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_peak_mb': mem_peak,
            'memory_increase_mb': mem_after - mem_before
        }
        
        print(f"   📊 Memoria antes: {stats['memory_before_mb']:.2f} MB")
        print(f"   📊 Memoria después: {stats['memory_after_mb']:.2f} MB")
        print(f"   📈 Pico de memoria: {stats['memory_peak_mb']:.2f} MB")
        print(f"   ➕ Incremento: {stats['memory_increase_mb']:.2f} MB\n")
        
        return stats
    
    def save_results(self, filename: str = None):
        """Guardar resultados en JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Resultados guardados en: {filepath}")
        return filepath
    
    def compare_with_baseline(self, baseline_file: str):
        """Comparar con resultados baseline"""
        if not Path(baseline_file).exists():
            print(f"⚠️  Archivo baseline no encontrado: {baseline_file}")
            return
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        print("\n" + "="*80)
        print("📊 COMPARACIÓN CON BASELINE")
        print("="*80)
        
        for i, (current, base) in enumerate(zip(
            self.results['benchmarks'], 
            baseline.get('benchmarks', [])
        )):
            if current['name'] != base['name']:
                continue
            
            current_time = current['times']['mean_ms']
            base_time = base['times']['mean_ms']
            speedup = base_time / current_time
            improvement = ((base_time - current_time) / base_time) * 100
            
            print(f"\n{current['name']}:")
            print(f"  Baseline:  {base_time:.2f} ms")
            print(f"  Actual:    {current_time:.2f} ms")
            print(f"  Speedup:   {speedup:.2f}x")
            print(f"  Mejora:    {improvement:+.1f}%")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark de rendimiento de OMPar')
    parser.add_argument('--vocab_file', default='tokenizer/gpt/gpt_vocab/gpt2-vocab.json')
    parser.add_argument('--merge_file', default='tokenizer/gpt/gpt_vocab/gpt2-merges.txt')
    parser.add_argument('--model_weights', default='model')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Número de iteraciones por benchmark')
    parser.add_argument('--output', default=None, 
                       help='Archivo de salida para resultados')
    parser.add_argument('--baseline', default=None,
                       help='Archivo baseline para comparación')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Guardar resultados como baseline')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("🚀 BENCHMARK DE RENDIMIENTO - OMPar")
    print("="*80)
    print(f"Dispositivo: {device}")
    print(f"Iteraciones: {args.iterations}")
    print("="*80 + "\n")
    
    # Inicializar benchmark
    benchmark = PerformanceBenchmark(args.model_weights, device, args)
    
    # Casos de prueba
    test_cases = [
        ('Simple loop', '''for (int i = 0; i < n; i++) {
    arr[i] = 0;
}'''),
        ('Reduction', '''for (int i = 0; i < n; i++) {
    total += arr[i];
}'''),
        ('Array copy', '''for (int i = 0; i < n; i++) {
    dest[i] = src[i];
}'''),
        ('Element-wise operation', '''for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i] * c[i];
}'''),
        ('Complex operation', '''for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        matrix[i][j] = matrix[i][j] * scalar + offset;
    }
}'''),
    ]
    
    # Ejecutar benchmarks individuales
    print("\n" + "="*80)
    print("📊 BENCHMARKS INDIVIDUALES")
    print("="*80 + "\n")
    
    for name, code in test_cases:
        result = benchmark.benchmark_single(code, name, args.iterations)
        benchmark.results['benchmarks'].append(result)
    
    # Benchmark de lotes
    print("\n" + "="*80)
    print("📦 BENCHMARK DE PROCESAMIENTO POR LOTES")
    print("="*80 + "\n")
    
    batch_result = benchmark.benchmark_batch(test_cases, len(test_cases))
    benchmark.results['batch_benchmark'] = batch_result
    
    # Benchmark de memoria
    print("\n" + "="*80)
    print("💾 BENCHMARK DE MEMORIA")
    print("="*80 + "\n")
    
    memory_result = benchmark.benchmark_memory(test_cases[0][1])
    benchmark.results['memory_benchmark'] = memory_result
    
    # Guardar resultados
    print("\n" + "="*80)
    print("💾 GUARDANDO RESULTADOS")
    print("="*80 + "\n")
    
    if args.save_baseline:
        output_file = 'benchmark_baseline.json'
    else:
        output_file = args.output
    
    saved_file = benchmark.save_results(output_file)
    
    # Comparar con baseline si existe
    if args.baseline:
        benchmark.compare_with_baseline(args.baseline)
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    avg_time = statistics.mean([b['times']['mean_ms'] for b in benchmark.results['benchmarks']])
    print(f"\n✅ Tiempo promedio de inferencia: {avg_time:.2f} ms")
    print(f"✅ Throughput promedio: {1000/avg_time:.2f} inferencias/segundo")
    print(f"✅ Memoria pico: {memory_result['memory_peak_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    main()
