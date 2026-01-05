#!/usr/bin/env python3
"""
Sistema de Benchmarking SIMPLE para OMPar (sin dependencias extra)
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
from compAI import OMPAR


class SimpleBenchmark:
    def __init__(self, model_path: str, device: str, args, use_tensorrt=False):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'benchmarks': []
        }
        
        # Medir tiempo de inicialización
        print("⏱️  Midiendo tiempo de inicialización del modelo...")
        start = time.perf_counter()
        self.ompar = OMPAR(model_path=model_path, device=device, args=args, use_tensorrt=use_tensorrt)
        init_time = time.perf_counter() - start
        
        self.results['initialization_time_ms'] = init_time * 1000
        print(f"✅ Inicialización: {init_time*1000:.2f} ms\n")
    
    def _measure_component_times(self, code: str) -> Dict:
        """Medir tiempos de cada componente del pipeline"""
        times = {}
        
        try:
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
        except Exception as e:
            print(f"      ⚠️  Error en medición: {str(e)[:50]}")
            times = {
                'classification_ms': 0,
                'generation_ms': 0,
                'formatting_ms': 0,
                'total_ms': 0
            }
        
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
        try:
            self.ompar.auto_comp(code)
            print(" ✓")
        except Exception as e:
            print(f" ⚠️  Error: {str(e)[:30]}")
            return None
        
        # Ejecutar múltiples iteraciones
        print("   ⏱️  Ejecutando iteraciones...", end='', flush=True)
        successful_iters = 0
        for i in range(iterations):
            try:
                start = time.perf_counter()
                result = self.ompar.auto_comp(code)
                elapsed = (time.perf_counter() - start) * 1000
                all_times.append(elapsed)
                
                # Medir componentes individuales
                comp_times = self._measure_component_times(code)
                for key, value in comp_times.items():
                    component_times[key].append(value)
                
                successful_iters += 1
                if (i + 1) % 5 == 0:
                    print(f" {i+1}", end='', flush=True)
            except Exception as e:
                print(f" E{i+1}", end='', flush=True)
                continue
        
        print(" ✓")
        
        if not all_times:
            print("   ❌ No se pudieron completar iteraciones")
            return None
        
        # Calcular estadísticas
        stats = {
            'name': name,
            'iterations': successful_iters,
            'code_length': len(code),
            'times': {
                'mean_ms': statistics.mean(all_times),
                'median_ms': statistics.median(all_times),
                'min_ms': min(all_times),
                'max_ms': max(all_times),
                'stdev_ms': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                'p95_ms': sorted(all_times)[int(len(all_times) * 0.95)] if len(all_times) > 1 else all_times[0],
                'p99_ms': sorted(all_times)[int(len(all_times) * 0.99)] if len(all_times) > 1 else all_times[0]
            },
            'components': {
                key: {
                    'mean_ms': statistics.mean(values) if values else 0,
                    'min_ms': min(values) if values else 0,
                    'max_ms': max(values) if values else 0
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
        print(f"   🚀 Throughput: {stats['throughput']['iterations_per_second']:.2f} iter/s")
        
        # Mostrar breakdown de componentes
        print(f"   📦 Componentes:")
        print(f"      - Clasificación: {stats['components']['classification_ms']['mean_ms']:.2f} ms")
        print(f"      - Generación:    {stats['components']['generation_ms']['mean_ms']:.2f} ms")
        print(f"      - Formateo:      {stats['components']['formatting_ms']['mean_ms']:.2f} ms")
        print()
        
        return stats
    
    def benchmark_batch(self, codes: List[Tuple[str, str]], batch_size: int = 10) -> Dict:
        """Benchmark de procesamiento por lotes"""
        print(f"📦 Benchmarking procesamiento por lotes")
        print(f"   Tamaño del lote: {batch_size}")
        print(f"   Total de snippets: {len(codes)}")
        
        start = time.perf_counter()
        results = []
        successful = 0
        
        for i, (code, name) in enumerate(codes[:batch_size]):
            try:
                pragma = self.ompar.auto_comp(code)
                results.append(pragma)
                successful += 1
            except:
                results.append(None)
            
            if (i + 1) % 5 == 0:
                print(f"   Procesados: {i+1}/{batch_size}", end='\r', flush=True)
        
        total_time = (time.perf_counter() - start) * 1000
        
        stats = {
            'batch_size': batch_size,
            'successful': successful,
            'total_time_ms': total_time,
            'avg_time_per_item_ms': total_time / successful if successful > 0 else 0,
            'throughput_items_per_second': (successful / total_time) * 1000 if total_time > 0 else 0
        }
        
        print(f"\n   ✅ Tiempo total: {stats['total_time_ms']:.2f} ms")
        print(f"   📊 Exitosos: {successful}/{batch_size}")
        print(f"   📊 Promedio por item: {stats['avg_time_per_item_ms']:.2f} ms")
        print(f"   🚀 Throughput: {stats['throughput_items_per_second']:.2f} items/s\n")
        
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
        
        # Tabla de comparación
        print(f"\n{'Test':<30} {'Baseline':>12} {'Actual':>12} {'Speedup':>10} {'Mejora':>10}")
        print("-" * 80)
        
        total_speedup = []
        
        for current in self.results['benchmarks']:
            if current is None:
                continue
                
            # Buscar el test correspondiente en baseline
            base = None
            for b in baseline.get('benchmarks', []):
                if b and b['name'] == current['name']:
                    base = b
                    break
            
            if not base:
                continue
            
            current_time = current['times']['mean_ms']
            base_time = base['times']['mean_ms']
            speedup = base_time / current_time
            improvement = ((base_time - current_time) / base_time) * 100
            
            total_speedup.append(speedup)
            
            print(f"{current['name']:<30} {base_time:>10.2f} ms {current_time:>10.2f} ms "
                  f"{speedup:>9.2f}x {improvement:>+9.1f}%")
        
        print("-" * 80)
        avg_speedup = statistics.mean(total_speedup) if total_speedup else 1.0
        print(f"{'PROMEDIO':<30} {'':<12} {'':<12} {avg_speedup:>9.2f}x")
        print("="*80)


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
    parser.add_argument('--use_tensorrt', action='store_true',
                        help='Usar optimización TensorRT')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Guardar resultados como baseline')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("🚀 BENCHMARK DE RENDIMIENTO - OMPar")
    print("="*80)
    print(f"Dispositivo: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Iteraciones: {args.iterations}")
    print(f"TensorRT: {'Activado' if args.use_tensorrt else 'Desactivado'}")
    print("="*80 + "\n")
    
    # Inicializar benchmark
    benchmark = SimpleBenchmark(args.model_weights, device, args, use_tensorrt=args.use_tensorrt)
    
    # Casos de prueba (solo códigos simples que funcionan)
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
        ('Element-wise', '''for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i];
}'''),
    ]
    
    # Ejecutar benchmarks individuales
    print("\n" + "="*80)
    print("📊 BENCHMARKS INDIVIDUALES")
    print("="*80 + "\n")
    
    for name, code in test_cases:
        result = benchmark.benchmark_single(code, name, args.iterations)
        if result:
            benchmark.results['benchmarks'].append(result)
    
    # Benchmark de lotes
    print("\n" + "="*80)
    print("📦 BENCHMARK DE PROCESAMIENTO POR LOTES")
    print("="*80 + "\n")
    
    batch_result = benchmark.benchmark_batch(test_cases, len(test_cases))
    benchmark.results['batch_benchmark'] = batch_result
    
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
    
    valid_benchmarks = [b for b in benchmark.results['benchmarks'] if b is not None]
    
    if valid_benchmarks:
        avg_time = statistics.mean([b['times']['mean_ms'] for b in valid_benchmarks])
        avg_classification = statistics.mean([
            b['components']['classification_ms']['mean_ms'] 
            for b in valid_benchmarks
        ])
        avg_generation = statistics.mean([
            b['components']['generation_ms']['mean_ms'] 
            for b in valid_benchmarks
        ])
        
        print(f"\n✅ Tiempo promedio total: {avg_time:.2f} ms")
        print(f"   - Clasificación (OMPify): {avg_classification:.2f} ms ({avg_classification/avg_time*100:.1f}%)")
        print(f"   - Generación (MonoCoder): {avg_generation:.2f} ms ({avg_generation/avg_time*100:.1f}%)")
        print(f"\n✅ Throughput promedio: {1000/avg_time:.2f} inferencias/segundo")
        print(f"✅ Tiempo de inicialización: {benchmark.results['initialization_time_ms']:.2f} ms")
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETADO")
    print("="*80)


if __name__ == '__main__':
    main()
