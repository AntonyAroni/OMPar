#!/usr/bin/env python3
"""
Fase 2: Benchmark de MonoCoder con Optimizaciones PyTorch
Opciones: torch.compile, half precision (FP16), flash attention
"""

import time
import statistics
import torch
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer

print("="*70)
print("🚀 BENCHMARK MONOCODER: OPTIMIZACIONES PYTORCH")
print("="*70)

# Configuración
MODEL_PATH = "MonoCoder/MonoCoder_OMP"
ITERATIONS = 5
MAX_LENGTH = 64

# Verificar GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️  Device: {device}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Tokenizer
tokenizer = GPT2Tokenizer(
    vocab_file='tokenizer/gpt/gpt_vocab/gpt2-vocab.json',
    merges_file='tokenizer/gpt/gpt_vocab/gpt2-merges.txt',
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Código de prueba
test_code = "for (int i = 0; i < n; i++) { total += arr[i]; }"
inputs = tokenizer(test_code, return_tensors="pt")

def benchmark_model(model, name, inputs, device, iterations=5):
    """Benchmark de generación"""
    input_ids = inputs["input_ids"].to(device)
    
    # Warm-up (importante para GPU)
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(
                input_ids,
                max_length=MAX_LENGTH,
                pad_token_id=tokenizer.pad_token_id
            )
    
    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            outputs = model.generate(
                input_ids,
                max_length=MAX_LENGTH,
                pad_token_id=tokenizer.pad_token_id
            )
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
    
    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pragma = output_text[len(test_code):]
    
    return {
        'name': name,
        'avg_ms': avg,
        'std_ms': std,
        'min_ms': min(times),
        'max_ms': max(times),
        'pragma': pragma[:60]
    }

results = []

# ===== 1. Baseline: PyTorch FP32 =====
print(f"\n🔧 Cargando modelo base (FP32)...")
model_fp32 = GPTNeoXForCausalLM.from_pretrained(MODEL_PATH, use_safetensors=True)
model_fp32.eval()
model_fp32.to(device)

print(f"   Parámetros: {sum(p.numel() for p in model_fp32.parameters()):,}")
print(f"   Memoria GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB" if device == 'cuda' else "")

print(f"\n⏱️  Benchmarking FP32...")
result = benchmark_model(model_fp32, "PyTorch FP32", inputs, device, ITERATIONS)
results.append(result)
print(f"   Promedio: {result['avg_ms']:.2f} ms (±{result['std_ms']:.2f})")

# ===== 2. Half Precision (FP16) =====
print(f"\n🔧 Convirtiendo a FP16 (Half Precision)...")
model_fp16 = model_fp32.half()  # Convertir in-place
torch.cuda.empty_cache()

print(f"   Memoria GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB" if device == 'cuda' else "")

print(f"\n⏱️  Benchmarking FP16...")
result = benchmark_model(model_fp16, "PyTorch FP16", inputs, device, ITERATIONS)
results.append(result)
print(f"   Promedio: {result['avg_ms']:.2f} ms (±{result['std_ms']:.2f})")

# Liberar modelo
del model_fp16
del model_fp32
torch.cuda.empty_cache()

# ===== 3. BetterTransformer (si disponible) =====
print(f"\n🔧 Probando BetterTransformer...")
try:
    from optimum.bettertransformer import BetterTransformer
    
    model_bt = GPTNeoXForCausalLM.from_pretrained(MODEL_PATH, use_safetensors=True)
    model_bt.eval()
    model_bt = BetterTransformer.transform(model_bt)
    model_bt.to(device)
    model_bt = model_bt.half()
    
    print(f"   ✅ BetterTransformer activado")
    
    print(f"\n⏱️  Benchmarking BetterTransformer FP16...")
    result = benchmark_model(model_bt, "BetterTransformer FP16", inputs, device, ITERATIONS)
    results.append(result)
    print(f"   Promedio: {result['avg_ms']:.2f} ms (±{result['std_ms']:.2f})")
    
    del model_bt
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"   ⚠️  BetterTransformer no disponible: {e}")

# ===== 4. torch.compile (PyTorch 2.0+) =====
print(f"\n🔧 Probando torch.compile...")
if hasattr(torch, 'compile'):
    try:
        model_compiled = GPTNeoXForCausalLM.from_pretrained(MODEL_PATH, use_safetensors=True)
        model_compiled.eval()
        model_compiled.to(device)
        model_compiled = model_compiled.half()
        
        # Compilar
        model_compiled = torch.compile(model_compiled, mode="reduce-overhead")
        
        print(f"   ✅ torch.compile activado (reduce-overhead)")
        
        # Más warm-up para compilación
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model_compiled.generate(
                    input_ids,
                    max_length=MAX_LENGTH,
                    pad_token_id=tokenizer.pad_token_id
                )
        
        print(f"\n⏱️  Benchmarking torch.compile FP16...")
        result = benchmark_model(model_compiled, "torch.compile FP16", inputs, device, ITERATIONS)
        results.append(result)
        print(f"   Promedio: {result['avg_ms']:.2f} ms (±{result['std_ms']:.2f})")
        
        del model_compiled
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ⚠️  torch.compile falló: {e}")
else:
    print(f"   ⚠️  torch.compile no disponible (requiere PyTorch 2.0+)")

# ===== Resumen =====
print("\n" + "="*70)
print("📊 RESUMEN")
print("="*70)

baseline = 462.0  # ms del baseline original (MonoCoder CPU)

print(f"\n{'Configuración':<30} {'Tiempo (ms)':>12} {'vs Baseline':>15}")
print("-"*60)

for r in results:
    speedup = baseline / r['avg_ms']
    print(f"{r['name']:<30} {r['avg_ms']:>12.2f} {f'{speedup:.2f}x':>15}")

print("-"*60)
print(f"{'Baseline (original)':<30} {baseline:>12.2f} {'1.00x':>15}")

# Mejor resultado
best = min(results, key=lambda x: x['avg_ms'])
best_speedup = baseline / best['avg_ms']

print(f"\n🏆 Mejor configuración: {best['name']}")
print(f"   Tiempo: {best['avg_ms']:.2f} ms")
print(f"   Speedup: {best_speedup:.2f}x vs baseline")
print(f"   Mejora: {((baseline - best['avg_ms']) / baseline) * 100:.1f}%")

print(f"\n📝 Pragma generado: {best['pragma']}...")

print("\n" + "="*70)
print("✅ BENCHMARK COMPLETADO")
print("="*70)
