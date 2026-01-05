#!/usr/bin/env python3
"""
Benchmark Comparativo: PyTorch FP16 vs TensorRT
"""

import time
import torch
import sys
import os
sys.path.append(os.getcwd())

from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from cpp_extensions.monocoder_tensorrt.monocoder_trt import MonoCoderTRT

print("="*70)
print("🚀 BENCHMARK: PYTORCH FP16 vs TENSORRT")
print("="*70)

# 1. Setup PyTorch Baseline
print("\n📦 Cargando PyTorch FP16...")
tokenizer = GPT2Tokenizer(
    vocab_file='tokenizer/gpt/gpt_vocab/gpt2-vocab.json',
    merges_file='tokenizer/gpt/gpt_vocab/gpt2-merges.txt',
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

pt_model = GPTNeoXForCausalLM.from_pretrained("MonoCoder/MonoCoder_OMP", use_safetensors=True)
pt_model.eval()
pt_model.half()
pt_model.cuda()

# 2. Setup TensorRT
print("\n📦 Cargando TensorRT...")
try:
    trt_model = MonoCoderTRT("cpp_extensions/monocoder_tensorrt/monocoder_fixed.engine")
except Exception as e:
    print(f"❌ Error cargando TRT: {e}")
    trt_model = None

# Test Data
test_code = "for (int i = 0; i < n; i++) { total += arr[i]; }"
inputs = tokenizer(test_code, return_tensors="pt")
input_ids = inputs["input_ids"].cuda()
input_list = inputs["input_ids"][0].tolist()

ITERATIONS = 10

def benchmark_pt():
    times = []
    print("\n⏱️  Running PyTorch...")
    with torch.no_grad():
        for _ in range(ITERATIONS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = pt_model.generate(input_ids, max_length=64, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return sum(times)/len(times)

def benchmark_trt():
    times = []
    print("\n⏱️  Running TensorRT...")
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = trt_model.generate(input_list, max_length=64)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return sum(times)/len(times)

# Run
pt_avg = benchmark_pt()
print(f"   PyTorch Avg: {pt_avg:.2f} ms")

if trt_model:
    trt_avg = benchmark_trt()
    print(f"   TensorRT Avg: {trt_avg:.2f} ms")
    
    print("\n" + "="*70)
    print("📊 RESULTADOS")
    print("="*70)
    print(f"PyTorch FP16: {pt_avg:.2f} ms")
    print(f"TensorRT:     {trt_avg:.2f} ms")
    print(f"Speedup:      {pt_avg/trt_avg:.2f}x")
else:
    print("Skipping TRT benchmark")
