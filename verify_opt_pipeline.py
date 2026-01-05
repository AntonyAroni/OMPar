import torch
import argparse
import sys
import os

# Mock args
class Args:
    vocab_file = 'tokenizer/gpt/gpt_vocab/gpt2-vocab.json'
    merge_file = 'tokenizer/gpt/gpt_vocab/gpt2-merges.txt'

args = Args()

print("1. Cargando OMPAR con TensorRT...")
from compAI import OMPAR
ompar = OMPAR(model_path='model', device='cuda', args=args, use_tensorrt=True)

test_code = "for (int i = 0; i < n; i++) { total += arr[i]; }"

print("\n2. Ejecutando auto_comp...")
res = ompar.auto_comp(test_code)

print(f"\n3. Resultado: {res}")

if res and "omp parallel" in res:
    print("✅ Verificación Exitosa!")
else:
    print("❌ Verificación Fallida")
