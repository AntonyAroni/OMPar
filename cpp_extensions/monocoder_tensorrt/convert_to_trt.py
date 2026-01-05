#!/usr/bin/env python3
"""
Paso 1: Convertir MonoCoder a TensorRT Engine (Fixed Shape)
Usa forma fija (1, 64) para evitar problemas de tracing dinámico
"""

import torch
import tensorrt as trt
import numpy as np
from pathlib import Path
import time
import os

print("="*70)
print("🚀 CONVERTIR MONOCODER A TENSORRT (FIXED SHAPE)")
print("="*70)
print(f"TensorRT version: {trt.__version__}")

from transformers import GPTNeoXForCausalLM, GPT2Tokenizer

# Configuración
MODEL_NAME = "MonoCoder/MonoCoder_OMP"
OUTPUT_DIR = Path("cpp_extensions/monocoder_tensorrt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUTPUT_DIR / "monocoder_fixed.onnx"
TRT_PATH = OUTPUT_DIR / "monocoder_fixed.engine"
FIXED_LEN = 64

# Cargar modelo
print("\n📦 Cargando modelo PyTorch...")
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
model.eval()
model.half()  # FP16
model.cuda()
print(f"   ✅ Modelo cargado en CUDA (FP16)")

# Tokenizer
tokenizer = GPT2Tokenizer(
    vocab_file='tokenizer/gpt/gpt_vocab/gpt2-vocab.json',
    merges_file='tokenizer/gpt/gpt_vocab/gpt2-merges.txt',
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Wrapper
class MonoCoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        # Asegura compatibilidad eliminando names dinámicos
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

wrapped_model = MonoCoderWrapper(model)
wrapped_model.eval()

# Paso 1: Exportar a ONNX con forma fija
print(f"\n📤 Exportando a ONNX (Fixed: {FIXED_LEN})...")
# Crear input fijo de 64 tokens
sample_input = torch.randint(0, 50000, (1, FIXED_LEN), dtype=torch.long).cuda()

# Eliminar ONNX anterior si existe para evitar conflictos
if ONNX_PATH.exists():
    os.remove(ONNX_PATH)

torch.onnx.export(
    wrapped_model,
    sample_input,
    str(ONNX_PATH),
    input_names=['input_ids'],
    output_names=['logits'],
    # NO dynamic_axes para forma fija
    opset_version=17,
    do_constant_folding=True,
)
print(f"   ✅ ONNX exportado: {ONNX_PATH}")

# Liberar memoria
del model
del wrapped_model
torch.cuda.empty_cache()

# Paso 2: Convertir ONNX a TensorRT
print("\n🔧 Convirtiendo a TensorRT...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine():
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    # Memoria: 4GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("   ✅ FP16 habilitado")
    
    # Parsear ONNX
    print("   📖 Parseando ONNX...")
    if not parser.parse_from_file(str(ONNX_PATH)):
        for error in range(parser.num_errors):
            print(f"   ❌ Error: {parser.get_error(error)}")
        return None
    
    # No optimization profile needed for fixed shape (implicit)
    
    # Construir
    print("   🔨 Construyendo engine TensorRT...")
    serialized_engine = builder.build_serialized_network(network, config)
    return serialized_engine

if TRT_PATH.exists():
    os.remove(TRT_PATH)

engine_bytes = build_engine()

if engine_bytes:
    with open(TRT_PATH, 'wb') as f:
        f.write(engine_bytes)
    print(f"\n   ✅ Engine TensorRT guardado: {TRT_PATH}")
else:
    print("   ❌ Falló la construcción del engine")
    exit(1)

# Paso 3: Probar el engine
print("\n🧪 Probando engine TensorRT (con padding)...")

try:
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(TRT_PATH, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Preparar input con padding
    test_code = "for (int i = 0; i < n; i++) { total += arr[i]; }"
    inputs = tokenizer(test_code, return_tensors="np")
    original_ids = inputs["input_ids"]
    
    # Padding a FIXED_LEN
    input_ids = np.full((1, FIXED_LEN), tokenizer.pad_token_id, dtype=np.int32)
    seq_len = min(original_ids.shape[1], FIXED_LEN)
    input_ids[0, :seq_len] = original_ids[0, :seq_len]
    
    # Inferencia
    d_input = torch.from_numpy(input_ids).cuda()
    d_output = torch.zeros((1, FIXED_LEN, 50688), dtype=torch.float16).cuda()
    
    bindings = [d_input.data_ptr(), d_output.data_ptr()]
    
    # Warm-up
    context.execute_v2(bindings)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        context.execute_v2(bindings)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"   ✅ Inferencia exitosa!")
    print(f"   ⏱️  Tiempo promedio: {avg_time:.2f} ms")
    print(f"   🚀 Throughput: {1000/avg_time:.0f} inferencias/segundo")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ COMPLETADO")
print("="*70)
