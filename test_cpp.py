
import torch
import monocoder_trt_cpp
import numpy as np

print("🚀 Loading MonoCoderTRT C++")
engine_path = "cpp_extensions/monocoder_tensorrt/monocoder_fixed.engine"
model = monocoder_trt_cpp.MonoCoderTRT(engine_path)

print("✅ Loaded.")

input_ids = [1, 2, 3, 4] # Dummy input
stream = torch.cuda.current_stream().cuda_stream

print("👉 Calling forward...")
try:
    logits = model.forward(input_ids, stream)
    print(f"✅ Forward success, logits len: {len(logits)}")
except Exception as e:
    print(f"❌ Forward failed: {e}")

print("👉 Calling generate...")
try:
    output = model.generate(input_ids, 64, stream)
    print(f"✅ Generate success, output: {output}")
except Exception as e:
    print(f"❌ Generate failed: {e}")
