import tensorrt as trt
import torch
import numpy as np
import os
import time

class MonoCoderTRT:
    def __init__(self, engine_path="cpp_extensions/monocoder_tensorrt/monocoder_fixed.engine"):
        self.logger = trt.Logger(trt.Logger.ERROR) # Changed to ERROR to see if anything pops up
        self.runtime = trt.Runtime(self.logger)
        
        print(f"📦 Cargando TensorRT engine: {engine_path}")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine no encontrado: {engine_path}")
            
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Falló deserialize_cuda_engine (return None)")
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # Fixed config
        self.FIXED_LEN = 64
        self.VOCAB_SIZE = 50688
        
        # Buffer allocation
        # Input: (1, 64) int32
        # Output: (1, 64, VOCAB_SIZE) float16
        
        self.d_input = torch.zeros((1, self.FIXED_LEN), dtype=torch.int32, device='cuda')
        self.d_output = torch.zeros((1, self.FIXED_LEN, self.VOCAB_SIZE), dtype=torch.float16, device='cuda')
        
        self.bindings = [int(self.d_input.data_ptr()), int(self.d_output.data_ptr())]
        
        # Warmup
        self.forward(torch.zeros((1, self.FIXED_LEN), dtype=torch.int32, device='cuda'))
        print("✅ MonoCoder TRT inicializado")

    def forward(self, input_ids):
        """
        Inferencia simple.
        input_ids: torch.Tensor (1, seq_len) o (1, FIXED_LEN) int32 cuda
        """
        # Padding si es necesario
        seq_len = input_ids.shape[1]
        
        if seq_len < self.FIXED_LEN:
            self.d_input.fill_(50256) # Pad token
            self.d_input[0, :seq_len] = input_ids[0]
        else:
            self.d_input.copy_(input_ids)
            
            
        # Inferencia síncrona
        self.context.execute_v2(bindings=self.bindings)
        # self.stream.synchronize() # No needed for execute_v2 if synchronous
        
        return self.d_output

    def generate(self, input_ids_list, max_length=64):
        """
        Generación greedy.
        input_ids_list: list of ints
        Returns: list of ints (generated tokens only)
        """
        current_ids = list(input_ids_list)
        original_len = len(current_ids)
        
        with torch.no_grad():
            while len(current_ids) < max_length:
                if len(current_ids) >= self.FIXED_LEN:
                    break
                
                # Preparar input tensor
                curr_tensor = torch.tensor([current_ids], dtype=torch.int32, device='cuda')
                
                # Forward
                logits = self.forward(curr_tensor)
                
                # Obtener logits del último token válido
                last_idx = len(current_ids) - 1
                next_token_logits = logits[0, last_idx, :]
                
                # Argmax
                next_token = torch.argmax(next_token_logits).item()
                
                if next_token == 50256: # EOS
                    break
                    
                current_ids.append(next_token)
        
        return current_ids[original_len:]
