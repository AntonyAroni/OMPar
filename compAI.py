import torch
import argparse
from OMPify.model import OMPify
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer


class OMPAR:

    def __init__(self, model_path, device, args, use_fp16=True, use_tensorrt=False):
        """
        Inicializar OMPAR con optimizaciones opcionales.
        
        Args:
            model_path: Ruta al modelo OMPify
            device: 'cuda' o 'cpu'
            args: Argumentos con vocab_file y merge_file
            use_fp16: Usar half precision (1.8x speedup)
            use_tensorrt: Usar TensorRT engine (requiere engine compilado)
        """
        self.device = device
        self.use_fp16 = use_fp16 and device == 'cuda'
        self.use_tensorrt = use_tensorrt and device == 'cuda'
        
        self.model_cls = OMPify(model_path, device)
        self.tokenizer_gen = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, model_input_names=['input_ids'])

        if self.use_tensorrt:
            try:
                from cpp_extensions.monocoder_tensorrt.monocoder_trt import MonoCoderTRT
                self.model_gen = MonoCoderTRT("cpp_extensions/monocoder_tensorrt/monocoder_fixed.engine")
                print("✅ MonoCoder: TensorRT activado (Máximo rendimiento)")
            except Exception as e:
                print(f"⚠️  Error cargando TensorRT: {e}. Usando PyTorch...")
                self.use_tensorrt = False
                
        if not self.use_tensorrt:
            self.model_gen = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP', use_safetensors=True).to(device)
            # Aplicar optimización FP16 si está habilitada
            if self.use_fp16:
                self.model_gen = self.model_gen.half()
                print("✅ MonoCoder: FP16 activado (14% más rápido)")
            self.model_gen.eval()

    def cls_par(self, loop) -> bool:
        """
        Return if a parallelization is aplicable/neccessary
        """
        pragma_cls, _, _ = self.model_cls.predict(loop)
        return pragma_cls
    
    def pragma_format(self, pragma):
        clauses = pragma.split('||')        
        private_vars = None
        reduction_op, reduction_vars = None, None

        for clause in clauses:
            cl = clause.strip()

            if private_vars is None and cl.startswith('private'):
                private_vars = cl[len('private'):].split()
                
            if reduction_vars is None and cl.startswith('reduction'):
                reduction = cl[len('reduction'):].split(':')
                
                if len(reduction) >=2:
                    reduction_op = reduction[0]
                    reduction_vars = reduction[1].split()

        pragma = 'omp parallel for'
        if private_vars is not None and len(private_vars) > 0:
            pragma += f" private({', '.join(private_vars)})"
        if reduction_vars is not None and len(reduction_vars) > 0:
            pragma += f" reduction({reduction_op}:{', '.join(reduction_vars)})"

        return pragma        

    def gen_par(self, loop) -> str:
        """
        Generate OMP pragma
        """
    def gen_par(self, loop) -> str:
        """
        Generate OMP pragma
        """
        inputs = self.tokenizer_gen(loop, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if self.use_tensorrt:
            # TensorRT expects list of ints
            input_list = input_ids[0].tolist()
            output_ids = self.model_gen.generate(input_list, max_length=64)
            # output_ids already contains generated tokens (including input if wrapper does that, let's check)
            # Wrapper generate returns ALL ids.
            generated_pragma = self.tokenizer_gen.decode(output_ids, skip_special_tokens=True)
        else:
            # PyTorch expects tensor on device
            input_ids = input_ids.to(self.device)
            outputs = self.model_gen.generate(input_ids, max_length=64)
            generated_pragma = self.tokenizer_gen.decode(outputs[0], skip_special_tokens=True)

        return generated_pragma[len(loop):]


    def auto_comp(self, loop) -> str or None:
        """
        Return an omp pragma if neccessary
        """
        if self.cls_par(loop):
            return self.pragma_format(self.gen_par(loop))
