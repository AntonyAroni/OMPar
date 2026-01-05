#include "monocoder_trt.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

namespace ompar {

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

MonoCoderTRT::MonoCoderTRT(const std::string& engine_path) {
    // 1. Load engine file
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("Error opening engine file: " + engine_path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Error reading engine file");
    }
    
    // 2. Create runtime and engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT Runtime");
    
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT Engine");
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create Execution Context");
    
    // 3. Allocate GPU buffers
    // Fixed shape: (1, 64)
    input_size_ = FIXED_LEN * sizeof(int);
    // Output is FP16 from engine (1, 64, 50688)
    output_size_ = FIXED_LEN * VOCAB_SIZE * sizeof(unsigned short); // using ushort for half storage
    
    if (cudaMalloc(&buffers_[0], input_size_) != cudaSuccess) throw std::runtime_error("CudaMalloc failed for input");
    if (cudaMalloc(&buffers_[1], output_size_) != cudaSuccess) throw std::runtime_error("CudaMalloc failed for output");
}

MonoCoderTRT::~MonoCoderTRT() {
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
}

// Half to Float conversion helper
// Simple version for CPU side
static float half_to_float(unsigned short h) {
    int s = (h >> 15) & 0x00000001;
    int e = (h >> 10) & 0x0000001f;
    int m =  h        & 0x000003ff;
    
    if (e == 0) {
        if (m == 0) return s ? -0.0f : 0.0f;
        while (!(m & 0x400)) { m <<= 1; e -= 1; }
        e += 1; m &= ~0x400;
        unsigned int x = (s << 31) | ((e + 112) << 23) | (m << 13);
        float f; memcpy(&f, &x, 4); return f;
    } else if (e == 31) {
        return m == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    unsigned int x = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

std::vector<float> MonoCoderTRT::forward(const std::vector<int>& input_ids_vec, uint64_t stream_ptr) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // Basic validation
    if (input_ids_vec.size() > (size_t)FIXED_LEN) {
        // Truncate or throw? Let's truncate for robustness
         // throw std::runtime_error("Input too long");
    }
    
    // Prepare input with padding (50256 is pad token)
    std::vector<int> padded_input(FIXED_LEN, 50256);
    size_t copy_len = std::min(input_ids_vec.size(), (size_t)FIXED_LEN);
    std::copy(input_ids_vec.begin(), input_ids_vec.begin() + copy_len, padded_input.begin());
    
    // Copy to GPU (Async on stream)
    cudaMemcpyAsync(buffers_[0], padded_input.data(), input_size_, cudaMemcpyHostToDevice, stream);
    
    // Set Input Shape for Dynamic Profile
    nvinfer1::Dims dims;
    dims.nbDims = 2;
    dims.d[0] = 1;
    dims.d[1] = FIXED_LEN;
    
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);
    
    // Debug info
    nvinfer1::Dims out_dims = engine_->getTensorShape(output_name);
    std::cout << "DEBUG: Output Tensor " << output_name << " dims: " << out_dims.nbDims << " [" << out_dims.d[0] << ", " << out_dims.d[1] << ", " << out_dims.d[2] << "]" << std::endl;
    std::cout << "DEBUG: Allocated Output Size: " << output_size_ << " bytes" << std::endl;
    
    if (!context_->setInputShape(input_name, dims)) {
         std::cerr << "Failed to set input shape" << std::endl;
    }
    
    // Set Tensor Addresses (Required for enqueueV3)
    context_->setTensorAddress(input_name, buffers_[0]);
    context_->setTensorAddress(output_name, buffers_[1]);
    
    // Run inference (EnqueueV3)
    if(!context_->enqueueV3(stream)) {
        std::cerr << "EnqueueV3 failed" << std::endl;
    }
    
    // Copy output to CPU
    size_t last_token_idx = copy_len > 0 ? copy_len - 1 : 0;
    size_t offset_elements = last_token_idx * VOCAB_SIZE;
    size_t size_bytes = VOCAB_SIZE * sizeof(unsigned short);
    
    std::vector<unsigned short> output_half(VOCAB_SIZE);
    
    void* src_ptr = (char*)buffers_[1] + offset_elements * sizeof(unsigned short);
    
    // Copy Async and Sync Stream
    cudaMemcpyAsync(output_half.data(), src_ptr, size_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Convert to float
    std::vector<float> logits(VOCAB_SIZE);
    for(int i=0; i<VOCAB_SIZE; ++i) {
        logits[i] = half_to_float(output_half[i]);
    }
    
    return logits;
}

std::vector<int> MonoCoderTRT::generate(const std::vector<int>& start_ids, int max_length, uint64_t stream_ptr) {
    std::vector<int> current_ids = start_ids;
    
    while (current_ids.size() < (size_t)max_length && current_ids.size() < (size_t)FIXED_LEN) {
        // Forward
        std::vector<float> logits = forward(current_ids, stream_ptr);
        
        // Greedy Argmax
        int next_token = 0;
        float max_val = -1e9;
        
        for(int i=0; i<VOCAB_SIZE; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                next_token = i;
            }
        }
        
        if (next_token == 50256) break; // EOS
        current_ids.push_back(next_token);
    }
    
    return current_ids;
}

} // namespace ompar
