#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "NvInfer.h"

namespace ompar {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class MonoCoderTRT {
public:
    MonoCoderTRT(const std::string& engine_path);
    ~MonoCoderTRT();

    // Inferencia simple (return logits)
    std::vector<float> forward(const std::vector<int>& input_ids, uint64_t stream_ptr = 0);
    
    // Generación greedy (return tokens)
    std::vector<int> generate(const std::vector<int>& start_ids, int max_length, uint64_t stream_ptr = 0);

private:
    Logger logger_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    
    // Buffers
    void* buffers_[2]; // 0: input, 1: output
    size_t input_size_;
    size_t output_size_;
    
    // Constants
    const int FIXED_LEN = 64;
    const int VOCAB_SIZE = 50688;
};

} // namespace ompar
