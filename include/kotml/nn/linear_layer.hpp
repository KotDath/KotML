#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/tensor.hpp"
#include <stdexcept>
#include <cmath>

namespace kotml {
namespace nn {

// Linear (fully connected) layer
class Linear : public Module {
private:
    Tensor m_weight;
    Tensor m_bias;
    size_t m_inputSize;
    size_t m_outputSize;
    bool m_useBias;

public:
    // Constructor
    Linear(size_t inputSize, size_t outputSize, bool useBias = true)
        : m_inputSize(inputSize), m_outputSize(outputSize), m_useBias(useBias) {
        
        // Weight initialization (Xavier/Glorot initialization)
        m_weight = Tensor::Randn({outputSize, inputSize}, true);
        float scale = std::sqrt(2.0f / (inputSize + outputSize));
        m_weight = m_weight * scale;
        
        // Bias initialization
        if (m_useBias) {
            m_bias = Tensor::Zeros({outputSize}, true);
        }
    }
    
    // Forward pass
    Tensor Forward(const Tensor& input) override {
        // input: [batch_size, input_size] or [input_size]
        // weight: [output_size, input_size]
        // output: [batch_size, output_size] or [output_size]
        
        Tensor output;
        
        if (input.Ndim() == 1) {
            // Vector case: input [input_size] -> output [output_size]
            if (input.Size() != m_inputSize) {
                throw std::invalid_argument("Input size mismatch. Expected " + 
                    std::to_string(m_inputSize) + ", got " + std::to_string(input.Size()));
            }
            
            output = m_weight.Matmul(input);
        } else if (input.Ndim() == 2) {
            // Batch case: input [batch_size, input_size] -> output [batch_size, output_size]
            if (input.Shape()[1] != m_inputSize) {
                throw std::invalid_argument("Input feature size mismatch. Expected " + 
                    std::to_string(m_inputSize) + ", got " + std::to_string(input.Shape()[1]));
            }
            
            // For batch processing: (batch_size, input_size) @ (input_size, output_size)^T
            // We need to transpose weight matrix for correct multiplication
            Tensor weight_t = m_weight.Transpose();
            output = input.Matmul(weight_t);
        } else {
            throw std::invalid_argument("Input must be 1D or 2D tensor");
        }
        
        // Add bias
        if (m_useBias) {
            if (output.Ndim() == 1) {
                // Vector case: direct addition
                output = output + m_bias;
            } else {
                // Batch case: add bias to each sample in batch
                for (size_t i = 0; i < output.Shape()[0]; ++i) {
                    for (size_t j = 0; j < output.Shape()[1]; ++j) {
                        output.At({i, j}) += m_bias[j];
                    }
                }
            }
        }
        
        return output;
    }
    
    // Get parameters
    std::vector<Tensor*> Parameters() override {
        std::vector<Tensor*> params;
        params.push_back(&m_weight);
        if (m_useBias) {
            params.push_back(&m_bias);
        }
        return params;
    }
    
    // Getters
    const Tensor& GetWeight() const { return m_weight; }
    const Tensor& GetBias() const { return m_bias; }
    size_t GetInputSize() const { return m_inputSize; }
    size_t GetOutputSize() const { return m_outputSize; }
    bool UsesBias() const { return m_useBias; }
    
    size_t CountParameters() const {
        size_t count = m_weight.Size();
        if (m_useBias) {
            count += m_bias.Size();
        }
        return count;
    }
    
    std::string GetName() const override { return "Linear"; }
};

} // namespace nn
} // namespace kotml 