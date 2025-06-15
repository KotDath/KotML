#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/tensor.hpp"

namespace kotml {
namespace nn {

// Input layer (for architecture clarity, has no parameters)
class InputLayer : public Module {
private:
    size_t m_inputSize;

public:
    InputLayer(size_t inputSize) : m_inputSize(inputSize) {}
    
    Tensor Forward(const Tensor& input) override {
        // Dimension check
        if (input.Ndim() == 1 && input.Size() != m_inputSize) {
            throw std::invalid_argument("Input size mismatch");
        } else if (input.Ndim() == 2 && input.Shape()[1] != m_inputSize) {
            throw std::invalid_argument("Input feature size mismatch");
        }
        
        // Simply pass data through unchanged
        return input;
    }
    
    std::vector<Tensor*> Parameters() override {
        return {}; // No parameters
    }
    
    size_t GetInputSize() const { return m_inputSize; }
    
    std::string GetName() const override { return "InputLayer"; }
};

} // namespace nn
} // namespace kotml 