#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/nn/activations.hpp"
#include "kotml/tensor.hpp"

namespace kotml {
namespace nn {

// Activation layer
class Activation : public Module {
private:
    ActivationType m_activationType;

public:
    Activation(ActivationType type) : m_activationType(type) {}
    
    Tensor Forward(const Tensor& input) override {
        return ApplyActivation(input, m_activationType);
    }
    
    std::vector<Tensor*> Parameters() override {
        return {}; // No parameters
    }
    
    ActivationType GetActivationType() const { return m_activationType; }
    
    std::string GetName() const override { 
        switch (m_activationType) {
            case ActivationType::Relu: return "ReLU";
            case ActivationType::Sigmoid: return "Sigmoid";
            case ActivationType::Tanh: return "Tanh";
            default: return "Activation";
        }
    }
};

} // namespace nn
} // namespace kotml 