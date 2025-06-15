#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/nn/linear_layer.hpp"
#include "kotml/nn/activation_layer.hpp"
#include "kotml/tensor.hpp"
#include <memory>

namespace kotml {
namespace nn {

// Output layer (combines Linear + Activation for clarity)
class OutputLayer : public Module {
private:
    std::unique_ptr<Linear> m_linear;
    std::unique_ptr<Activation> m_activation;
    bool m_hasActivation;

public:
    OutputLayer(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::None)
        : m_hasActivation(activation != ActivationType::None) {
        
        m_linear = std::make_unique<Linear>(inputSize, outputSize);
        
        if (m_hasActivation) {
            m_activation = std::make_unique<Activation>(activation);
        }
    }
    
    Tensor Forward(const Tensor& input) override {
        Tensor output = m_linear->Forward(input);
        
        if (m_hasActivation) {
            output = m_activation->Forward(output);
        }
        
        return output;
    }
    
    std::vector<Tensor*> Parameters() override {
        return m_linear->Parameters();
    }
    
    void SetTraining(bool training) override {
        Module::SetTraining(training);
        m_linear->SetTraining(training);
        if (m_hasActivation) {
            m_activation->SetTraining(training);
        }
    }
    
    void ZeroGrad() override {
        m_linear->ZeroGrad();
    }
    
    // Getters
    Linear& GetLinear() { return *m_linear; }
    const Linear& GetLinear() const { return *m_linear; }
    
    Activation& GetActivation() { 
        if (!m_hasActivation) {
            throw std::runtime_error("OutputLayer has no activation");
        }
        return *m_activation; 
    }
    
    const Activation& GetActivation() const { 
        if (!m_hasActivation) {
            throw std::runtime_error("OutputLayer has no activation");
        }
        return *m_activation; 
    }
    
    size_t GetInputSize() const { return m_linear->GetInputSize(); }
    size_t GetOutputSize() const { return m_linear->GetOutputSize(); }
    bool HasActivation() const { return m_hasActivation; }
    
    size_t CountParameters() const {
        return m_linear->CountParameters();
    }
    
    std::string GetName() const override { 
        if (m_hasActivation) {
            return "OutputLayer(" + m_activation->GetName() + ")";
        }
        return "OutputLayer";
    }
};

} // namespace nn
} // namespace kotml 