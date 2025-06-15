#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/tensor.hpp"
#include <random>

namespace kotml {
namespace nn {

// Dropout layer (for regularization)
class Dropout : public Module {
private:
    float m_dropoutRate;
    mutable std::mt19937 m_generator;
    mutable std::uniform_real_distribution<float> m_distribution;

public:
    Dropout(float dropoutRate = 0.5f) 
        : m_dropoutRate(dropoutRate), m_generator(std::random_device{}()), m_distribution(0.0f, 1.0f) {}
    
    Tensor Forward(const Tensor& input) override {
        if (!IsTraining() || m_dropoutRate == 0.0f) {
            return input; // In inference mode or without dropout
        }
        
        Tensor output = input;
        float scale = 1.0f / (1.0f - m_dropoutRate);
        
        for (size_t i = 0; i < output.Size(); ++i) {
            if (m_distribution(m_generator) < m_dropoutRate) {
                output[i] = 0.0f; // Zero out
            } else {
                output[i] = input[i] * scale; // Scale
            }
        }
        
        return output;
    }
    
    std::vector<Tensor*> Parameters() override {
        return {}; // No parameters
    }
    
    float GetDropoutRate() const { return m_dropoutRate; }
    void SetDropoutRate(float rate) { m_dropoutRate = rate; }
    
    std::string GetName() const override { return "Dropout"; }
};

} // namespace nn
} // namespace kotml 