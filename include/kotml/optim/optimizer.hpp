#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <memory>

namespace kotml {
namespace optim {

/**
 * Base class for all optimizers
 * Provides common interface for parameter optimization
 */
class Optimizer {
protected:
    std::vector<Tensor*> m_parameters;
    float m_learningRate;

public:
    // Constructor
    explicit Optimizer(float learningRate) 
        : m_learningRate(learningRate) {}
    
    // Virtual destructor
    virtual ~Optimizer() = default;
    
    // Add parameters from a module
    void AddParameters(nn::Module& module) {
        auto params = module.Parameters();
        m_parameters.insert(m_parameters.end(), params.begin(), params.end());
    }
    
    // Add a single parameter
    void AddParameter(Tensor& parameter) {
        m_parameters.push_back(&parameter);
    }
    
    // Clear all parameters
    void ClearParameters() {
        m_parameters.clear();
    }
    
    // Get current learning rate
    float GetLearningRate() const { return m_learningRate; }
    
    // Set learning rate
    void SetLearningRate(float learningRate) { m_learningRate = learningRate; }
    
    // Get number of parameters
    size_t GetParameterCount() const { return m_parameters.size(); }
    
    // Zero gradients for all parameters
    virtual void ZeroGrad() {
        for (auto* param : m_parameters) {
            param->ZeroGrad();
        }
    }
    
    // Update parameters (pure virtual)
    virtual void Step() = 0;
    
    // Get optimizer name
    virtual std::string GetName() const = 0;
};

} // namespace optim
} // namespace kotml 