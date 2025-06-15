#pragma once

#include "kotml/tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace kotml {
namespace nn {

// Base class for all neural network modules
class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    // Main forward pass method
    virtual Tensor Forward(const Tensor& input) = 0;
    
    // Get all module parameters
    virtual std::vector<Tensor*> Parameters() { return {}; }
    
    // Set training/inference mode
    virtual void SetTraining(bool training) { m_training = training; }
    bool IsTraining() const { return m_training; }
    
    // Zero gradients for all parameters
    virtual void ZeroGrad() {
        auto params = Parameters();
        for (auto* param : params) {
            param->ZeroGrad();
        }
    }
    
    // Get module name
    virtual std::string GetName() const = 0;

protected:
    bool m_training = true;
};

} // namespace nn
} // namespace kotml