#pragma once

#include "kotml/optim/optimizer.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>

namespace kotml {
namespace optim {

/**
 * Stochastic Gradient Descent (SGD) optimizer
 * Supports momentum and weight decay regularization
 * 
 * Update rule:
 * v_t = momentum * v_{t-1} + (1 - dampening) * (g_t + weight_decay * p_t)
 * p_t = p_{t-1} - learning_rate * v_t
 * 
 * where:
 * - v_t is the velocity (momentum buffer)
 * - g_t is the gradient at time t
 * - p_t is the parameter at time t
 */
class SGD : public Optimizer {
private:
    float m_momentum;
    float m_dampening;
    float m_weightDecay;
    bool m_nesterov;
    float m_maxGradNorm;  // Добавлено: максимальная норма градиента (0 = без ограничений)
    
    // Momentum buffers for each parameter
    std::unordered_map<Tensor*, Tensor> m_momentumBuffers;
    
    // Initialize momentum buffer for a parameter
    void InitializeMomentumBuffer(Tensor* param) {
        if (m_momentumBuffers.find(param) == m_momentumBuffers.end()) {
            m_momentumBuffers[param] = Tensor::Zeros(param->Shape());
        }
    }

public:
    /**
     * Constructor for SGD optimizer
     * 
     * @param learningRate Learning rate (step size)
     * @param momentum Momentum factor (0-1), default 0 (no momentum)
     * @param dampening Dampening for momentum, default 0
     * @param weightDecay Weight decay (L2 penalty), default 0
     * @param nesterov Enable Nesterov momentum, default false
     * @param maxGradNorm Maximum gradient norm for clipping (0 = no clipping), default 0
     */
    explicit SGD(float learningRate, 
                 float momentum = 0.0f,
                 float dampening = 0.0f, 
                 float weightDecay = 0.0f,
                 bool nesterov = false,
                 float maxGradNorm = 0.0f)
        : Optimizer(learningRate),
          m_momentum(momentum),
          m_dampening(dampening),
          m_weightDecay(weightDecay),
          m_nesterov(nesterov),
          m_maxGradNorm(maxGradNorm) {
        
        // Validate parameters
        if (learningRate < 0.0f) {
            throw std::invalid_argument("Learning rate must be non-negative");
        }
        if (momentum < 0.0f || momentum > 1.0f) {
            throw std::invalid_argument("Momentum must be between 0 and 1");
        }
        if (dampening < 0.0f || dampening > 1.0f) {
            throw std::invalid_argument("Dampening must be between 0 and 1");
        }
        if (weightDecay < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
        if (nesterov && (momentum <= 0.0f || dampening != 0.0f)) {
            throw std::invalid_argument("Nesterov momentum requires momentum > 0 and dampening = 0");
        }
        if (maxGradNorm < 0.0f) {
            throw std::invalid_argument("Max gradient norm must be non-negative");
        }
    }
    
    /**
     * Perform single optimization step
     * Updates all registered parameters using their gradients
     */
    void Step() override {
        for (auto* param : m_parameters) {
            if (!param->RequiresGrad()) {
                continue;
            }
            
            // Get gradient
            const auto& grad = param->Grad();
            if (grad.empty()) {
                continue; // Skip parameters with no gradient
            }
            
            // Create gradient tensor for computation
            Tensor gradTensor(grad, param->Shape());
            
            // Add weight decay if specified
            if (m_weightDecay != 0.0f) {
                gradTensor = gradTensor + (*param * m_weightDecay);
            }
            
            // Optional gradient clipping (only if maxGradNorm > 0)
            if (m_maxGradNorm > 0.0f) {
                float gradNorm = 0.0f;
                for (size_t i = 0; i < gradTensor.Size(); ++i) {
                    gradNorm += gradTensor[i] * gradTensor[i];
                }
                gradNorm = std::sqrt(gradNorm);
                
                if (gradNorm > m_maxGradNorm) {
                    float clipFactor = m_maxGradNorm / gradNorm;
                    for (size_t i = 0; i < gradTensor.Size(); ++i) {
                        gradTensor[i] *= clipFactor;
                    }
                }
            }
            
            // Apply momentum if specified
            if (m_momentum != 0.0f) {
                InitializeMomentumBuffer(param);
                
                auto& momentumBuffer = m_momentumBuffers[param];
                
                // Update momentum buffer: v_t = momentum * v_{t-1} + (1 - dampening) * g_t
                momentumBuffer = (momentumBuffer * m_momentum) + (gradTensor * (1.0f - m_dampening));
                
                if (m_nesterov) {
                    // Nesterov momentum: g_t = g_t + momentum * v_t
                    gradTensor = gradTensor + (momentumBuffer * m_momentum);
                } else {
                    // Standard momentum: use velocity as gradient
                    gradTensor = momentumBuffer;
                }
            }
            
            // Update parameter: p_t = p_{t-1} - learning_rate * g_t
            *param = *param - (gradTensor * m_learningRate);
        }
    }
    
    /**
     * Clear momentum buffers and gradients
     */
    void ZeroGrad() override {
        Optimizer::ZeroGrad();
        // Note: We don't clear momentum buffers as they should persist across steps
    }
    
    /**
     * Clear momentum buffers (useful for resetting optimizer state)
     */
    void ClearMomentumBuffers() {
        m_momentumBuffers.clear();
    }
    
    // Getters
    float GetMomentum() const { return m_momentum; }
    float GetDampening() const { return m_dampening; }
    float GetWeightDecay() const { return m_weightDecay; }
    bool IsNesterov() const { return m_nesterov; }
    float GetMaxGradNorm() const { return m_maxGradNorm; }
    
    // Setters
    void SetMomentum(float momentum) {
        if (momentum < 0.0f || momentum > 1.0f) {
            throw std::invalid_argument("Momentum must be between 0 and 1");
        }
        m_momentum = momentum;
    }
    
    void SetDampening(float dampening) {
        if (dampening < 0.0f || dampening > 1.0f) {
            throw std::invalid_argument("Dampening must be between 0 and 1");
        }
        m_dampening = dampening;
    }
    
    void SetWeightDecay(float weightDecay) {
        if (weightDecay < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
        m_weightDecay = weightDecay;
    }
    
    void SetNesterov(bool nesterov) {
        if (nesterov && (m_momentum <= 0.0f || m_dampening != 0.0f)) {
            throw std::invalid_argument("Nesterov momentum requires momentum > 0 and dampening = 0");
        }
        m_nesterov = nesterov;
    }
    
    void SetMaxGradNorm(float maxGradNorm) {
        if (maxGradNorm < 0.0f) {
            throw std::invalid_argument("Max gradient norm must be non-negative");
        }
        m_maxGradNorm = maxGradNorm;
    }
    
    std::string GetName() const override { return "SGD"; }
    
    /**
     * Get detailed configuration string
     */
    std::string GetConfig() const {
        std::string config = "SGD(lr=" + std::to_string(m_learningRate);
        if (m_momentum != 0.0f) {
            config += ", momentum=" + std::to_string(m_momentum);
        }
        if (m_dampening != 0.0f) {
            config += ", dampening=" + std::to_string(m_dampening);
        }
        if (m_weightDecay != 0.0f) {
            config += ", weight_decay=" + std::to_string(m_weightDecay);
        }
        if (m_nesterov) {
            config += ", nesterov=true";
        }
        if (m_maxGradNorm > 0.0f) {
            config += ", max_grad_norm=" + std::to_string(m_maxGradNorm);
        }
        config += ")";
        return config;
    }
};

} // namespace optim
} // namespace kotml 