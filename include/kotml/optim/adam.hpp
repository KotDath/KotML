#pragma once

#include "kotml/optim/optimizer.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>

namespace kotml {
namespace optim {

/**
 * Adam (Adaptive Moment Estimation) optimizer
 * Combines the advantages of AdaGrad and RMSProp
 * 
 * Update rule:
 * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 * m_hat_t = m_t / (1 - beta1^t)
 * v_hat_t = v_t / (1 - beta2^t)
 * p_t = p_{t-1} - learning_rate * m_hat_t / (sqrt(v_hat_t) + epsilon)
 * 
 * where:
 * - m_t is the first moment estimate (mean of gradients)
 * - v_t is the second moment estimate (uncentered variance of gradients)
 * - g_t is the gradient at time t
 * - p_t is the parameter at time t
 */
class Adam : public Optimizer {
private:
    float m_beta1;        // Exponential decay rate for first moment estimates
    float m_beta2;        // Exponential decay rate for second moment estimates
    float m_epsilon;      // Small constant for numerical stability
    float m_weightDecay;  // Weight decay (L2 penalty)
    bool m_amsgrad;       // Whether to use AMSGrad variant
    float m_maxGradNorm;  // Maximum gradient norm for clipping (0 = no clipping)
    
    size_t m_step;        // Current step number (for bias correction)
    
    // First moment buffers (momentum)
    std::unordered_map<Tensor*, Tensor> m_firstMomentBuffers;
    
    // Second moment buffers (RMSProp)
    std::unordered_map<Tensor*, Tensor> m_secondMomentBuffers;
    
    // Maximum second moment buffers (for AMSGrad)
    std::unordered_map<Tensor*, Tensor> m_maxSecondMomentBuffers;
    
    // Initialize buffers for a parameter
    void InitializeBuffers(Tensor* param) {
        if (m_firstMomentBuffers.find(param) == m_firstMomentBuffers.end()) {
            m_firstMomentBuffers[param] = Tensor::Zeros(param->Shape());
            m_secondMomentBuffers[param] = Tensor::Zeros(param->Shape());
            if (m_amsgrad) {
                m_maxSecondMomentBuffers[param] = Tensor::Zeros(param->Shape());
            }
        }
    }

public:
    /**
     * Constructor for Adam optimizer
     * 
     * @param learningRate Learning rate, default 0.001
     * @param beta1 Coefficient for computing running averages of gradient, default 0.9
     * @param beta2 Coefficient for computing running averages of squared gradient, default 0.999
     * @param epsilon Term added to denominator for numerical stability, default 1e-8
     * @param weightDecay Weight decay (L2 penalty), default 0
     * @param amsgrad Whether to use AMSGrad variant, default false
     * @param maxGradNorm Maximum gradient norm for clipping (0 = no clipping), default 0
     */
    explicit Adam(float learningRate = 0.001f,
                  float beta1 = 0.9f,
                  float beta2 = 0.999f,
                  float epsilon = 1e-8f,
                  float weightDecay = 0.0f,
                  bool amsgrad = false,
                  float maxGradNorm = 0.0f)
        : Optimizer(learningRate),
          m_beta1(beta1),
          m_beta2(beta2),
          m_epsilon(epsilon),
          m_weightDecay(weightDecay),
          m_amsgrad(amsgrad),
          m_maxGradNorm(maxGradNorm),
          m_step(0) {
        
        // Validate parameters
        if (learningRate < 0.0f) {
            throw std::invalid_argument("Learning rate must be non-negative");
        }
        if (beta1 < 0.0f || beta1 >= 1.0f) {
            throw std::invalid_argument("Beta1 must be in [0, 1)");
        }
        if (beta2 < 0.0f || beta2 >= 1.0f) {
            throw std::invalid_argument("Beta2 must be in [0, 1)");
        }
        if (epsilon <= 0.0f) {
            throw std::invalid_argument("Epsilon must be positive");
        }
        if (weightDecay < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
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
        m_step++;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(m_beta1, static_cast<float>(m_step));
        float bias_correction2 = 1.0f - std::pow(m_beta2, static_cast<float>(m_step));
        
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
            
            // Initialize buffers if needed
            InitializeBuffers(param);
            
            auto& firstMoment = m_firstMomentBuffers[param];
            auto& secondMoment = m_secondMomentBuffers[param];
            
            // Update first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            firstMoment = (firstMoment * m_beta1) + (gradTensor * (1.0f - m_beta1));
            
            // Update second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            Tensor gradSquared = gradTensor * gradTensor;
            secondMoment = (secondMoment * m_beta2) + (gradSquared * (1.0f - m_beta2));
            
            // Bias-corrected first moment estimate
            Tensor firstMomentHat = firstMoment / bias_correction1;
            
            // Bias-corrected second moment estimate
            Tensor secondMomentHat = secondMoment / bias_correction2;
            
            // Use AMSGrad variant if enabled
            if (m_amsgrad) {
                auto& maxSecondMoment = m_maxSecondMomentBuffers[param];
                
                // Update maximum second moment: v_max_t = max(v_max_{t-1}, v_t)
                for (size_t i = 0; i < secondMomentHat.Size(); ++i) {
                    maxSecondMoment[i] = std::max(maxSecondMoment[i], secondMomentHat[i]);
                }
                
                // Use maximum for denominator
                secondMomentHat = maxSecondMoment;
            }
            
            // Compute update: delta = learning_rate * m_hat_t / (sqrt(v_hat_t) + epsilon)
            Tensor update(param->Shape());
            for (size_t i = 0; i < param->Size(); ++i) {
                float denominator = std::sqrt(secondMomentHat[i]) + m_epsilon;
                update[i] = m_learningRate * firstMomentHat[i] / denominator;
            }
            
            // Update parameter: p_t = p_{t-1} - delta
            *param = *param - update;
        }
    }
    
    /**
     * Clear all buffers and reset step counter
     */
    void Reset() {
        m_firstMomentBuffers.clear();
        m_secondMomentBuffers.clear();
        m_maxSecondMomentBuffers.clear();
        m_step = 0;
    }
    
    // Getters
    float GetBeta1() const { return m_beta1; }
    float GetBeta2() const { return m_beta2; }
    float GetEpsilon() const { return m_epsilon; }
    float GetWeightDecay() const { return m_weightDecay; }
    bool IsAmsgrad() const { return m_amsgrad; }
    float GetMaxGradNorm() const { return m_maxGradNorm; }
    size_t GetStep() const { return m_step; }
    
    // Setters
    void SetBeta1(float beta1) {
        if (beta1 < 0.0f || beta1 >= 1.0f) {
            throw std::invalid_argument("Beta1 must be in [0, 1)");
        }
        m_beta1 = beta1;
    }
    
    void SetBeta2(float beta2) {
        if (beta2 < 0.0f || beta2 >= 1.0f) {
            throw std::invalid_argument("Beta2 must be in [0, 1)");
        }
        m_beta2 = beta2;
    }
    
    void SetEpsilon(float epsilon) {
        if (epsilon <= 0.0f) {
            throw std::invalid_argument("Epsilon must be positive");
        }
        m_epsilon = epsilon;
    }
    
    void SetWeightDecay(float weightDecay) {
        if (weightDecay < 0.0f) {
            throw std::invalid_argument("Weight decay must be non-negative");
        }
        m_weightDecay = weightDecay;
    }
    
    void SetAmsgrad(bool amsgrad) {
        m_amsgrad = amsgrad;
        if (amsgrad) {
            // Initialize max buffers for existing parameters
            for (const auto& pair : m_secondMomentBuffers) {
                if (m_maxSecondMomentBuffers.find(pair.first) == m_maxSecondMomentBuffers.end()) {
                    m_maxSecondMomentBuffers[pair.first] = Tensor::Zeros(pair.first->Shape());
                }
            }
        } else {
            // Clear max buffers if AMSGrad is disabled
            m_maxSecondMomentBuffers.clear();
        }
    }
    
    void SetMaxGradNorm(float maxGradNorm) {
        if (maxGradNorm < 0.0f) {
            throw std::invalid_argument("Max gradient norm must be non-negative");
        }
        m_maxGradNorm = maxGradNorm;
    }
    
    std::string GetName() const override { return "Adam"; }
    
    /**
     * Get detailed configuration string
     */
    std::string GetConfig() const {
        std::string config = "Adam(lr=" + std::to_string(m_learningRate);
        config += ", beta1=" + std::to_string(m_beta1);
        config += ", beta2=" + std::to_string(m_beta2);
        config += ", epsilon=" + std::to_string(m_epsilon);
        if (m_weightDecay != 0.0f) {
            config += ", weight_decay=" + std::to_string(m_weightDecay);
        }
        if (m_amsgrad) {
            config += ", amsgrad=true";
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