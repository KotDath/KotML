#pragma once

#include "kotml/tensor.hpp"
#include <string>
#include <stdexcept>
#include <cmath>

namespace kotml {
namespace nn {

/**
 * Base class for all loss functions
 * Provides common interface for loss computation and gradient calculation
 */
class Loss {
public:
    Loss() = default;
    virtual ~Loss() = default;
    
    /**
     * Compute loss between predictions and targets
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @return Loss tensor (scalar)
     */
    virtual Tensor Forward(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * Compute gradients with respect to predictions
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @return Gradients tensor (same shape as predictions)
     */
    virtual Tensor Backward(const Tensor& predictions, const Tensor& targets) = 0;
    
    /**
     * Get loss function name
     */
    virtual std::string GetName() const = 0;
    
    /**
     * Compute both forward and backward pass
     * @param predictions Model predictions
     * @param targets Ground truth targets
     * @return Pair of (loss, gradients)
     */
    virtual std::pair<Tensor, Tensor> ForwardBackward(const Tensor& predictions, const Tensor& targets) {
        Tensor loss = Forward(predictions, targets);
        Tensor gradients = Backward(predictions, targets);
        return {loss, gradients};
    }
};

/**
 * Mean Squared Error (MSE) Loss
 * L = (1/n) * Σ(y_pred - y_true)²
 * 
 * Used for regression tasks
 */
class MSELoss : public Loss {
public:
    MSELoss() = default;
    
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        // Compute squared differences
        Tensor diff = predictions - targets;
        Tensor squared_diff = diff * diff;
        
        // Return mean
        return squared_diff.Mean();
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        // Gradient: 2/n * (y_pred - y_true)
        Tensor diff = predictions - targets;
        float scale = 2.0f / static_cast<float>(predictions.Size());
        
        return diff * scale;
    }
    
    std::string GetName() const override { return "MSELoss"; }

private:
    void ValidateInputs(const Tensor& predictions, const Tensor& targets) {
        if (predictions.Shape() != targets.Shape()) {
            throw std::invalid_argument("Predictions and targets must have the same shape");
        }
        if (predictions.Empty() || targets.Empty()) {
            throw std::invalid_argument("Predictions and targets cannot be empty");
        }
    }
};

/**
 * Mean Absolute Error (MAE) Loss / L1 Loss
 * L = (1/n) * Σ|y_pred - y_true|
 * 
 * More robust to outliers than MSE
 */
class MAELoss : public Loss {
public:
    MAELoss() = default;
    
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        // Compute absolute differences
        Tensor diff = predictions - targets;
        
        // Manual absolute value computation
        Tensor abs_diff = Tensor::Zeros(diff.Shape());
        for (size_t i = 0; i < diff.Size(); ++i) {
            abs_diff[i] = std::abs(diff[i]);
        }
        
        return abs_diff.Mean();
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        // Gradient: sign(y_pred - y_true) / n
        Tensor diff = predictions - targets;
        Tensor gradients = Tensor::Zeros(diff.Shape());
        float scale = 1.0f / static_cast<float>(predictions.Size());
        
        for (size_t i = 0; i < diff.Size(); ++i) {
            if (diff[i] > 0) {
                gradients[i] = scale;
            } else if (diff[i] < 0) {
                gradients[i] = -scale;
            } else {
                gradients[i] = 0.0f; // Subgradient at 0
            }
        }
        
        return gradients;
    }
    
    std::string GetName() const override { return "MAELoss"; }

private:
    void ValidateInputs(const Tensor& predictions, const Tensor& targets) {
        if (predictions.Shape() != targets.Shape()) {
            throw std::invalid_argument("Predictions and targets must have the same shape");
        }
        if (predictions.Empty() || targets.Empty()) {
            throw std::invalid_argument("Predictions and targets cannot be empty");
        }
    }
};

/**
 * Binary Cross Entropy Loss
 * L = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
 * 
 * Used for binary classification tasks
 * Predictions should be probabilities (0-1)
 */
class BCELoss : public Loss {
private:
    float m_epsilon; // Small value to prevent log(0)
    
public:
    explicit BCELoss(float epsilon = 1e-7f) : m_epsilon(epsilon) {}
    
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor loss = Tensor::Zeros({1});
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < predictions.Size(); ++i) {
            float pred = std::max(m_epsilon, std::min(1.0f - m_epsilon, predictions[i]));
            float target = targets[i];
            
            total_loss += -(target * std::log(pred) + (1.0f - target) * std::log(1.0f - pred));
        }
        
        loss[0] = total_loss / static_cast<float>(predictions.Size());
        return loss;
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor gradients = Tensor::Zeros(predictions.Shape());
        float scale = 1.0f / static_cast<float>(predictions.Size());
        
        for (size_t i = 0; i < predictions.Size(); ++i) {
            float pred = std::max(m_epsilon, std::min(1.0f - m_epsilon, predictions[i]));
            float target = targets[i];
            
            // Gradient: -(target/pred - (1-target)/(1-pred)) / n
            gradients[i] = scale * (pred - target) / (pred * (1.0f - pred));
        }
        
        return gradients;
    }
    
    std::string GetName() const override { return "BCELoss"; }
    
    float GetEpsilon() const { return m_epsilon; }
    void SetEpsilon(float epsilon) { m_epsilon = epsilon; }

private:
    void ValidateInputs(const Tensor& predictions, const Tensor& targets) {
        if (predictions.Shape() != targets.Shape()) {
            throw std::invalid_argument("Predictions and targets must have the same shape");
        }
        if (predictions.Empty() || targets.Empty()) {
            throw std::invalid_argument("Predictions and targets cannot be empty");
        }
        
        // Check if targets are binary (0 or 1)
        for (size_t i = 0; i < targets.Size(); ++i) {
            float target = targets[i];
            if (target != 0.0f && target != 1.0f) {
                throw std::invalid_argument("BCE targets must be binary (0 or 1)");
            }
        }
    }
};

/**
 * Categorical Cross Entropy Loss
 * L = -(1/n) * Σ Σ y_true[i,j] * log(y_pred[i,j])
 * 
 * Used for multi-class classification
 * Predictions should be probabilities (softmax output)
 * Targets should be one-hot encoded
 */
class CrossEntropyLoss : public Loss {
private:
    float m_epsilon; // Small value to prevent log(0)
    
public:
    explicit CrossEntropyLoss(float epsilon = 1e-7f) : m_epsilon(epsilon) {}
    
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor loss = Tensor::Zeros({1});
        float total_loss = 0.0f;
        
        if (predictions.Ndim() == 1) {
            // Single sample case
            for (size_t i = 0; i < predictions.Size(); ++i) {
                if (targets[i] > 0.0f) {
                    float pred = std::max(m_epsilon, predictions[i]);
                    total_loss += -targets[i] * std::log(pred);
                }
            }
        } else if (predictions.Ndim() == 2) {
            // Batch case
            size_t batch_size = predictions.Shape()[0];
            size_t num_classes = predictions.Shape()[1];
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t c = 0; c < num_classes; ++c) {
                    size_t idx = b * num_classes + c;
                    if (targets[idx] > 0.0f) {
                        float pred = std::max(m_epsilon, predictions[idx]);
                        total_loss += -targets[idx] * std::log(pred);
                    }
                }
            }
            total_loss /= static_cast<float>(batch_size);
        }
        
        loss[0] = total_loss;
        return loss;
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor gradients = Tensor::Zeros(predictions.Shape());
        
        if (predictions.Ndim() == 1) {
            // Single sample case
            for (size_t i = 0; i < predictions.Size(); ++i) {
                float pred = std::max(m_epsilon, predictions[i]);
                gradients[i] = -targets[i] / pred;
            }
        } else if (predictions.Ndim() == 2) {
            // Batch case
            size_t batch_size = predictions.Shape()[0];
            size_t num_classes = predictions.Shape()[1];
            float scale = 1.0f / static_cast<float>(batch_size);
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t c = 0; c < num_classes; ++c) {
                    size_t idx = b * num_classes + c;
                    float pred = std::max(m_epsilon, predictions[idx]);
                    gradients[idx] = -scale * targets[idx] / pred;
                }
            }
        }
        
        return gradients;
    }
    
    std::string GetName() const override { return "CrossEntropyLoss"; }
    
    float GetEpsilon() const { return m_epsilon; }
    void SetEpsilon(float epsilon) { m_epsilon = epsilon; }

private:
    void ValidateInputs(const Tensor& predictions, const Tensor& targets) {
        if (predictions.Shape() != targets.Shape()) {
            throw std::invalid_argument("Predictions and targets must have the same shape");
        }
        if (predictions.Empty() || targets.Empty()) {
            throw std::invalid_argument("Predictions and targets cannot be empty");
        }
        if (predictions.Ndim() > 2) {
            throw std::invalid_argument("CrossEntropyLoss supports only 1D or 2D tensors");
        }
    }
};

/**
 * Huber Loss (Smooth L1 Loss)
 * Combines MSE and MAE - quadratic for small errors, linear for large errors
 * 
 * L = { 0.5 * (y_pred - y_true)²           if |y_pred - y_true| <= delta
 *     { delta * |y_pred - y_true| - 0.5 * delta²  otherwise
 * 
 * More robust to outliers than MSE, smoother than MAE
 */
class HuberLoss : public Loss {
private:
    float m_delta;
    
public:
    explicit HuberLoss(float delta = 1.0f) : m_delta(delta) {
        if (delta <= 0.0f) {
            throw std::invalid_argument("Huber loss delta must be positive");
        }
    }
    
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor loss = Tensor::Zeros({1});
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < predictions.Size(); ++i) {
            float diff = predictions[i] - targets[i];
            float abs_diff = std::abs(diff);
            
            if (abs_diff <= m_delta) {
                total_loss += 0.5f * diff * diff;
            } else {
                total_loss += m_delta * abs_diff - 0.5f * m_delta * m_delta;
            }
        }
        
        loss[0] = total_loss / static_cast<float>(predictions.Size());
        return loss;
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        ValidateInputs(predictions, targets);
        
        Tensor gradients = Tensor::Zeros(predictions.Shape());
        float scale = 1.0f / static_cast<float>(predictions.Size());
        
        for (size_t i = 0; i < predictions.Size(); ++i) {
            float diff = predictions[i] - targets[i];
            float abs_diff = std::abs(diff);
            
            if (abs_diff <= m_delta) {
                gradients[i] = scale * diff;
            } else {
                gradients[i] = scale * m_delta * (diff > 0 ? 1.0f : -1.0f);
            }
        }
        
        return gradients;
    }
    
    std::string GetName() const override { return "HuberLoss"; }
    
    float GetDelta() const { return m_delta; }
    void SetDelta(float delta) {
        if (delta <= 0.0f) {
            throw std::invalid_argument("Huber loss delta must be positive");
        }
        m_delta = delta;
    }

private:
    void ValidateInputs(const Tensor& predictions, const Tensor& targets) {
        if (predictions.Shape() != targets.Shape()) {
            throw std::invalid_argument("Predictions and targets must have the same shape");
        }
        if (predictions.Empty() || targets.Empty()) {
            throw std::invalid_argument("Predictions and targets cannot be empty");
        }
    }
};

// Convenience functions for common loss computations
namespace loss {

/**
 * Compute MSE loss
 */
inline Tensor MSE(const Tensor& predictions, const Tensor& targets) {
    MSELoss loss;
    return loss.Forward(predictions, targets);
}

/**
 * Compute MAE loss
 */
inline Tensor MAE(const Tensor& predictions, const Tensor& targets) {
    MAELoss loss;
    return loss.Forward(predictions, targets);
}

/**
 * Compute BCE loss
 */
inline Tensor BCE(const Tensor& predictions, const Tensor& targets, float epsilon = 1e-7f) {
    BCELoss loss(epsilon);
    return loss.Forward(predictions, targets);
}

/**
 * Compute Cross Entropy loss
 */
inline Tensor CrossEntropy(const Tensor& predictions, const Tensor& targets, float epsilon = 1e-7f) {
    CrossEntropyLoss loss(epsilon);
    return loss.Forward(predictions, targets);
}

/**
 * Compute Huber loss
 */
inline Tensor Huber(const Tensor& predictions, const Tensor& targets, float delta = 1.0f) {
    HuberLoss loss(delta);
    return loss.Forward(predictions, targets);
}

} // namespace loss

} // namespace nn
} // namespace kotml 