#pragma once

#include "kotml/tensor.hpp"
#include <cmath>

namespace kotml {
namespace nn {

// Activation functions
namespace activations {

// ReLU activation
inline Tensor Relu(const Tensor& input) {
    Tensor result(input.Shape(), input.RequiresGrad());
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = std::max(0.0f, input[i]);
    }
    return result;
}

// Sigmoid activation
inline Tensor Sigmoid(const Tensor& input) {
    Tensor result(input.Shape(), input.RequiresGrad());
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    return result;
}

// Tanh activation
inline Tensor Tanh(const Tensor& input) {
    Tensor result(input.Shape(), input.RequiresGrad());
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = std::tanh(input[i]);
    }
    return result;
}

// ReLU derivative
inline Tensor ReluDerivative(const Tensor& input) {
    Tensor result(input.Shape(), false);
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = input[i] > 0.0f ? 1.0f : 0.0f;
    }
    return result;
}

// Sigmoid derivative
inline Tensor SigmoidDerivative(const Tensor& input) {
    Tensor sigmoid_val = Sigmoid(input);
    Tensor result(input.Shape(), false);
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = sigmoid_val[i] * (1.0f - sigmoid_val[i]);
    }
    return result;
}

// Tanh derivative
inline Tensor TanhDerivative(const Tensor& input) {
    Tensor tanh_val = Tanh(input);
    Tensor result(input.Shape(), false);
    for (size_t i = 0; i < input.Size(); ++i) {
        result[i] = 1.0f - tanh_val[i] * tanh_val[i];
    }
    return result;
}

} // namespace activations

// Activation type enumeration
enum class ActivationType {
    None,
    Relu,
    Sigmoid,
    Tanh
};

// Apply activation by type
inline Tensor ApplyActivation(const Tensor& input, ActivationType type) {
    switch (type) {
        case ActivationType::Relu:
            return activations::Relu(input);
        case ActivationType::Sigmoid:
            return activations::Sigmoid(input);
        case ActivationType::Tanh:
            return activations::Tanh(input);
        case ActivationType::None:
        default:
            return input;
    }
}

} // namespace nn
} // namespace kotml 