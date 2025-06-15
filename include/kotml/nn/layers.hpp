#pragma once
// Main header file for all neural network layers
// Includes all individual layer files for convenience

// Core modules
#include "kotml/nn/module.hpp"
#include "kotml/nn/activations.hpp"

// Individual layers
#include "kotml/nn/input_layer.hpp"
#include "kotml/nn/linear_layer.hpp"
#include "kotml/nn/activation_layer.hpp"
#include "kotml/nn/dropout_layer.hpp"
#include "kotml/nn/output_layer.hpp"

// Composite architectures
#include "kotml/nn/ffn.hpp"
#include "kotml/nn/sequential.hpp"

// All layer classes are available through namespace kotml::nn:
// - InputLayer: Input layer for dimension validation
// - Linear: Fully connected (linear) layer
// - Activation: Activation layer (ReLU, Sigmoid, Tanh)
// - Dropout: Regularization layer
// - OutputLayer: Output layer (Linear + Activation)
// - FFN: Feed-Forward Network (multi-layer perceptron)
// - Sequential: Sequential network builder with Builder pattern