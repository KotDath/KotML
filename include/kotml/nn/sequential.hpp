#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/nn/input_layer.hpp"
#include "kotml/nn/linear_layer.hpp"
#include "kotml/nn/activation_layer.hpp"
#include "kotml/nn/dropout_layer.hpp"
#include "kotml/nn/output_layer.hpp"
#include "kotml/nn/loss.hpp"
#include "kotml/optim/optimizer.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <typeinfo>
#include "kotml/utils/progress_bar.hpp"

namespace kotml {

namespace nn {

// Sequential - class for sequential neural network construction
// Uses Builder pattern for convenient architecture creation
class Sequential : public Module {
private:
    std::vector<std::unique_ptr<Module>> m_layers;
    bool m_built = false;
    
    // Training components
    std::unique_ptr<optim::Optimizer> m_optimizer;
    std::unique_ptr<Loss> m_lossFunction;
    bool m_compiled = false;

public:
    // Constructor
    Sequential() = default;
    
    // Copy constructor (deleted for safety)
    Sequential(const Sequential&) = delete;
    Sequential& operator=(const Sequential&) = delete;
    
    // Move constructor
    Sequential(Sequential&& other) noexcept 
        : m_layers(std::move(other.m_layers)), 
          m_built(other.m_built),
          m_optimizer(std::move(other.m_optimizer)),
          m_lossFunction(std::move(other.m_lossFunction)),
          m_compiled(other.m_compiled) {
        other.m_built = false;
        other.m_compiled = false;
    }
    
    Sequential& operator=(Sequential&& other) noexcept {
        if (this != &other) {
            m_layers = std::move(other.m_layers);
            m_built = other.m_built;
            m_optimizer = std::move(other.m_optimizer);
            m_lossFunction = std::move(other.m_lossFunction);
            m_compiled = other.m_compiled;
            other.m_built = false;
            other.m_compiled = false;
        }
        return *this;
    }

    /**
     * Compile the model with optimizer and loss function
     * @param optimizer Unique pointer to optimizer (SGD, Adam, etc.)
     * @param lossFunction Unique pointer to loss function (MSE, BCE, etc.)
     */
    void Compile(std::unique_ptr<optim::Optimizer> optimizer, std::unique_ptr<Loss> lossFunction) {
        if (!m_built) {
            throw std::runtime_error("Sequential must be built before compilation. Call Build() first.");
        }
        
        if (!optimizer) {
            throw std::invalid_argument("Optimizer cannot be null");
        }
        if (!lossFunction) {
            throw std::invalid_argument("Loss function cannot be null");
        }
        
        m_optimizer = std::move(optimizer);
        m_lossFunction = std::move(lossFunction);
        
        // Add model parameters to optimizer
        m_optimizer->ClearParameters();
        auto params = Parameters();
        for (auto* param : params) {
            m_optimizer->AddParameter(*param);
        }
        
        m_compiled = true;
        
        std::cout << "Sequential model compiled successfully:" << std::endl;
        std::cout << "  Optimizer: " << m_optimizer->GetName() << std::endl;
        std::cout << "  Loss function: " << m_lossFunction->GetName() << std::endl;
        std::cout << "  Parameters: " << CountParameters() << std::endl;
        std::cout << "  Layers: " << GetNumLayers() << std::endl;
    }
    
    /**
     * Train the model on provided data
     * @param trainInputs Training input data
     * @param trainTargets Training target data
     * @param epochs Number of training epochs
     * @param batchSize Batch size for training (0 = full batch)
     * @param validationInputs Optional validation inputs
     * @param validationTargets Optional validation targets
     * @param verbose Print training progress
     * @return Training history (losses per epoch)
     */
    std::vector<float> Train(const std::vector<Tensor>& trainInputs,
                            const std::vector<Tensor>& trainTargets,
                            int epochs = 100,
                            int batchSize = 32,
                            const std::vector<Tensor>* validationInputs = nullptr,
                            const std::vector<Tensor>* validationTargets = nullptr,
                            bool verbose = true) {
        
        if (!m_compiled) {
            throw std::runtime_error("Model must be compiled before training. Call Compile() first.");
        }
        
        if (trainInputs.size() != trainTargets.size()) {
            throw std::invalid_argument("Number of training inputs and targets must match");
        }
        
        if (trainInputs.empty()) {
            throw std::invalid_argument("Training data cannot be empty");
        }
        
        // Validate validation data if provided
        if (validationInputs && validationTargets) {
            if (validationInputs->size() != validationTargets->size()) {
                throw std::invalid_argument("Number of validation inputs and targets must match");
            }
        }
        
        std::vector<float> trainingHistory;
        size_t numSamples = trainInputs.size();
        
        if (verbose) {
            std::cout << "\nStarting Sequential training..." << std::endl;
            std::cout << "Training samples: " << numSamples << std::endl;
            std::cout << "Epochs: " << epochs << std::endl;
            std::cout << "Batch size: " << (batchSize > 0 ? std::to_string(batchSize) : "full batch") << std::endl;
            if (validationInputs && validationTargets) {
                std::cout << "Validation samples: " << validationInputs->size() << std::endl;
            }
            std::cout << std::endl;
        }
        
        // Create progress bar
        std::unique_ptr<kotml::utils::ProgressBar> progressBar;
        if (verbose) {
            // Determine if we should show sample progress (for mini-batch training)
            int totalSamples = (batchSize > 0 && batchSize < static_cast<int>(numSamples)) ? 
                static_cast<int>(numSamples) : 0;
            progressBar = std::make_unique<kotml::utils::ProgressBar>(epochs, totalSamples);
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            SetTraining(true);
            float epochLoss = 0.0f;
            int numBatches = 0;
            int processedSamples = 0;
            
            // Determine actual batch size
            int actualBatchSize = (batchSize > 0) ? batchSize : static_cast<int>(numSamples);
            
            // Training loop
            for (size_t i = 0; i < numSamples; i += actualBatchSize) {
                size_t endIdx = std::min(i + actualBatchSize, numSamples);
                
                // Process batch
                float batchLoss = 0.0f;
                m_optimizer->ZeroGrad();
                
                for (size_t j = i; j < endIdx; ++j) {
                    // Forward pass
                    Tensor prediction = Forward(trainInputs[j]);
                    
                    // Compute loss
                    Tensor loss = m_lossFunction->Forward(prediction, trainTargets[j]);
                    batchLoss += loss[0];
                    
                    // Compute gradients
                    Tensor lossGradients = m_lossFunction->Backward(prediction, trainTargets[j]);
                    
                    // Backward pass through network
                    BackwardPass(trainInputs[j], lossGradients);
                }
                
                // Update parameters
                m_optimizer->Step();
                
                epochLoss += batchLoss / (endIdx - i);
                numBatches++;
                processedSamples = static_cast<int>(endIdx);
                
                // Update progress bar for mini-batch training
                if (verbose && progressBar && batchSize > 0 && batchSize < static_cast<int>(numSamples)) {
                    float currentLoss = epochLoss / numBatches;
                    progressBar->Update(epoch + 1, processedSamples, currentLoss);
                }
            }
            
            epochLoss /= numBatches;
            trainingHistory.push_back(epochLoss);
            
            // Check for numerical instability
            if (std::isnan(epochLoss) || std::isinf(epochLoss)) {
                if (verbose && progressBar) {
                    progressBar->Finish();
                }
                std::cerr << "Warning: Numerical instability detected at epoch " << epoch + 1 
                          << " (loss = " << epochLoss << "). Consider reducing learning rate." << std::endl;
                break;
            }
            
            // Validation
            float validationLoss = 0.0f;
            if (validationInputs && validationTargets) {
                validationLoss = Evaluate(*validationInputs, *validationTargets);
            }
            
            // Update progress bar for epoch-only training or finish epoch for mini-batch
            if (verbose && progressBar) {
                if (batchSize > 0 && batchSize < static_cast<int>(numSamples)) {
                    // Mini-batch training - finish the epoch
                    progressBar->FinishEpoch();
                } else {
                    // Full batch or single batch training - update epoch progress
                    progressBar->Update(epoch + 1, epochLoss);
                }
            }
        }
        
        if (verbose) {
            if (progressBar) {
                progressBar->Finish();
            }
            std::cout << "Sequential training completed!" << std::endl;
        }
        
        return trainingHistory;
    }
    
    /**
     * Evaluate the model on provided data
     * @param inputs Input data for evaluation
     * @param targets Target data for evaluation
     * @return Average loss on the dataset
     */
    float Evaluate(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
        if (!m_compiled) {
            throw std::runtime_error("Model must be compiled before evaluation. Call Compile() first.");
        }
        
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("Number of inputs and targets must match");
        }
        
        if (inputs.empty()) {
            throw std::invalid_argument("Evaluation data cannot be empty");
        }
        
        SetTraining(false);
        float totalLoss = 0.0f;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            Tensor prediction = Forward(inputs[i]);
            Tensor loss = m_lossFunction->Forward(prediction, targets[i]);
            totalLoss += loss[0];
        }
        
        return totalLoss / inputs.size();
    }
    
    /**
     * Make predictions on new data
     * @param inputs Input data for prediction
     * @return Vector of prediction tensors
     */
    std::vector<Tensor> Predict(const std::vector<Tensor>& inputs) {
        if (!m_built) {
            throw std::runtime_error("Sequential must be built before prediction. Call Build() first.");
        }
        
        SetTraining(false);
        std::vector<Tensor> predictions;
        predictions.reserve(inputs.size());
        
        for (const auto& input : inputs) {
            predictions.push_back(Forward(input));
        }
        
        return predictions;
    }
    
    // Check if model is compiled
    bool IsCompiled() const { return m_compiled; }
    
    // Get optimizer (if compiled)
    const optim::Optimizer* GetOptimizer() const { return m_optimizer.get(); }
    
    // Get loss function (if compiled)
    const Loss* GetLossFunction() const { return m_lossFunction.get(); }

    // Methods for adding layers (Builder pattern)
    
    // Add input layer
    Sequential&& Input(size_t inputSize) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::make_unique<InputLayer>(inputSize));
        return std::move(*this);
    }
    
    // Add linear layer
    Sequential&& Linear(size_t inputSize, size_t outputSize, bool useBias = true) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::make_unique<kotml::nn::Linear>(inputSize, outputSize, useBias));
        return std::move(*this);
    }
    
    // Add activation layer
    Sequential&& Activation(ActivationType type) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::make_unique<kotml::nn::Activation>(type));
        return std::move(*this);
    }
    
    // Convenient methods for popular activations
    Sequential&& ReLU() && {
        return std::move(*this).Activation(ActivationType::Relu);
    }
    
    Sequential&& Sigmoid() && {
        return std::move(*this).Activation(ActivationType::Sigmoid);
    }
    
    Sequential&& Tanh() && {
        return std::move(*this).Activation(ActivationType::Tanh);
    }
    
    // Add Dropout layer
    Sequential&& Dropout(float rate) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::make_unique<kotml::nn::Dropout>(rate));
        return std::move(*this);
    }
    
    // Add output layer
    Sequential&& Output(size_t inputSize, size_t outputSize, ActivationType activation = ActivationType::None) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::make_unique<OutputLayer>(inputSize, outputSize, activation));
        return std::move(*this);
    }
    
    // Add arbitrary module
    Sequential&& Add(std::unique_ptr<Module> module) && {
        if (m_built) {
            throw std::runtime_error("Cannot modify Sequential after Build() has been called");
        }
        m_layers.push_back(std::move(module));
        return std::move(*this);
    }
    
    // Build final network
    Sequential Build() && {
        if (m_built) {
            throw std::runtime_error("Build() has already been called");
        }
        m_built = true;
        return std::move(*this);
    }
    
    // Check if network is built
    bool IsBuilt() const { return m_built; }
    
    // Forward pass
    Tensor Forward(const Tensor& input) override {
        if (!m_built) {
            throw std::runtime_error("Sequential must be built before use. Call Build() first.");
        }
        
        Tensor output = input;
        for (auto& layer : m_layers) {
            output = layer->Forward(output);
        }
        return output;
    }
    
    // Get all parameters
    std::vector<Tensor*> Parameters() override {
        std::vector<Tensor*> allParams;
        
        for (auto& layer : m_layers) {
            auto layerParams = layer->Parameters();
            allParams.insert(allParams.end(), layerParams.begin(), layerParams.end());
        }
        
        return allParams;
    }
    
    // Set training mode for all layers
    void SetTraining(bool training) override {
        Module::SetTraining(training);
        for (auto& layer : m_layers) {
            layer->SetTraining(training);
        }
    }
    
    // Zero gradients for all layers
    void ZeroGrad() override {
        for (auto& layer : m_layers) {
            layer->ZeroGrad();
        }
    }
    
    // Get network information
    size_t GetNumLayers() const { return m_layers.size(); }
    
    // Get specific layer
    Module* GetLayer(size_t index) {
        if (index >= m_layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return m_layers[index].get();
    }
    
    const Module* GetLayer(size_t index) const {
        if (index >= m_layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return m_layers[index].get();
    }
    
    // Print network architecture
    void Summary() const {
        if (!m_built) {
            std::cout << "Sequential (not built yet)" << std::endl;
            return;
        }
        
        std::cout << "Sequential Architecture:" << std::endl;
        for (size_t i = 0; i < m_layers.size(); ++i) {
            std::cout << "  Layer " << i << ": " << m_layers[i]->GetName();
            
            // Additional information for some layers
            if (auto* linear = dynamic_cast<kotml::nn::Linear*>(m_layers[i].get())) {
                std::cout << " (" << linear->GetInputSize() << " -> " << linear->GetOutputSize() << ")";
            } else if (auto* input = dynamic_cast<InputLayer*>(m_layers[i].get())) {
                std::cout << " (size: " << input->GetInputSize() << ")";
            } else if (auto* output = dynamic_cast<OutputLayer*>(m_layers[i].get())) {
                std::cout << " (" << output->GetInputSize() << " -> " << output->GetOutputSize() << ")";
            } else if (auto* dropout = dynamic_cast<kotml::nn::Dropout*>(m_layers[i].get())) {
                std::cout << " (rate: " << dropout->GetDropoutRate() << ")";
            }
            
            std::cout << std::endl;
        }
        std::cout << "Total parameters: " << CountParameters() << std::endl;
    }
    
    // Count total number of parameters
    size_t CountParameters() const {
        size_t totalParams = 0;
        for (const auto& layer : m_layers) {
            auto params = const_cast<Module*>(layer.get())->Parameters();
            for (const auto* param : params) {
                totalParams += param->Size();
            }
        }
        return totalParams;
    }
    
    // Get brief summary
    std::string GetName() const override { 
        return "Sequential(" + std::to_string(m_layers.size()) + " layers)"; 
    }

private:
    /**
     * Backward pass for gradient computation using chain rule
     * Implements proper backpropagation algorithm
     */
    void BackwardPass(const Tensor& input, const Tensor& outputGradients) {
        // Store activations for gradient computation
        std::vector<Tensor> activations;
        activations.push_back(input);
        
        // Forward pass to collect activations
        Tensor current = input;
        for (auto& layer : m_layers) {
            current = layer->Forward(current);
            activations.push_back(current);
        }
        
        // Backward pass through layers
        Tensor currentGrad = outputGradients;
        
        // Go through layers in reverse order
        for (int i = static_cast<int>(m_layers.size()) - 1; i >= 0; --i) {
            auto& layer = m_layers[i];
            
            // Get layer parameters
            auto layerParams = layer->Parameters();
            
            if (!layerParams.empty()) {
                // This is a linear layer - compute gradients
                auto* linearLayer = dynamic_cast<kotml::nn::Linear*>(layer.get());
                if (linearLayer) {
                    // Get input to this layer
                    const Tensor& layerInput = activations[i];
                    
                    // Compute weight gradients: dW = input^T * grad_output
                    auto& weightGrad = layerParams[0]->Grad(); // weights
                    if (weightGrad.empty()) {
                        weightGrad.resize(layerParams[0]->Size());
                        std::fill(weightGrad.begin(), weightGrad.end(), 0.0f); // Инициализируем нулями
                    }
                    
                    // For simplicity, use a basic gradient approximation
                    // In a full implementation, this would be proper matrix multiplication
                    size_t inputSize = linearLayer->GetInputSize();
                    size_t outputSize = linearLayer->GetOutputSize();
                    
                    for (size_t out = 0; out < outputSize; ++out) {
                        for (size_t in = 0; in < inputSize; ++in) {
                            size_t weightIdx = out * inputSize + in;
                            if (weightIdx < weightGrad.size()) {
                                float inputVal = (in < layerInput.Size()) ? layerInput[in] : 0.0f;
                                float gradVal = (out < currentGrad.Size()) ? currentGrad[out] : 0.0f;
                                weightGrad[weightIdx] += inputVal * gradVal; // += вместо = для накопления
                            }
                        }
                    }
                    
                    // Compute bias gradients: db = grad_output
                    if (layerParams.size() > 1) {
                        auto& biasGrad = layerParams[1]->Grad(); // bias
                        if (biasGrad.empty()) {
                            biasGrad.resize(layerParams[1]->Size());
                            std::fill(biasGrad.begin(), biasGrad.end(), 0.0f); // Инициализируем нулями
                        }
                        
                        for (size_t out = 0; out < outputSize && out < biasGrad.size(); ++out) {
                            if (out < currentGrad.Size()) {
                                biasGrad[out] += currentGrad[out]; // += вместо = для накопления
                            }
                        }
                    }
                    
                    // Compute input gradients for next layer: dX = W^T * grad_output
                    if (i > 0) { // Not the first layer
                        Tensor nextGrad = Tensor::Zeros({inputSize});
                        
                        for (size_t in = 0; in < inputSize; ++in) {
                            float gradSum = 0.0f;
                            for (size_t out = 0; out < outputSize; ++out) {
                                size_t weightIdx = out * inputSize + in;
                                if (weightIdx < layerParams[0]->Size() && out < currentGrad.Size()) {
                                    gradSum += (*layerParams[0])[weightIdx] * currentGrad[out];
                                }
                            }
                            nextGrad[in] = gradSum;
                        }
                        
                        currentGrad = nextGrad;
                    }
                }
            } else {
                // This is an activation layer - compute derivative
                auto* activationLayer = dynamic_cast<kotml::nn::Activation*>(layer.get());
                if (activationLayer) {
                    // Apply activation derivative
                    const Tensor& layerInput = activations[i];
                    Tensor activationGrad = Tensor::Zeros(currentGrad.Shape());
                    
                    for (size_t j = 0; j < currentGrad.Size() && j < layerInput.Size(); ++j) {
                        float derivative = ComputeActivationDerivative(activationLayer->GetActivationType(), layerInput[j]);
                        activationGrad[j] = currentGrad[j] * derivative;
                    }
                    
                    currentGrad = activationGrad;
                }
            }
        }
    }
    
    /**
     * Compute derivative of activation function
     */
    float ComputeActivationDerivative(ActivationType type, float input) {
        switch (type) {
            case ActivationType::Relu:
                return (input > 0.0f) ? 1.0f : 0.0f;
            case ActivationType::Sigmoid: {
                float sigmoid = 1.0f / (1.0f + std::exp(-input));
                return sigmoid * (1.0f - sigmoid);
            }
            case ActivationType::Tanh: {
                float tanh_val = std::tanh(input);
                return 1.0f - tanh_val * tanh_val;
            }
            default:
                return 1.0f; // Linear activation
        }
    }
};

} // namespace nn
} // namespace kotml 