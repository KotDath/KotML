#pragma once

#include "kotml/nn/module.hpp"
#include "kotml/nn/linear_layer.hpp"
#include "kotml/nn/activation_layer.hpp"
#include "kotml/nn/dropout_layer.hpp"
#include "kotml/nn/loss.hpp"
#include "kotml/optim/optimizer.hpp"
#include "kotml/tensor.hpp"
#include "kotml/utils/progress_bar.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>
#include <typeinfo>

namespace kotml {
namespace nn {

// Feed-Forward Network (multi-layer perceptron)
class FFN : public Module {
private:
    std::vector<std::unique_ptr<Module>> m_layers;
    std::vector<size_t> m_layerSizes;
    ActivationType m_hiddenActivation;
    ActivationType m_outputActivation;
    float m_dropoutRate;
    
    // Training components
    std::unique_ptr<optim::Optimizer> m_optimizer;
    std::unique_ptr<Loss> m_lossFunction;
    bool m_compiled = false;

public:
    // Constructor
    FFN(const std::vector<size_t>& layerSizes, 
        ActivationType hiddenActivation = ActivationType::Relu,
        ActivationType outputActivation = ActivationType::None,
        float dropoutRate = 0.0f)
        : m_layerSizes(layerSizes), 
          m_hiddenActivation(hiddenActivation),
          m_outputActivation(outputActivation),
          m_dropoutRate(dropoutRate) {
        
        if (layerSizes.size() < 2) {
            throw std::invalid_argument("FFN requires at least 2 layer sizes (input and output)");
        }
        
        BuildNetwork();
    }
    
    /**
     * Compile the model with optimizer and loss function
     * @param optimizer Unique pointer to optimizer (SGD, Adam, etc.)
     * @param lossFunction Unique pointer to loss function (MSE, BCE, etc.)
     */
    void Compile(std::unique_ptr<optim::Optimizer> optimizer, std::unique_ptr<Loss> lossFunction) {
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
        
        std::cout << "Model compiled successfully:" << std::endl;
        std::cout << "  Optimizer: " << m_optimizer->GetName() << std::endl;
        std::cout << "  Loss function: " << m_lossFunction->GetName() << std::endl;
        std::cout << "  Parameters: " << CountParameters() << std::endl;
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
            std::cout << "\nStarting FFN training..." << std::endl;
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
            std::cout << "FFN training completed!" << std::endl;
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
    
    // Forward pass
    Tensor Forward(const Tensor& input) override {
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
    size_t GetNumLayers() const { return m_layerSizes.size() - 1; }
    const std::vector<size_t>& GetLayerSizes() const { return m_layerSizes; }
    size_t GetInputSize() const { return m_layerSizes.front(); }
    size_t GetOutputSize() const { return m_layerSizes.back(); }
    ActivationType GetHiddenActivation() const { return m_hiddenActivation; }
    ActivationType GetOutputActivation() const { return m_outputActivation; }
    float GetDropoutRate() const { return m_dropoutRate; }
    
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
    
    size_t GetNumModules() const { return m_layers.size(); }
    
    // Print network architecture
    void PrintArchitecture() const {
        std::cout << "FFN Architecture:" << std::endl;
        std::cout << "Input size: " << GetInputSize() << std::endl;
        
        for (size_t i = 0; i < m_layerSizes.size() - 1; ++i) {
            std::cout << "Layer " << (i + 1) << ": " << m_layerSizes[i] << " -> " << m_layerSizes[i + 1];
            
            // Show activation
            if (i < m_layerSizes.size() - 2) { // Hidden layers
                std::cout << " + " << GetActivationName(m_hiddenActivation);
            } else { // Output layer
                if (m_outputActivation != ActivationType::None) {
                    std::cout << " + " << GetActivationName(m_outputActivation);
                }
            }
            
            // Show dropout
            if (m_dropoutRate > 0.0f && i < m_layerSizes.size() - 2) {
                std::cout << " + Dropout(" << m_dropoutRate << ")";
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
    
    std::string GetName() const override { return "FFN"; }

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
    
    void BuildNetwork() {
        m_layers.clear();
        
        // Create layers
        for (size_t i = 0; i < m_layerSizes.size() - 1; ++i) {
            // Linear layer
            m_layers.push_back(std::make_unique<Linear>(m_layerSizes[i], m_layerSizes[i + 1]));
            
            // Activation
            ActivationType activation = (i == m_layerSizes.size() - 2) ? m_outputActivation : m_hiddenActivation;
            if (activation != ActivationType::None) {
                m_layers.push_back(std::make_unique<Activation>(activation));
            }
            
            // Dropout (only for hidden layers)
            if (m_dropoutRate > 0.0f && i < m_layerSizes.size() - 2) {
                m_layers.push_back(std::make_unique<Dropout>(m_dropoutRate));
            }
        }
    }
    
    std::string GetActivationName(ActivationType type) const {
        switch (type) {
            case ActivationType::Relu: return "ReLU";
            case ActivationType::Sigmoid: return "Sigmoid";
            case ActivationType::Tanh: return "Tanh";
            default: return "None";
        }
    }
};

} // namespace nn
} // namespace kotml 