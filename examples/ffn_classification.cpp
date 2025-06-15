#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace kotml;
using namespace kotml::nn;

// Simple dataset generation
std::pair<std::vector<Tensor>, std::vector<Tensor>> GenerateDataset(size_t numSamples, size_t inputSize) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < numSamples; ++i) {
        std::vector<float> inputData(inputSize);
        for (size_t j = 0; j < inputSize; ++j) {
            inputData[j] = dis(gen);
        }
        
        // Simple classification rule: sum > 0 -> class 1, else class 0
        float sum = 0.0f;
        for (float val : inputData) {
            sum += val;
        }
        
        inputs.emplace_back(inputData, std::vector<size_t>{inputSize}, true);
        
        // One-hot encoding for 2 classes
        std::vector<float> targetData = {0.0f, 0.0f};
        targetData[sum > 0.0f ? 1 : 0] = 1.0f;
        targets.emplace_back(targetData, std::vector<size_t>{2}, false);
    }
    
    return {inputs, targets};
}

// Simple loss function (Mean Squared Error)
float CalculateLoss(const std::vector<Tensor>& predictions, const std::vector<Tensor>& targets) {
    float totalLoss = 0.0f;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].Size(); ++j) {
            float diff = predictions[i][j] - targets[i][j];
            totalLoss += diff * diff;
        }
    }
    
    return totalLoss / predictions.size();
}

// Calculate accuracy
float CalculateAccuracy(const std::vector<Tensor>& predictions, const std::vector<Tensor>& targets) {
    size_t correct = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        // Find predicted class (argmax)
        size_t predClass = 0;
        float maxPred = predictions[i][0];
        for (size_t j = 1; j < predictions[i].Size(); ++j) {
            if (predictions[i][j] > maxPred) {
                maxPred = predictions[i][j];
                predClass = j;
            }
        }
        
        // Find true class (argmax)
        size_t trueClass = 0;
        float maxTrue = targets[i][0];
        for (size_t j = 1; j < targets[i].Size(); ++j) {
            if (targets[i][j] > maxTrue) {
                maxTrue = targets[i][j];
                trueClass = j;
            }
        }
        
        if (predClass == trueClass) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

int main() {
    std::cout << "=== FFN Binary Classification Demo ===" << std::endl;
    
    // 1. Dataset generation
    std::cout << "\n1. Generating dataset..." << std::endl;
    
    const size_t inputSize = 10;
    const size_t numSamples = 1000;
    const size_t numClasses = 2;
    
    auto [trainInputs, trainTargets] = GenerateDataset(numSamples, inputSize);
    auto [testInputs, testTargets] = GenerateDataset(200, inputSize);
    
    std::cout << "Training samples: " << trainInputs.size() << std::endl;
    std::cout << "Test samples: " << testInputs.size() << std::endl;
    std::cout << "Input size: " << inputSize << std::endl;
    std::cout << "Number of classes: " << numClasses << std::endl;
    
    // 2. Creating neural network
    std::cout << "\n2. Creating neural network..." << std::endl;
    
    FFN classifier({inputSize, 32, 16, numClasses}, 
                   ActivationType::Relu,      // Hidden activation
                   ActivationType::Sigmoid);  // Output activation
    
    std::cout << "Network architecture:" << std::endl;
    classifier.PrintArchitecture();
    
    // 3. Initial evaluation
    std::cout << "\n3. Initial evaluation (before training)..." << std::endl;
    
    std::vector<Tensor> initialPredictions;
    classifier.SetTraining(false);
    
    for (const auto& input : testInputs) {
        Tensor prediction = classifier.Forward(input);
        initialPredictions.push_back(prediction);
    }
    
    float initialLoss = CalculateLoss(initialPredictions, testTargets);
    float initialAccuracy = CalculateAccuracy(initialPredictions, testTargets);
    
    std::cout << "Initial loss: " << initialLoss << std::endl;
    std::cout << "Initial accuracy: " << (initialAccuracy * 100.0f) << "%" << std::endl;
    
    // 4. Training simulation (simplified)
    std::cout << "\n4. Training simulation..." << std::endl;
    
    const size_t numEpochs = 10;
    const float learningRate = 0.01f;
    
    classifier.SetTraining(true);
    
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {
        std::vector<Tensor> epochPredictions;
        float epochLoss = 0.0f;
        
        // Forward pass for all training samples
        for (size_t i = 0; i < trainInputs.size(); ++i) {
            Tensor prediction = classifier.Forward(trainInputs[i]);
            epochPredictions.push_back(prediction);
            
            // Calculate loss for this sample
            for (size_t j = 0; j < prediction.Size(); ++j) {
                float diff = prediction[j] - trainTargets[i][j];
                epochLoss += diff * diff;
            }
        }
        
        epochLoss /= trainInputs.size();
        
        // Simple parameter update simulation (not real backpropagation)
        auto parameters = classifier.Parameters();
        for (auto* param : parameters) {
            for (size_t i = 0; i < param->Size(); ++i) {
                // Add small random noise to simulate learning
                std::random_device rd;
                std::mt19937 gen(rd());
                std::normal_distribution<float> noise(0.0f, learningRate * 0.1f);
                param->Data()[i] += noise(gen);
            }
        }
        
        if (epoch % 2 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << epochLoss << std::endl;
        }
    }
    
    // 5. Final evaluation
    std::cout << "\n5. Final evaluation (after training)..." << std::endl;
    
    std::vector<Tensor> finalPredictions;
    classifier.SetTraining(false);
    
    for (const auto& input : testInputs) {
        Tensor prediction = classifier.Forward(input);
        finalPredictions.push_back(prediction);
    }
    
    float finalLoss = CalculateLoss(finalPredictions, testTargets);
    float finalAccuracy = CalculateAccuracy(finalPredictions, testTargets);
    
    std::cout << "Final loss: " << finalLoss << std::endl;
    std::cout << "Final accuracy: " << (finalAccuracy * 100.0f) << "%" << std::endl;
    
    // 6. Sample predictions
    std::cout << "\n6. Sample predictions:" << std::endl;
    
    for (size_t i = 0; i < 5 && i < testInputs.size(); ++i) {
        Tensor prediction = classifier.Forward(testInputs[i]);
        
        std::cout << "Sample " << i << ":" << std::endl;
        std::cout << "  Input sum: ";
        float inputSum = 0.0f;
        for (size_t j = 0; j < testInputs[i].Size(); ++j) {
            inputSum += testInputs[i][j];
        }
        std::cout << inputSum << std::endl;
        
        std::cout << "  True class: " << (testTargets[i][1] > 0.5f ? 1 : 0) << std::endl;
        std::cout << "  Predicted: [" << prediction[0] << ", " << prediction[1] << "]" << std::endl;
        std::cout << "  Predicted class: " << (prediction[1] > prediction[0] ? 1 : 0) << std::endl;
        std::cout << std::endl;
    }
    
    // 7. Network analysis
    std::cout << "\n7. Network analysis:" << std::endl;
    
    std::cout << "Total parameters: " << classifier.CountParameters() << std::endl;
    std::cout << "Number of layers: " << classifier.GetNumLayers() << std::endl;
    std::cout << "Input size: " << classifier.GetInputSize() << std::endl;
    std::cout << "Output size: " << classifier.GetOutputSize() << std::endl;
    
    auto parameters = classifier.Parameters();
    std::cout << "Parameter tensors: " << parameters.size() << std::endl;
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << "  Tensor " << i << ": ";
        for (size_t j = 0; j < parameters[i]->Shape().size(); ++j) {
            std::cout << parameters[i]->Shape()[j];
            if (j < parameters[i]->Shape().size() - 1) std::cout << "x";
        }
        std::cout << " (" << parameters[i]->Size() << " elements)" << std::endl;
    }
    
    std::cout << "\n=== Classification demo completed ===" << std::endl;
    
    return 0;
} 