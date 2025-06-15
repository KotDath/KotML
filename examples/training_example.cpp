/**
 * KotML Training Example
 * 
 * Demonstrates the new Compile() and Train() methods for FFN and Sequential classes
 * Shows complete training workflows with different optimizers and loss functions
 */

#include "kotml/kotml.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

// Generate synthetic regression data
std::pair<std::vector<Tensor>, std::vector<Tensor>> GenerateRegressionData(size_t numSamples, size_t inputDim) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < numSamples; ++i) {
        // Generate random input
        std::vector<float> inputData(inputDim);
        for (size_t j = 0; j < inputDim; ++j) {
            inputData[j] = dist(gen);
        }
        inputs.emplace_back(inputData, std::vector<size_t>{1, inputDim});
        
        // Generate target: simple linear combination + noise
        float target = 0.0f;
        for (size_t j = 0; j < inputDim; ++j) {
            target += inputData[j] * (j + 1) * 0.5f; // Different weights for each feature
        }
        target += dist(gen) * 0.1f; // Add noise
        
        targets.emplace_back(std::vector<float>{target}, std::vector<size_t>{1, 1});
    }
    
    return {inputs, targets};
}

// Generate synthetic classification data
std::pair<std::vector<Tensor>, std::vector<Tensor>> GenerateClassificationData(size_t numSamples, size_t inputDim, size_t numClasses) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> classDist(0, numClasses - 1);
    
    for (size_t i = 0; i < numSamples; ++i) {
        // Generate random input
        std::vector<float> inputData(inputDim);
        for (size_t j = 0; j < inputDim; ++j) {
            inputData[j] = dist(gen);
        }
        inputs.emplace_back(inputData, std::vector<size_t>{1, inputDim});
        
        // Generate one-hot encoded target
        int targetClass = classDist(gen);
        std::vector<float> targetData(numClasses, 0.0f);
        targetData[targetClass] = 1.0f;
        
        targets.emplace_back(targetData, std::vector<size_t>{1, numClasses});
    }
    
    return {inputs, targets};
}

void DemoFFNRegression() {
    std::cout << "=== FFN Regression Demo ===" << std::endl;
    
    // Generate training data
    auto [trainInputs, trainTargets] = GenerateRegressionData(1000, 5);
    auto [valInputs, valTargets] = GenerateRegressionData(200, 5);
    
    std::cout << "Generated " << trainInputs.size() << " training samples" << std::endl;
    std::cout << "Generated " << valInputs.size() << " validation samples" << std::endl;
    
    // Create FFN model
    FFN model({5, 32, 16, 1}, ActivationType::Relu, ActivationType::None, 0.1f);
    
    std::cout << "\nModel architecture:" << std::endl;
    model.PrintArchitecture();
    
    // Compile model
    auto optimizer = std::make_unique<SGD>(0.001f, 0.9f, 0.0f, 1e-4f, false); // lr=0.001 (уменьшено), momentum=0.9, dampening=0.0, weight_decay=1e-4, nesterov=false
    auto lossFunction = std::make_unique<MSELoss>();
    
    model.Compile(std::move(optimizer), std::move(lossFunction));
    
    // Train model
    std::cout << "\nStarting training..." << std::endl;
    auto history = model.Train(trainInputs, trainTargets, 500, 32, &valInputs, &valTargets, true);
    
    // Evaluate model
    float finalLoss = model.Evaluate(valInputs, valTargets);
    std::cout << "\nFinal validation loss: " << finalLoss << std::endl;
    
    // Make predictions
    auto predictions = model.Predict({valInputs[0], valInputs[1], valInputs[2]});
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Target: " << valTargets[i][0] << ", Prediction: " << predictions[i][0] << std::endl;
    }
    
    std::cout << std::endl;
}

void DemoSequentialClassification() {
    std::cout << "=== Sequential Classification Demo ===" << std::endl;
    
    // Generate training data
    auto [trainInputs, trainTargets] = GenerateClassificationData(800, 4, 3);
    auto [valInputs, valTargets] = GenerateClassificationData(200, 4, 3);
    
    std::cout << "Generated " << trainInputs.size() << " training samples" << std::endl;
    std::cout << "Generated " << valInputs.size() << " validation samples" << std::endl;
    
    // Create Sequential model using builder pattern
    auto model = Sequential()
        .Input(4)
        .Linear(4, 16)
        .ReLU()
        .Dropout(0.2f)
        .Linear(16, 8)
        .ReLU()
        .Linear(8, 3)
        .Sigmoid()
        .Build();
    
    std::cout << "\nModel architecture:" << std::endl;
    model.Summary();
    
    // Compile model with different optimizer
    auto optimizer = std::make_unique<SGD>(0.01f); // lr=0.01, простой SGD без momentum
    auto lossFunction = std::make_unique<BCELoss>();
    
    model.Compile(std::move(optimizer), std::move(lossFunction));
    
    // Train model
    std::cout << "\nStarting training..." << std::endl;
    auto history = model.Train(trainInputs, trainTargets, 300, 16, &valInputs, &valTargets, true);
    
    // Evaluate model
    float finalLoss = model.Evaluate(valInputs, valTargets);
    std::cout << "\nFinal validation loss: " << finalLoss << std::endl;
    
    // Make predictions
    auto predictions = model.Predict({valInputs[0], valInputs[1], valInputs[2]});
    std::cout << "\nSample predictions:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  Target: [";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << valTargets[i][j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "], Prediction: [";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::fixed << std::setprecision(3) << predictions[i][j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << std::endl;
}

void DemoAdvancedFeatures() {
    std::cout << "=== Advanced Features Demo ===" << std::endl;
    
    // Generate small dataset for quick demo
    auto [trainInputs, trainTargets] = GenerateRegressionData(100, 3);
    
    // Create two identical models for comparison
    FFN model1({3, 8, 1}, ActivationType::Tanh);
    FFN model2({3, 8, 1}, ActivationType::Tanh);
    
    std::cout << "Comparing different optimizers and loss functions..." << std::endl;
    
    // Model 1: SGD with MSE
    auto optimizer1 = std::make_unique<SGD>(0.01f);
    auto loss1 = std::make_unique<MSELoss>();
    model1.Compile(std::move(optimizer1), std::move(loss1));
    
    // Model 2: SGD with MAE
    auto optimizer2 = std::make_unique<SGD>(0.01f);
    auto loss2 = std::make_unique<MAELoss>();
    model2.Compile(std::move(optimizer2), std::move(loss2));
    
    std::cout << "\nTraining Model 1 (MSE Loss):" << std::endl;
    auto history1 = model1.Train(trainInputs, trainTargets, 20, 0, nullptr, nullptr, false);
    
    std::cout << "Training Model 2 (MAE Loss):" << std::endl;
    auto history2 = model2.Train(trainInputs, trainTargets, 20, 0, nullptr, nullptr, false);
    
    std::cout << "\nFinal losses:" << std::endl;
    std::cout << "  Model 1 (MSE): " << history1.back() << std::endl;
    std::cout << "  Model 2 (MAE): " << history2.back() << std::endl;
    
    std::cout << std::endl;
}

void DemoErrorHandling() {
    std::cout << "=== Error Handling Demo ===" << std::endl;
    
    FFN model({2, 4, 1});
    
    // Try to train without compiling
    try {
        auto [inputs, targets] = GenerateRegressionData(10, 2);
        model.Train(inputs, targets, 5);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Caught expected error: " << e.what() << std::endl;
    }
    
    // Compile with null optimizer
    try {
        auto loss = std::make_unique<MSELoss>();
        model.Compile(nullptr, std::move(loss));
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Caught expected error: " << e.what() << std::endl;
    }
    
    // Try to train with mismatched data
    try {
        auto optimizer = std::make_unique<SGD>(0.01f);
        auto loss = std::make_unique<MSELoss>();
        model.Compile(std::move(optimizer), std::move(loss));
        
        auto [inputs, targets] = GenerateRegressionData(10, 2);
        targets.pop_back(); // Remove one target to create mismatch
        
        model.Train(inputs, targets, 5);
        std::cout << "ERROR: Should have thrown exception!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✓ Caught expected error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

void DemoPerformanceComparison() {
    std::cout << "=== Performance Comparison ===" << std::endl;
    
    // Generate larger dataset
    auto [trainInputs, trainTargets] = GenerateRegressionData(1000, 10);
    
    // FFN model
    FFN ffnModel({10, 20, 10, 1});
    auto ffnOptimizer = std::make_unique<SGD>(0.01f);
    auto ffnLoss = std::make_unique<MSELoss>();
    ffnModel.Compile(std::move(ffnOptimizer), std::move(ffnLoss));
    
    // Sequential model with equivalent architecture
    auto seqModel = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Linear(20, 10)
        .ReLU()
        .Linear(10, 1)
        .Build();
    
    auto seqOptimizer = std::make_unique<SGD>(0.01f);
    auto seqLoss = std::make_unique<MSELoss>();
    seqModel.Compile(std::move(seqOptimizer), std::move(seqLoss));
    
    // Time FFN training
    auto start = std::chrono::high_resolution_clock::now();
    ffnModel.Train(trainInputs, trainTargets, 10, 32, nullptr, nullptr, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto ffnTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Time Sequential training
    start = std::chrono::high_resolution_clock::now();
    seqModel.Train(trainInputs, trainTargets, 10, 32, nullptr, nullptr, false);
    end = std::chrono::high_resolution_clock::now();
    auto seqTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time comparison:" << std::endl;
    std::cout << "  FFN: " << ffnTime.count() << " ms" << std::endl;
    std::cout << "  Sequential: " << seqTime.count() << " ms" << std::endl;
    
    std::cout << std::endl;
}

int main() {
    std::cout << "KotML Training System Demo" << std::endl;
    std::cout << "=========================" << std::endl << std::endl;
    
    try {
        // Run all demos
        DemoFFNRegression();
        DemoSequentialClassification();
        DemoAdvancedFeatures();
        DemoErrorHandling();
        DemoPerformanceComparison();
        
        std::cout << "All demos completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 