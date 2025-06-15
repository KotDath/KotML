/**
 * KotML Linear Regression Example
 * 
 * Simple example demonstrating training a model to learn y = 2x + 1
 * Shows both FFN and Sequential approaches
 */

#include "kotml/kotml.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

// Generate training data for y = 2x + 1 with some noise
std::pair<std::vector<Tensor>, std::vector<Tensor>> GenerateLinearData(size_t numSamples, float noise = 0.1f) {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> xDist(-5.0f, 5.0f);  // x values from -5 to 5
    std::normal_distribution<float> noiseDist(0.0f, noise);    // Gaussian noise
    
    for (size_t i = 0; i < numSamples; ++i) {
        float x = xDist(gen);
        float y = 2.0f * x + 1.0f + noiseDist(gen);  // y = 2x + 1 + noise
        
        inputs.emplace_back(std::vector<float>{x}, std::vector<size_t>{1, 1});
        targets.emplace_back(std::vector<float>{y}, std::vector<size_t>{1, 1});
    }
    
    return {inputs, targets};
}

void DemoFFNLinearRegression() {
    std::cout << "=== FFN Linear Regression Demo ===" << std::endl;
    std::cout << "Learning function: y = 2x + 1" << std::endl << std::endl;
    
    // Generate training and validation data
    auto [trainInputs, trainTargets] = GenerateLinearData(1000, 0.1f);
    auto [valInputs, valTargets] = GenerateLinearData(200, 0.1f);
    
    std::cout << "Generated " << trainInputs.size() << " training samples" << std::endl;
    std::cout << "Generated " << valInputs.size() << " validation samples" << std::endl;
    
    // Create simple FFN model: 1 input -> 1 output (linear layer only)
    FFN model({1, 1}, ActivationType::None, ActivationType::None, 0.0f);
    
    std::cout << "\nModel architecture:" << std::endl;
    model.PrintArchitecture();
    
    // Compile model with small learning rate for stable convergence
    auto optimizer = std::make_unique<SGD>(0.01f);  // Simple SGD
    auto lossFunction = std::make_unique<MSELoss>();
    
    model.Compile(std::move(optimizer), std::move(lossFunction));
    
    // Train model
    std::cout << "\nStarting training..." << std::endl;
    auto history = model.Train(trainInputs, trainTargets, 100, 32, &valInputs, &valTargets, true);
    
    // Evaluate model
    float finalLoss = model.Evaluate(valInputs, valTargets);
    std::cout << "\nFinal validation loss: " << std::fixed << std::setprecision(6) << finalLoss << std::endl;
    
    // Test predictions on specific values
    std::cout << "\nTesting learned function:" << std::endl;
    std::cout << "Expected: y = 2x + 1" << std::endl;
    std::cout << "x\tExpected\tPredicted\tError" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<float> testValues = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    for (float x : testValues) {
        Tensor input({x}, {1, 1});
        auto prediction = model.Predict({input});
        
        float expected = 2.0f * x + 1.0f;
        float predicted = prediction[0][0];
        float error = std::abs(expected - predicted);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << x << "\t" << expected << "\t\t" << predicted << "\t\t" << error << std::endl;
    }
    
    std::cout << std::endl;
}

void DemoSequentialLinearRegression() {
    std::cout << "=== Sequential Linear Regression Demo ===" << std::endl;
    std::cout << "Learning function: y = 2x + 1" << std::endl << std::endl;
    
    // Generate training data
    auto [trainInputs, trainTargets] = GenerateLinearData(800, 0.1f);
    auto [valInputs, valTargets] = GenerateLinearData(200, 0.1f);
    
    std::cout << "Generated " << trainInputs.size() << " training samples" << std::endl;
    std::cout << "Generated " << valInputs.size() << " validation samples" << std::endl;
    
    // Create Sequential model with single linear layer
    auto model = Sequential()
        .Linear(1, 1)  // 1 input -> 1 output
        .Build();
    
    std::cout << "\nModel architecture:" << std::endl;
    model.Summary();
    
    // Compile model
    auto optimizer = std::make_unique<SGD>(0.01f);
    auto lossFunction = std::make_unique<MSELoss>();
    
    model.Compile(std::move(optimizer), std::move(lossFunction));
    
    // Train model
    std::cout << "\nStarting training..." << std::endl;
    auto history = model.Train(trainInputs, trainTargets, 1000, 32, &valInputs, &valTargets, true);
    
    // Evaluate model
    float finalLoss = model.Evaluate(valInputs, valTargets);
    std::cout << "\nFinal validation loss: " << std::fixed << std::setprecision(6) << finalLoss << std::endl;
    
    // Test predictions
    std::cout << "\nTesting learned function:" << std::endl;
    std::cout << "Expected: y = 2x + 1" << std::endl;
    std::cout << "x\tExpected\tPredicted\tError" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<float> testValues = {-3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.0f};
    for (float x : testValues) {
        Tensor input({x}, {1, 1});
        auto prediction = model.Predict({input});
        
        float expected = 2.0f * x + 1.0f;
        float predicted = prediction[0][0];
        float error = std::abs(expected - predicted);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << x << "\t" << expected << "\t\t" << predicted << "\t\t" << error << std::endl;
    }
    
    std::cout << std::endl;
}

void DemoNonLinearRegression() {
    std::cout << "=== Non-Linear Regression Demo ===" << std::endl;
    std::cout << "Learning function: y = 2x + 1 (using hidden layer)" << std::endl << std::endl;
    
    // Generate training data
    auto [trainInputs, trainTargets] = GenerateLinearData(1000, 0.05f);
    auto [valInputs, valTargets] = GenerateLinearData(200, 0.05f);
    
    std::cout << "Generated " << trainInputs.size() << " training samples" << std::endl;
    
    // Create model with hidden layer (can still learn linear function)
    auto model = Sequential()
        .Linear(1, 8)    // 1 -> 8 hidden units
        .ReLU()          // Non-linear activation
        .Linear(8, 1)    // 8 -> 1 output
        .Build();
    
    std::cout << "\nModel architecture:" << std::endl;
    model.Summary();
    
    // Compile with slightly smaller learning rate due to more parameters
    auto optimizer = std::make_unique<SGD>(0.005f, 0.9f);  // With momentum
    auto lossFunction = std::make_unique<MSELoss>();
    
    model.Compile(std::move(optimizer), std::move(lossFunction));
    
    // Train model
    std::cout << "\nStarting training..." << std::endl;
    auto history = model.Train(trainInputs, trainTargets, 150, 32, &valInputs, &valTargets, true);
    
    // Evaluate model
    float finalLoss = model.Evaluate(valInputs, valTargets);
    std::cout << "\nFinal validation loss: " << std::fixed << std::setprecision(6) << finalLoss << std::endl;
    
    // Test predictions
    std::cout << "\nTesting learned function:" << std::endl;
    std::cout << "Expected: y = 2x + 1" << std::endl;
    std::cout << "x\tExpected\tPredicted\tError" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    std::vector<float> testValues = {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f, 5.0f};
    for (float x : testValues) {
        Tensor input({x}, {1, 1});
        auto prediction = model.Predict({input});
        
        float expected = 2.0f * x + 1.0f;
        float predicted = prediction[0][0];
        float error = std::abs(expected - predicted);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << x << "\t" << expected << "\t\t" << predicted << "\t\t" << error << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "KotML Linear Regression Example" << std::endl;
    std::cout << "===============================" << std::endl << std::endl;
    
    try {
        // Run all demos
        DemoFFNLinearRegression();
        DemoSequentialLinearRegression();
        DemoNonLinearRegression();
        
        std::cout << "All linear regression demos completed successfully!" << std::endl;
        std::cout << "\nKey observations:" << std::endl;
        std::cout << "- Simple linear models (1->1) learn y=2x+1 very accurately" << std::endl;
        std::cout << "- Non-linear models can also learn linear functions" << std::endl;
        std::cout << "- Final predictions should be very close to expected values" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 