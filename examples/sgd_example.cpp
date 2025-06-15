#include "kotml/kotml.hpp"
#include "kotml/tensor.hpp"
#include "kotml/nn/linear_layer.hpp"
#include "kotml/optim/sgd.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace kotml;

// Simple training function for demonstration
void TrainStep(nn::Linear& layer, optim::SGD& optimizer) {
    // For demonstration, we'll manually set some gradients
    // In real training, this would be done by the loss.Backward() call
    optimizer.ZeroGrad();
    
    auto params = layer.Parameters();
    for (auto* param : params) {
        auto& grad = param->Grad();
        if (grad.empty()) {
            grad.resize(param->Size());
        }
        
        // Set random gradients for demonstration
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 0.1f);
        
        for (size_t i = 0; i < grad.size(); ++i) {
            grad[i] = dis(gen);
        }
    }
    
    // Update parameters
    optimizer.Step();
    
    std::cout << "Parameters updated successfully" << std::endl;
}

int main() {
    std::cout << "=== SGD Optimizer Example ===" << std::endl;
    
    // Create a simple linear layer
    nn::Linear layer(3, 1);
    
    std::cout << "Layer architecture:" << std::endl;
    std::cout << "Input size: " << layer.GetInputSize() << std::endl;
    std::cout << "Output size: " << layer.GetOutputSize() << std::endl;
    std::cout << "Total parameters: " << layer.CountParameters() << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Basic SGD (no momentum) ===" << std::endl;
    {
        // Create basic SGD optimizer
        optim::SGD optimizer(0.01f);
        optimizer.AddParameters(layer);
        
        std::cout << "Optimizer config: " << optimizer.GetConfig() << std::endl;
        std::cout << "Parameters registered: " << optimizer.GetParameterCount() << std::endl;
        
        // Train for a few steps
        for (int step = 0; step < 3; ++step) {
            std::cout << "Step " << (step + 1) << " - ";
            TrainStep(layer, optimizer);
        }
    }
    
    std::cout << std::endl << "=== SGD with Momentum ===" << std::endl;
    {
        // Create SGD with momentum
        optim::SGD optimizer(0.01f, 0.9f);
        optimizer.AddParameters(layer);
        
        std::cout << "Optimizer config: " << optimizer.GetConfig() << std::endl;
        
        // Train for a few steps
        for (int step = 0; step < 3; ++step) {
            std::cout << "Step " << (step + 1) << " - ";
            TrainStep(layer, optimizer);
        }
    }
    
    std::cout << std::endl << "=== SGD with Momentum and Weight Decay ===" << std::endl;
    {
        // Create SGD with momentum and weight decay
        optim::SGD optimizer(0.01f, 0.9f, 0.0f, 0.0001f);
        optimizer.AddParameters(layer);
        
        std::cout << "Optimizer config: " << optimizer.GetConfig() << std::endl;
        
        // Train for a few steps
        for (int step = 0; step < 3; ++step) {
            std::cout << "Step " << (step + 1) << " - ";
            TrainStep(layer, optimizer);
        }
    }
    
    std::cout << std::endl << "=== Nesterov SGD ===" << std::endl;
    {
        // Create Nesterov SGD
        optim::SGD optimizer(0.01f, 0.9f, 0.0f, 0.0f, true);
        optimizer.AddParameters(layer);
        
        std::cout << "Optimizer config: " << optimizer.GetConfig() << std::endl;
        
        // Train for a few steps
        for (int step = 0; step < 3; ++step) {
            std::cout << "Step " << (step + 1) << " - ";
            TrainStep(layer, optimizer);
        }
    }
    
    std::cout << std::endl << "=== Dynamic Learning Rate Adjustment ===" << std::endl;
    {
        optim::SGD optimizer(0.1f);
        optimizer.AddParameters(layer);
        
        for (int epoch = 0; epoch < 3; ++epoch) {
            // Reduce learning rate every epoch
            float newLr = 0.1f * std::pow(0.5f, epoch);
            optimizer.SetLearningRate(newLr);
            
            std::cout << "Epoch " << (epoch + 1) << " (lr=" << newLr << ") - ";
            TrainStep(layer, optimizer);
        }
    }
    
    std::cout << std::endl << "=== Optimizer State Management ===" << std::endl;
    {
        optim::SGD optimizer(0.01f, 0.9f);
        optimizer.AddParameters(layer);
        
        std::cout << "Initial state:" << std::endl;
        std::cout << "  Learning rate: " << optimizer.GetLearningRate() << std::endl;
        std::cout << "  Momentum: " << optimizer.GetMomentum() << std::endl;
        std::cout << "  Weight decay: " << optimizer.GetWeightDecay() << std::endl;
        std::cout << "  Nesterov: " << (optimizer.IsNesterov() ? "true" : "false") << std::endl;
        
        // Train a few steps to build momentum
        for (int step = 0; step < 2; ++step) {
            TrainStep(layer, optimizer);
        }
        
        std::cout << "Clearing momentum buffers..." << std::endl;
        optimizer.ClearMomentumBuffers();
        
        // Continue training
        std::cout << "After clearing momentum - ";
        TrainStep(layer, optimizer);
    }
    
    std::cout << std::endl << "=== Parameter Validation ===" << std::endl;
    {
        try {
            // This should throw an exception
            optim::SGD optimizer(-0.1f);
            std::cout << "ERROR: Should have thrown exception for negative learning rate!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught exception: " << e.what() << std::endl;
        }
        
        try {
            // This should throw an exception
            optim::SGD optimizer(0.01f, 1.5f);
            std::cout << "ERROR: Should have thrown exception for momentum > 1!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught exception: " << e.what() << std::endl;
        }
        
        try {
            // This should throw an exception (Nesterov requires dampening = 0)
            optim::SGD optimizer(0.01f, 0.9f, 0.1f, 0.0f, true);
            std::cout << "ERROR: Should have thrown exception for Nesterov with dampening!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught exception: " << e.what() << std::endl;
        }
    }
    
    std::cout << std::endl << "=== Parameter Monitoring ===" << std::endl;
    {
        optim::SGD optimizer(0.01f, 0.9f);
        optimizer.AddParameters(layer);
        
        // Show initial parameter values
        auto params = layer.Parameters();
        std::cout << "Initial parameter statistics:" << std::endl;
        for (size_t i = 0; i < params.size(); ++i) {
            auto* param = params[i];
            float sum = 0.0f;
            for (float val : param->Data()) {
                sum += val * val;
            }
            float norm = std::sqrt(sum);
            std::cout << "  Parameter " << i << " norm: " << norm << std::endl;
        }
        
        // Train one step
        TrainStep(layer, optimizer);
        
        // Show updated parameter values
        std::cout << "After optimization step:" << std::endl;
        for (size_t i = 0; i < params.size(); ++i) {
            auto* param = params[i];
            float sum = 0.0f;
            for (float val : param->Data()) {
                sum += val * val;
            }
            float norm = std::sqrt(sum);
            std::cout << "  Parameter " << i << " norm: " << norm << std::endl;
        }
    }
    
    std::cout << std::endl << "=== SGD Example Complete ===" << std::endl;
    
    return 0;
} 