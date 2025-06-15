#include "kotml/kotml.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>

using namespace kotml;

int main() {
    std::cout << "=== KotML Progress Bar Demo ===" << std::endl;
    std::cout << "Demonstrating progress bar during training" << std::endl << std::endl;
    
    // Create simple XOR dataset
    std::vector<Tensor> inputs = {
        Tensor({0.0f, 0.0f}, {1, 2}),
        Tensor({0.0f, 1.0f}, {1, 2}),
        Tensor({1.0f, 0.0f}, {1, 2}),
        Tensor({1.0f, 1.0f}, {1, 2})
    };
    
    std::vector<Tensor> targets = {
        Tensor({0.0f}, {1, 1}),  // 0 XOR 0 = 0
        Tensor({1.0f}, {1, 1}),  // 0 XOR 1 = 1
        Tensor({1.0f}, {1, 1}),  // 1 XOR 0 = 1
        Tensor({0.0f}, {1, 1})   // 1 XOR 1 = 0
    };
    
    std::cout << "1. FFN Training with Progress Bar (Full Batch)" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Create FFN model
    FFN ffnModel({2, 8, 1}, ActivationType::Relu, ActivationType::Sigmoid);
    ffnModel.Compile(std::make_unique<SGD>(0.01f), std::make_unique<BCELoss>());
    
    // Train with progress bar (full batch - epoch progress only)
    ffnModel.Train(inputs, targets, 200, 0, nullptr, nullptr, true);
    
    std::cout << std::endl << std::endl;
    
    std::cout << "2. Sequential Training with Progress Bar (Mini-batch)" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Create Sequential model
    auto seqModel = Sequential()
        .Linear(2, 8)
        .ReLU()
        .Linear(8, 1)
        .Sigmoid()
        .Build();
    
    seqModel.Compile(std::make_unique<SGD>(0.01f), std::make_unique<BCELoss>());
    
    // Create larger dataset for mini-batch demonstration
    std::vector<Tensor> largeInputs;
    std::vector<Tensor> largeTargets;
    
    // Replicate XOR data 25 times (100 samples total)
    for (int i = 0; i < 2500; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            largeInputs.push_back(inputs[j]);
            largeTargets.push_back(targets[j]);
        }
    }
    
    std::cout << "Training on " << largeInputs.size() << " samples with batch size 10" << std::endl;
    
    // Train with mini-batch (will show sample progress within epochs)
    seqModel.Train(largeInputs, largeTargets, 10, 10, nullptr, nullptr, true);
    
    std::cout << std::endl << std::endl;
    
    std::cout << "3. Slow Training Demo (to see real-time updates)" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // Create another Sequential model for slow demo
    auto slowModel = Sequential()
        .Linear(2, 16)
        .ReLU()
        .Linear(16, 8)
        .ReLU()
        .Linear(8, 1)
        .Sigmoid()
        .Build();
    
    slowModel.Compile(std::make_unique<SGD>(0.01f), std::make_unique<BCELoss>());
    
    // Create even larger dataset for slower demonstration
    std::vector<Tensor> veryLargeInputs;
    std::vector<Tensor> veryLargeTargets;
    
    // Replicate XOR data 50 times (200 samples total)
    for (int i = 0; i < 2500; ++i) {
        for (size_t j = 0; j < inputs.size(); ++j) {
            veryLargeInputs.push_back(inputs[j]);
            veryLargeTargets.push_back(targets[j]);
        }
    }
    
    std::cout << "Training on " << veryLargeInputs.size() << " samples with batch size 20" << std::endl;
    std::cout << "Watch the progress bar update in real-time!" << std::endl;
    
    // Train with smaller batch size for more frequent updates
    slowModel.Train(veryLargeInputs, veryLargeTargets, 10, 200, nullptr, nullptr, true);
    
    std::cout << std::endl;
    std::cout << "Demo completed! The progress bar shows:" << std::endl;
    std::cout << "- Epoch X / Y format" << std::endl;
    std::cout << "- [===...] progress bar" << std::endl;
    std::cout << "- Sample progress for mini-batch training" << std::endl;
    std::cout << "- Current loss value" << std::endl;
    std::cout << "- Updates in the same line (no new lines during epoch)" << std::endl;
    
    return 0;
} 