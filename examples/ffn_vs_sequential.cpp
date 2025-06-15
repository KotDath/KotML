#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;
using namespace kotml::nn;

int main() {
    std::cout << "=== FFN vs Sequential Comparison ===" << std::endl;
    
    // ========================================
    // 1. Simple equivalent networks
    // ========================================
    std::cout << "\n1. Equivalent simple networks:" << std::endl;
    
    // FFN version
    FFN ffnSimple({10, 20, 5}, ActivationType::Relu);
    
    // Sequential version (equivalent)
    auto seqSimple = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Linear(20, 5)
        .Build();
    
    std::cout << "FFN simple network:" << std::endl;
    ffnSimple.PrintArchitecture();
    
    std::cout << "\nSequential simple network:" << std::endl;
    seqSimple.Summary();
    
    // Check that results are structurally identical
    Tensor testInput({10}, 1.0f);
    std::cout << "FFN parameters: " << ffnSimple.CountParameters() << std::endl;
    std::cout << "Sequential parameters: " << seqSimple.CountParameters() << std::endl;
    
    // ========================================
    // 2. What FFN CANNOT do
    // ========================================
    std::cout << "\n\n2. Sequential-only capabilities:" << std::endl;
    
    // Different activations for different layers
    auto mixedActivations = Sequential()
        .Linear(10, 32)
        .ReLU()           // ReLU for first layer
        .Linear(32, 16)
        .Tanh()           // Tanh for second layer
        .Linear(16, 8)
        .Sigmoid()        // Sigmoid for third layer
        .Linear(8, 3)
        .Build();
    
    std::cout << "Network with mixed activations (FFN cannot do):" << std::endl;
    mixedActivations.Summary();
    
    // Network with Dropout (regularization)
    auto withDropout = Sequential()
        .Linear(784, 256)
        .ReLU()
        .Dropout(0.3f)    // Dropout - FFN doesn't support!
        .Linear(256, 128)
        .ReLU()
        .Dropout(0.2f)
        .Linear(128, 10)
        .Build();
    
    std::cout << "\nNetwork with Dropout (FFN cannot do):" << std::endl;
    withDropout.Summary();
    
    // Network with input validation
    auto withValidation = Sequential()
        .Input(784)       // Input validation - FFN doesn't have
        .Linear(784, 128)
        .ReLU()
        .Output(128, 10, ActivationType::Sigmoid)  // Composite output
        .Build();
    
    std::cout << "\nNetwork with validation and composite output:" << std::endl;
    withValidation.Summary();
    
    // ========================================
    // 3. FFN advantages
    // ========================================
    std::cout << "\n\n3. FFN advantages:" << std::endl;
    
    // Very simple creation of large networks
    FFN deepFFN({100, 512, 256, 128, 64, 32, 10}, ActivationType::Relu);
    std::cout << "Deep FFN (simple creation):" << std::endl;
    deepFFN.PrintArchitecture();
    
    // Same network in Sequential (more verbose)
    auto deepSeq = Sequential()
        .Linear(100, 512).ReLU()
        .Linear(512, 256).ReLU()
        .Linear(256, 128).ReLU()
        .Linear(128, 64).ReLU()
        .Linear(64, 32).ReLU()
        .Linear(32, 10)
        .Build();
    
    std::cout << "\nSame network in Sequential (more verbose):" << std::endl;
    deepSeq.Summary();
    
    // ========================================
    // 4. Performance and usage testing
    // ========================================
    std::cout << "\n\n4. Performance testing:" << std::endl;
    
    Tensor input784({784}, 0.5f);
    
    // FFN - training modes
    ffnSimple.SetTraining(true);
    std::cout << "FFN in training mode: " << (ffnSimple.IsTraining() ? "Yes" : "No") << std::endl;
    
    // Sequential - training modes (with Dropout)
    withDropout.SetTraining(true);
    std::cout << "Sequential in training mode: " << (withDropout.IsTraining() ? "Yes" : "No") << std::endl;
    
    // Testing Dropout effect
    Tensor dropoutInput({784}, 1.0f);
    
    withDropout.SetTraining(true);
    Tensor trainOutput = withDropout.Forward(dropoutInput);
    
    withDropout.SetTraining(false);
    Tensor inferOutput = withDropout.Forward(dropoutInput);
    
    std::cout << "Dropout effect (different outputs in train/infer): ";
    std::cout << (trainOutput.Data()[0] != inferOutput.Data()[0] ? "Yes" : "No") << std::endl;
    
    // ========================================
    // 5. Summary of differences
    // ========================================
    std::cout << "\n\n=== SUMMARY OF DIFFERENCES ===" << std::endl;
    std::cout << "\nFFN is suitable for:" << std::endl;
    std::cout << "  ✅ Simple multi-layer perceptrons" << std::endl;
    std::cout << "  ✅ Rapid prototyping" << std::endl;
    std::cout << "  ✅ Uniform activations everywhere" << std::endl;
    std::cout << "  ✅ Minimal code" << std::endl;
    
    std::cout << "\nSequential is suitable for:" << std::endl;
    std::cout << "  ✅ Complex architectures" << std::endl;
    std::cout << "  ✅ Different activations" << std::endl;
    std::cout << "  ✅ Regularization (Dropout)" << std::endl;
    std::cout << "  ✅ Input/output validation" << std::endl;
    std::cout << "  ✅ Maximum flexibility" << std::endl;
    
    std::cout << "\n=== Comparison completed ===" << std::endl;
    
    return 0;
} 