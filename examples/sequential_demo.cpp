#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>
#include <vector>

using namespace kotml;
using namespace kotml::nn;

int main() {
    std::cout << "=== Sequential Neural Network Builder Demo ===" << std::endl;
    
    // 1. Creating simple classification network
    std::cout << "\n1. Creating simple classification network:" << std::endl;
    
    auto classifier = Sequential()
        .Linear(784, 128)  // Input layer: 784 -> 128
        .ReLU()            // ReLU activation
        .Linear(128, 64)   // Hidden layer: 128 -> 64
        .ReLU()            // ReLU activation
        .Linear(64, 10)    // Output layer: 64 -> 10
        .Build();
    
    std::cout << "Classification network created:" << std::endl;
    classifier.Summary();
    
    // Testing forward pass
    std::cout << "\nTesting forward pass:" << std::endl;
    Tensor input = Tensor::Randn({784});
    Tensor output = classifier.Forward(input);
    
    std::cout << "Input size: " << input.Size() << std::endl;
    std::cout << "Output size: " << output.Size() << std::endl;
    std::cout << "First 5 output values: ";
    for (size_t i = 0; i < 5 && i < output.Size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    // 2. Creating regression network
    std::cout << "\n\n2. Creating regression network:" << std::endl;
    
    auto regressor = Sequential()
        .Linear(20, 64)    // Input: 20 features
        .Tanh()            // Tanh activation
        .Linear(64, 32)    // Hidden layer
        .ReLU()            // ReLU activation
        .Linear(32, 16)    // Another hidden layer
        .ReLU()            // ReLU activation
        .Linear(16, 1)     // Output: single value
        .Build();
    
    std::cout << "Regression network:" << std::endl;
    regressor.Summary();
    
    Tensor regInput = Tensor::Randn({20});
    Tensor regOutput = regressor.Forward(regInput);
    std::cout << "Regressor output: " << regOutput.Data()[0] << std::endl;
    
    // 3. Creating deep network
    std::cout << "\n\n3. Creating deep network:" << std::endl;
    
    auto deepNet = Sequential()
        .Linear(100, 256)
        .ReLU()
        .Linear(256, 128)
        .Sigmoid()         // Different activation
        .Linear(128, 64)
        .Tanh()           // Another different activation
        .Linear(64, 32)
        .ReLU()
        .Linear(32, 10)
        .Build();
    
    std::cout << "Deep network with mixed activations:" << std::endl;
    deepNet.Summary();
    
    // 4. Network with output layer
    std::cout << "\n\n4. Network with output layer:" << std::endl;
    
    auto withOutput = Sequential()
        .Input(28*28)      // Input validation
        .Linear(784, 128)
        .ReLU()
        .Output(128, 10, ActivationType::Sigmoid)  // Composite output
        .Build();
    
    withOutput.Summary();
    
    // 5. Training mode demonstration
    std::cout << "\n\n5. Training mode demonstration:" << std::endl;
    
    auto dropoutNet = Sequential()
        .Linear(100, 64)
        .ReLU()
        .Dropout(0.5f)     // 50% dropout
        .Linear(64, 32)
        .ReLU()
        .Dropout(0.3f)     // 30% dropout
        .Linear(32, 10)
        .Build();
    
    Tensor dropoutInput = Tensor::Ones({100});
    
    std::cout << "Training mode (with Dropout):" << std::endl;
    dropoutNet.SetTraining(true);
    Tensor trainOut = dropoutNet.Forward(dropoutInput);
    std::cout << "Output: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << trainOut[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Inference mode (without Dropout):" << std::endl;
    dropoutNet.SetTraining(false);
    Tensor inferOut = dropoutNet.Forward(dropoutInput);
    std::cout << "Output: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << inferOut[i] << " ";
    }
    std::cout << std::endl;
    
    // 6. Access to individual layers
    std::cout << "\n\n6. Access to individual layers:" << std::endl;
    
    auto inspectableNet = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Linear(20, 5)
        .Build();
    
    std::cout << "Number of layers: " << inspectableNet.GetNumLayers() << std::endl;
    
    for (size_t i = 0; i < inspectableNet.GetNumLayers(); ++i) {
        auto* layer = inspectableNet.GetLayer(i);
        std::cout << "Layer " << i << ": " << layer->GetName() << std::endl;
    }
    
    // 7. Working with parameters
    std::cout << "\n\n7. Working with parameters:" << std::endl;
    
    auto paramNet = Sequential()
        .Linear(5, 10)
        .ReLU()
        .Linear(10, 3)
        .Build();
    
    auto params = paramNet.Parameters();
    std::cout << "Number of parameter tensors: " << params.size() << std::endl;
    std::cout << "Total parameters: " << paramNet.CountParameters() << std::endl;
    
    // Zero gradients
    paramNet.ZeroGrad();
    std::cout << "Gradients zeroed" << std::endl;
    
    // 8. Comparison of Sequential with FFN
    std::cout << "\n\n8. Comparison of Sequential with FFN:" << std::endl;
    
    // Sequential version
    auto seqNet = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Linear(20, 5)
        .Build();
    
    // FFN version
    FFN ffnNet({10, 20, 5}, ActivationType::Relu);
    
    std::cout << "Sequential parameters: " << seqNet.CountParameters() << std::endl;
    std::cout << "FFN parameters: " << ffnNet.CountParameters() << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    return 0;
} 