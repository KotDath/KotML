#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;
using namespace kotml::nn;

int main() {
    std::cout << "=== Sequential Simple Examples ===" << std::endl;
    
    // Example 1: Simple FFN network
    std::cout << "\nExample 1: Simple FFN network" << std::endl;
    
    // Creating network using Sequential builder
    auto network = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Linear(20, 5)
        .Build();
    
    // Testing
    Tensor input = Tensor::Randn({10});
    Tensor output = network.Forward(input);
    
    std::cout << "Input: size " << input.Size() << std::endl;
    std::cout << "Output: size " << output.Size() << std::endl;
    network.Summary();
    
    // Example 2: Complex network with different activations
    std::cout << "\n\nExample 2: Complex network with different activations" << std::endl;
    
    auto complex_net = Sequential()
        .Linear(784, 256)
        .ReLU()
        .Dropout(0.3f)
        .Linear(256, 128)
        .Tanh()
        .Linear(128, 10)
        .Sigmoid()
        .Build();
    
    complex_net.Summary();
    
    // Example 3: Regression network
    std::cout << "\n\nExample 3: Regression network" << std::endl;
    
    auto regressor = Sequential()
        .Linear(5, 32)
        .ReLU()
        .Linear(32, 1)
        .Build();
    
    regressor.Summary();
    
    std::cout << "\n=== All examples work! ===" << std::endl;
    
    return 0;
} 