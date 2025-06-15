#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;
using namespace kotml::nn;

int main() {
    std::cout << "=== FFN (Feed-Forward Network) Demo ===" << std::endl;
    
    // 1. Creating simple FFN
    std::cout << "\n1. Creating simple FFN:" << std::endl;
    
    // Architecture: 4 -> 8 -> 6 -> 2
    FFN simple_ffn({4, 8, 6, 2}, ActivationType::Relu);
    simple_ffn.PrintArchitecture();
    
    // 2. Testing forward pass
    std::cout << "\n2. Testing forward pass:" << std::endl;
    
    // Creating input data
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, {4}, false);
    std::cout << "Input: " << input << std::endl;
    
    // Forward pass
    Tensor output = simple_ffn.Forward(input);
    std::cout << "Output: " << output << std::endl;
    
    // 3. Testing with batch data
    std::cout << "\n3. Testing with batch data:" << std::endl;
    
    Tensor batch_input = Tensor::Randn({3, 4}, true); // 3 samples, 4 features
    std::cout << "Batch input shape: [" << batch_input.Shape()[0] << ", " << batch_input.Shape()[1] << "]" << std::endl;
    
    Tensor batch_output = simple_ffn.Forward(batch_input);
    std::cout << "Batch output shape: [" << batch_output.Shape()[0] << ", " << batch_output.Shape()[1] << "]" << std::endl;
    
    // 4. Testing individual layers
    std::cout << "\n4. Testing individual layers:" << std::endl;
    
    // Creating separate linear layer
    Linear linear_layer(4, 8);
    std::cout << "Linear layer (4->8) parameters: " << simple_ffn.CountParameters() << std::endl;
    
    Tensor linear_output = linear_layer.Forward(input);
    std::cout << "Linear layer output: " << linear_output << std::endl;
    
    // Testing activation
    Activation relu_layer(ActivationType::Relu);
    Tensor activated_output = relu_layer.Forward(linear_output);
    std::cout << "After ReLU activation: " << activated_output << std::endl;
    
    // 5. Testing activation functions
    std::cout << "\n5. Testing activation functions:" << std::endl;
    
    FFN relu_net({4, 6, 2}, ActivationType::Relu);
    FFN sigmoid_net({4, 6, 2}, ActivationType::Sigmoid);
    FFN tanh_net({4, 6, 2}, ActivationType::Tanh);
    
    Tensor test_input({-1.0f, 0.0f, 1.0f, 2.0f}, {4}, false);
    
    std::cout << "ReLU output: " << relu_net.Forward(test_input) << std::endl;
    std::cout << "Sigmoid output: " << sigmoid_net.Forward(test_input) << std::endl;
    std::cout << "Tanh output: " << tanh_net.Forward(test_input) << std::endl;
    
    // 6. Testing Dropout
    std::cout << "\n6. Testing Dropout:" << std::endl;
    
    FFN dropout_net({4, 8, 2}, ActivationType::Relu, ActivationType::None, 0.5f);
    
    std::cout << "Network with Dropout (rate=0.5):" << std::endl;
    dropout_net.PrintArchitecture();
    
    // In training mode
    dropout_net.SetTraining(true);
    Tensor train_output = dropout_net.Forward(test_input);
    std::cout << "Training mode output: " << train_output << std::endl;
    
    // In inference mode
    dropout_net.SetTraining(false);
    Tensor eval_output = dropout_net.Forward(test_input);
    std::cout << "Inference mode output: " << eval_output << std::endl;
    
    // 7. Network parameter information
    std::cout << "\n7. Network parameter information:" << std::endl;
    
    auto parameters = simple_ffn.Parameters();
    std::cout << "Number of parameter tensors: " << parameters.size() << std::endl;
    
    size_t total_params = 0;
    for (size_t i = 0; i < parameters.size(); ++i) {
        std::cout << "Parameter " << i << ": shape=[";
        for (size_t j = 0; j < parameters[i]->Shape().size(); ++j) {
            std::cout << parameters[i]->Shape()[j];
            if (j < parameters[i]->Shape().size() - 1) std::cout << ", ";
        }
        std::cout << "], size=" << parameters[i]->Size() << std::endl;
        total_params += parameters[i]->Size();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    
    // 8. Different FFN architectures
    std::cout << "\n8. Different FFN architectures:" << std::endl;
    
    // Simple binary classifier
    FFN binary_classifier({10, 16, 1}, ActivationType::Relu, ActivationType::Sigmoid);
    std::cout << "\nBinary classifier:" << std::endl;
    binary_classifier.PrintArchitecture();
    
    // Multi-class classifier
    FFN multiclass_classifier({784, 128, 64, 10}, ActivationType::Relu);
    std::cout << "\nMulti-class classifier:" << std::endl;
    multiclass_classifier.PrintArchitecture();
    
    // Regressor with Dropout
    FFN regressor({20, 64, 32, 1}, ActivationType::Relu, ActivationType::None, 0.3f);
    std::cout << "\nRegressor with Dropout:" << std::endl;
    regressor.PrintArchitecture();
    
    std::cout << "\n=== FFN Demo completed ===" << std::endl;
    
    return 0;
} 