#include "kotml/nn/layers.hpp"
#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;
using namespace kotml::nn;

int main() {
    std::cout << "=== Neural Network Layers Demo ===" << std::endl;
    
    // 1. Input Layer
    std::cout << "\n1. Input Layer:" << std::endl;
    
    InputLayer input_layer(784);  // For 28x28 images
    std::cout << "Input layer created for size: " << input_layer.GetInputSize() << std::endl;
    
    // Testing with correct input
    Tensor correct_input = Tensor::Randn({784});
    std::cout << "Input tensor size: " << correct_input.Size() << std::endl;
    
    Tensor input_output = input_layer.Forward(correct_input);
    std::cout << "Output after input layer: " << input_output.Size() << std::endl;
    
    // Testing with batch
    Tensor batch_input = Tensor::Randn({32, 784});  // 32 samples
    Tensor batch_output = input_layer.Forward(batch_input);
    std::cout << "Batch processing: [" << batch_input.Shape()[0] << ", " << batch_input.Shape()[1] 
              << "] -> [" << batch_output.Shape()[0] << ", " << batch_output.Shape()[1] << "]" << std::endl;
    
    // 2. Linear Layer
    std::cout << "\n2. Linear Layer:" << std::endl;
    
    Linear linear_layer(784, 128);
    std::cout << "Linear layer: 784 -> 128" << std::endl;
    std::cout << "Parameters: " << linear_layer.CountParameters() << std::endl;
    
    Tensor linear_output = linear_layer.Forward(correct_input);
    std::cout << "Linear layer output size: " << linear_output.Size() << std::endl;
    
    // 3. Activation Layers
    std::cout << "\n3. Activation Layers:" << std::endl;
    
    // ReLU
    Activation relu_layer(ActivationType::Relu);
    Tensor relu_output = relu_layer.Forward(linear_output);
    std::cout << "ReLU activation applied" << std::endl;
    
    // Sigmoid
    Activation sigmoid_layer(ActivationType::Sigmoid);
    Tensor sigmoid_output = sigmoid_layer.Forward(linear_output);
    std::cout << "Sigmoid activation applied" << std::endl;
    
    // Tanh
    Activation tanh_layer(ActivationType::Tanh);
    Tensor tanh_output = tanh_layer.Forward(linear_output);
    std::cout << "Tanh activation applied" << std::endl;
    
    // 4. Dropout Layer
    std::cout << "\n4. Dropout Layer:" << std::endl;
    
    Dropout dropout_layer(0.5f);
    std::cout << "Dropout layer with rate: " << dropout_layer.GetDropoutRate() << std::endl;
    
    // Training mode
    dropout_layer.SetTraining(true);
    Tensor dropout_train = dropout_layer.Forward(relu_output);
    std::cout << "Training mode - some neurons dropped" << std::endl;
    
    // Inference mode
    dropout_layer.SetTraining(false);
    Tensor dropout_eval = dropout_layer.Forward(relu_output);
    std::cout << "Inference mode - all neurons active" << std::endl;
    
    // 5. Output Layer
    std::cout << "\n5. Output Layer:" << std::endl;
    
    OutputLayer output_layer(128, 10, ActivationType::Sigmoid);
    std::cout << "Output layer: 128 -> 10 with Sigmoid" << std::endl;
    std::cout << "Parameters: " << output_layer.CountParameters() << std::endl;
    
    Tensor final_output = output_layer.Forward(relu_output);
    std::cout << "Final output size: " << final_output.Size() << std::endl;
    
    // Access to components
    Linear& output_linear = output_layer.GetLinear();
    Activation& output_activation = output_layer.GetActivation();
    std::cout << "Linear component parameters: " << output_linear.CountParameters() << std::endl;
    std::cout << "Activation type: " << (int)output_activation.GetActivationType() << std::endl;
    
    // 6. Complete Network Example
    std::cout << "\n6. Complete Network Example:" << std::endl;
    
    // Creating layers
    InputLayer input(784);
    Linear hidden1(784, 256);
    Activation relu1(ActivationType::Relu);
    Dropout dropout1(0.3f);
    Linear hidden2(256, 128);
    Activation relu2(ActivationType::Relu);
    OutputLayer output(128, 10, ActivationType::Sigmoid);
    
    // Forward pass
    Tensor x = Tensor::Randn({784});
    std::cout << "Input: " << x.Size() << std::endl;
    
    x = input.Forward(x);
    std::cout << "After input layer: " << x.Size() << std::endl;
    
    x = hidden1.Forward(x);
    std::cout << "After first linear: " << x.Size() << std::endl;
    
    x = relu1.Forward(x);
    std::cout << "After first ReLU: " << x.Size() << std::endl;
    
    dropout1.SetTraining(false);  // Inference mode
    x = dropout1.Forward(x);
    std::cout << "After dropout: " << x.Size() << std::endl;
    
    x = hidden2.Forward(x);
    std::cout << "After second linear: " << x.Size() << std::endl;
    
    x = relu2.Forward(x);
    std::cout << "After second ReLU: " << x.Size() << std::endl;
    
    x = output.Forward(x);
    std::cout << "Final output: " << x.Size() << std::endl;
    
    // 7. Parameter counting
    std::cout << "\n7. Parameter counting:" << std::endl;
    
    size_t total_params = 0;
    
    // Count parameters for each layer that has them
    auto input_params = input.Parameters();
    for (auto* param : input_params) {
        total_params += param->Size();
    }
    
    total_params += hidden1.CountParameters();
    
    auto relu1_params = relu1.Parameters();
    for (auto* param : relu1_params) {
        total_params += param->Size();
    }
    
    auto dropout1_params = dropout1.Parameters();
    for (auto* param : dropout1_params) {
        total_params += param->Size();
    }
    
    total_params += hidden2.CountParameters();
    
    auto relu2_params = relu2.Parameters();
    for (auto* param : relu2_params) {
        total_params += param->Size();
    }
    
    total_params += output.CountParameters();
    
    std::cout << "Total network parameters: " << total_params << std::endl;
    
    // 8. Comparison with FFN
    std::cout << "\n8. Comparison with FFN:" << std::endl;
    
    FFN equivalent_ffn({784, 256, 128, 10}, ActivationType::Relu, ActivationType::Sigmoid, 0.3f);
    std::cout << "Equivalent FFN parameters: " << equivalent_ffn.CountParameters() << std::endl;
    
    Tensor ffn_output = equivalent_ffn.Forward(Tensor::Randn({784}));
    std::cout << "FFN output size: " << ffn_output.Size() << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    return 0;
} 