/**
 * AND Gate - Training Analysis with BCE Loss
 */

#include "kotml/kotml.hpp"
#include <iostream>
#include <iomanip>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

int main() {
    std::cout << "AND Gate - BCE Loss Analysis" << std::endl;
    
    // Training data: [0,0]->0, [0,1]->0, [1,0]->0, [1,1]->1
    std::vector<Tensor> inputs = {
        Tensor({0.0f, 0.0f}, {1, 2}),
        Tensor({0.0f, 1.0f}, {1, 2}),
        Tensor({1.0f, 0.0f}, {1, 2}),
        Tensor({1.0f, 1.0f}, {1, 2})
    };
    
    std::vector<Tensor> targets = {
        Tensor({0.0f}, {1, 1}),
        Tensor({0.0f}, {1, 1}),
        Tensor({0.0f}, {1, 1}),
        Tensor({1.0f}, {1, 1})
    };
    
    // Model: y = sigmoid(w1*x1 + w2*x2 + b)
    auto model = Sequential()
        .Linear(2, 1, true)  // 2 inputs -> 1 output, with bias
        .Sigmoid()
        .Build();
    
    // Check initial weights
    auto initial_params = model.Parameters();
    std::cout << "\nInitial: w1=" << std::fixed << std::setprecision(3) 
              << (*initial_params[0])[0] << ", w2=" << (*initial_params[0])[1] 
              << ", b=" << (*initial_params[1])[0] << std::endl;
    
    // Test with BCE Loss (correct for classification)
    std::cout << "\n=== BCE Loss (Aggressive Training) ===" << std::endl;
    
    auto model_bce = Sequential()
        .Linear(2, 1, true)
        .Sigmoid()
        .Build();
    
    model_bce.Compile(std::make_unique<SGD>(5.0f), std::make_unique<BCELoss>());  // Higher LR
    
    // Manual training with loss tracking
    for (int epoch = 0; epoch < 200; ++epoch) {  // More epochs
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto pred = model_bce.Predict({inputs[i]});
            // BCE loss calculation
            float p = std::max(1e-7f, std::min(1.0f - 1e-7f, pred[0][0]));
            float t = targets[i][0];
            float loss = -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
            total_loss += loss;
        }
        
        if (epoch % 50 == 0) {  // Less frequent output
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " BCE Loss: " << std::fixed << std::setprecision(4) << total_loss << std::endl;
        }
        
        // One training step
        model_bce.Train(inputs, targets, 1, 0, nullptr, nullptr, false);
    }
    
    // Final test with BCE
    int correct = 0;
    std::cout << "\nBCE Results:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred = model_bce.Predict({inputs[i]});
        float x1 = inputs[i][0], x2 = inputs[i][1];
        float expected = targets[i][0], predicted = pred[0][0];
        int rounded = (predicted > 0.5f) ? 1 : 0;
        if (rounded == (int)expected) correct++;
        
        std::cout << "[" << x1 << "," << x2 << "] -> " << expected 
                  << " (pred: " << std::fixed << std::setprecision(3) << predicted 
                  << ", rounded: " << rounded 
                  << (rounded == (int)expected ? " ✓)" : " ✗)") << std::endl;
    }
    std::cout << "BCE Accuracy: " << correct << "/4 (" << (correct*25) << "%)" << std::endl;
    
    auto bce_params = model_bce.Parameters();
    std::cout << "BCE Final: w1=" << std::fixed << std::setprecision(3) 
              << (*bce_params[0])[0] << ", w2=" << (*bce_params[0])[1] 
              << ", b=" << (*bce_params[1])[0] << std::endl;
    
    // Compare with MSE Loss (wrong for classification)
    std::cout << "\n=== MSE Loss (Wrong for Classification) ===" << std::endl;
    
    auto model_mse = Sequential()
        .Linear(2, 1, true)
        .Sigmoid()
        .Build();
    
    model_mse.Compile(std::make_unique<SGD>(1.0f), std::make_unique<MSELoss>());
    model_mse.Train(inputs, targets, 50, 0, nullptr, nullptr, false);
    
    int mse_correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred = model_mse.Predict({inputs[i]});
        int rounded = (pred[0][0] > 0.5f) ? 1 : 0;
        if (rounded == (int)targets[i][0]) mse_correct++;
    }
    std::cout << "MSE Accuracy: " << mse_correct << "/4 (" << (mse_correct*25) << "%)" << std::endl;
    
    auto mse_params = model_mse.Parameters();
    std::cout << "MSE Final: w1=" << std::fixed << std::setprecision(3) 
              << (*mse_params[0])[0] << ", w2=" << (*mse_params[0])[1] 
              << ", b=" << (*mse_params[1])[0] << std::endl;
    
    std::cout << "\n=== Manual Ideal Weights Test ===" << std::endl;
    
    // Create model with manual weight setting
    auto model_ideal = Sequential()
        .Linear(2, 1, true)
        .Sigmoid()
        .Build();
    
    // Set ideal weights manually: w1=5, w2=5, b=-7
    auto ideal_params = model_ideal.Parameters();
    (*ideal_params[0])[0] = 5.0f;  // w1
    (*ideal_params[0])[1] = 5.0f;  // w2
    (*ideal_params[1])[0] = -7.0f; // b (negative!)
    
    std::cout << "Manual weights: w1=5.0, w2=5.0, b=-7.0" << std::endl;
    
    // Test ideal solution
    int ideal_correct = 0;
    std::cout << "Ideal Results:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto pred = model_ideal.Predict({inputs[i]});
        float x1 = inputs[i][0], x2 = inputs[i][1];
        float expected = targets[i][0], predicted = pred[0][0];
        int rounded = (predicted > 0.5f) ? 1 : 0;
        if (rounded == (int)expected) ideal_correct++;
        
        // Show the linear combination before sigmoid
        float linear = 5.0f * x1 + 5.0f * x2 - 7.0f;
        
        std::cout << "[" << x1 << "," << x2 << "] -> " << expected 
                  << " (linear: " << std::fixed << std::setprecision(1) << linear
                  << ", sigmoid: " << std::setprecision(3) << predicted 
                  << ", rounded: " << rounded 
                  << (rounded == (int)expected ? " ✓)" : " ✗)") << std::endl;
    }
    std::cout << "Ideal Accuracy: " << ideal_correct << "/4 (" << (ideal_correct*25) << "%)" << std::endl;
    
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "BCE Loss is designed for classification - should work better!" << std::endl;
    std::cout << "MSE Loss is for regression - suboptimal for binary classification" << std::endl;
    std::cout << "Ideal AND: w1≈w2>0, b<0 (e.g., w1=5, w2=5, b=-7)" << std::endl;
    
    return 0;
} 