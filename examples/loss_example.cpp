#include "kotml/kotml.hpp"
#include "kotml/tensor.hpp"
#include "kotml/nn/loss.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace kotml;

// Helper function to print tensor values
void PrintTensor(const Tensor& tensor, const std::string& name) {
    std::cout << name << ": ";
    if (tensor.Size() <= 10) {
        std::cout << "[";
        for (size_t i = 0; i < tensor.Size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << tensor[i];
            if (i < tensor.Size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    } else {
        std::cout << "shape=" << tensor.Shape()[0];
        if (tensor.Ndim() > 1) {
            for (size_t i = 1; i < tensor.Ndim(); ++i) {
                std::cout << "x" << tensor.Shape()[i];
            }
        }
        std::cout << " (first 5: ";
        for (size_t i = 0; i < std::min(5ul, tensor.Size()); ++i) {
            std::cout << std::fixed << std::setprecision(4) << tensor[i];
            if (i < std::min(5ul, tensor.Size()) - 1) std::cout << ", ";
        }
        std::cout << "...)";
    }
    std::cout << std::endl;
}

// Generate sample data for regression
std::pair<Tensor, Tensor> GenerateRegressionData(size_t n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    
    Tensor predictions = Tensor::Zeros({n_samples});
    Tensor targets = Tensor::Zeros({n_samples});
    
    for (size_t i = 0; i < n_samples; ++i) {
        float true_val = uniform(gen);
        targets[i] = true_val;
        predictions[i] = true_val + noise(gen); // Add some noise
    }
    
    return {predictions, targets};
}

// Generate sample data for binary classification
std::pair<Tensor, Tensor> GenerateBinaryClassificationData(size_t n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> prob_dist(0.1f, 0.9f);
    std::uniform_int_distribution<int> label_dist(0, 1);
    
    Tensor predictions = Tensor::Zeros({n_samples});
    Tensor targets = Tensor::Zeros({n_samples});
    
    for (size_t i = 0; i < n_samples; ++i) {
        targets[i] = static_cast<float>(label_dist(gen));
        // Generate predictions that are somewhat correlated with targets
        if (targets[i] == 1.0f) {
            predictions[i] = prob_dist(gen) * 0.5f + 0.5f; // Higher probability for class 1
        } else {
            predictions[i] = prob_dist(gen) * 0.5f; // Lower probability for class 1
        }
    }
    
    return {predictions, targets};
}

// Generate sample data for multi-class classification
std::pair<Tensor, Tensor> GenerateMultiClassData(size_t n_samples, size_t n_classes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> prob_dist(0.1f, 0.9f);
    std::uniform_int_distribution<size_t> class_dist(0, n_classes - 1);
    
    Tensor predictions = Tensor::Zeros({n_samples, n_classes});
    Tensor targets = Tensor::Zeros({n_samples, n_classes});
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Create one-hot target
        size_t true_class = class_dist(gen);
        targets.At({i, true_class}) = 1.0f;
        
        // Generate softmax-like predictions
        float sum = 0.0f;
        for (size_t j = 0; j < n_classes; ++j) {
            float val = prob_dist(gen);
            if (j == true_class) val *= 2.0f; // Boost true class probability
            predictions.At({i, j}) = val;
            sum += val;
        }
        
        // Normalize to sum to 1 (softmax-like)
        for (size_t j = 0; j < n_classes; ++j) {
            predictions.At({i, j}) /= sum;
        }
    }
    
    return {predictions, targets};
}

int main() {
    std::cout << "=== Loss Functions Example ===" << std::endl;
    
    // ===== MSE Loss =====
    std::cout << std::endl << "=== Mean Squared Error (MSE) Loss ===" << std::endl;
    {
        auto [predictions, targets] = GenerateRegressionData(5);
        
        PrintTensor(predictions, "Predictions");
        PrintTensor(targets, "Targets");
        
        nn::MSELoss mse_loss;
        auto loss = mse_loss.Forward(predictions, targets);
        auto gradients = mse_loss.Backward(predictions, targets);
        
        PrintTensor(loss, "MSE Loss");
        PrintTensor(gradients, "Gradients");
        
        // Test convenience function
        auto loss_conv = nn::loss::MSE(predictions, targets);
        std::cout << "Convenience function result: " << loss_conv[0] << std::endl;
        
        std::cout << "Loss function: " << mse_loss.GetName() << std::endl;
    }
    
    // ===== MAE Loss =====
    std::cout << std::endl << "=== Mean Absolute Error (MAE) Loss ===" << std::endl;
    {
        auto [predictions, targets] = GenerateRegressionData(5);
        
        PrintTensor(predictions, "Predictions");
        PrintTensor(targets, "Targets");
        
        nn::MAELoss mae_loss;
        auto loss = mae_loss.Forward(predictions, targets);
        auto gradients = mae_loss.Backward(predictions, targets);
        
        PrintTensor(loss, "MAE Loss");
        PrintTensor(gradients, "Gradients");
        
        std::cout << "Loss function: " << mae_loss.GetName() << std::endl;
    }
    
    // ===== Binary Cross Entropy Loss =====
    std::cout << std::endl << "=== Binary Cross Entropy (BCE) Loss ===" << std::endl;
    {
        auto [predictions, targets] = GenerateBinaryClassificationData(5);
        
        PrintTensor(predictions, "Predictions (probabilities)");
        PrintTensor(targets, "Targets (binary labels)");
        
        nn::BCELoss bce_loss;
        auto loss = bce_loss.Forward(predictions, targets);
        auto gradients = bce_loss.Backward(predictions, targets);
        
        PrintTensor(loss, "BCE Loss");
        PrintTensor(gradients, "Gradients");
        
        std::cout << "Loss function: " << bce_loss.GetName() << std::endl;
        std::cout << "Epsilon: " << bce_loss.GetEpsilon() << std::endl;
    }
    
    // ===== Cross Entropy Loss =====
    std::cout << std::endl << "=== Categorical Cross Entropy Loss ===" << std::endl;
    {
        auto [predictions, targets] = GenerateMultiClassData(3, 4);
        
        std::cout << "Predictions (softmax probabilities):" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "  Sample " << i << ": [";
            for (size_t j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(3) << predictions.At({i, j});
                if (j < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Targets (one-hot encoded):" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "  Sample " << i << ": [";
            for (size_t j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(0) << targets.At({i, j});
                if (j < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        nn::CrossEntropyLoss ce_loss;
        auto loss = ce_loss.Forward(predictions, targets);
        auto gradients = ce_loss.Backward(predictions, targets);
        
        PrintTensor(loss, "Cross Entropy Loss");
        std::cout << "Gradients shape: " << gradients.Shape()[0] << "x" << gradients.Shape()[1] << std::endl;
        
        std::cout << "Loss function: " << ce_loss.GetName() << std::endl;
    }
    
    // ===== Huber Loss =====
    std::cout << std::endl << "=== Huber Loss (Smooth L1) ===" << std::endl;
    {
        // Create data with some outliers
        std::vector<float> pred_data = {-2.0f, -0.5f, 0.0f, 0.5f, 3.0f};
        std::vector<float> target_data = {-1.8f, -0.3f, 0.1f, 0.7f, 1.0f};
        Tensor predictions(pred_data, {5});
        Tensor targets(target_data, {5});
        
        PrintTensor(predictions, "Predictions");
        PrintTensor(targets, "Targets");
        
        nn::HuberLoss huber_loss(1.0f); // delta = 1.0
        auto loss = huber_loss.Forward(predictions, targets);
        auto gradients = huber_loss.Backward(predictions, targets);
        
        PrintTensor(loss, "Huber Loss");
        PrintTensor(gradients, "Gradients");
        
        std::cout << "Loss function: " << huber_loss.GetName() << std::endl;
        std::cout << "Delta: " << huber_loss.GetDelta() << std::endl;
        
        // Compare with different delta values
        std::cout << std::endl << "Comparison with different delta values:" << std::endl;
        for (float delta : {0.5f, 1.0f, 2.0f}) {
            nn::HuberLoss huber(delta);
            auto loss_val = huber.Forward(predictions, targets);
            std::cout << "  Delta=" << delta << ": loss=" << loss_val[0] << std::endl;
        }
    }
    
    // ===== Loss Comparison =====
    std::cout << std::endl << "=== Loss Function Comparison ===" << std::endl;
    {
        // Create data with outliers to show robustness differences
        std::vector<float> pred_data = {1.0f, 2.0f, 10.0f, 4.0f, 5.0f}; // One outlier
        std::vector<float> target_data = {1.1f, 2.1f, 3.0f, 4.1f, 5.1f};
        Tensor predictions(pred_data, {5});
        Tensor targets(target_data, {5});
        
        PrintTensor(predictions, "Predictions (with outlier)");
        PrintTensor(targets, "Targets");
        
        nn::MSELoss mse;
        nn::MAELoss mae;
        nn::HuberLoss huber(1.0f);
        
        auto mse_loss = mse.Forward(predictions, targets);
        auto mae_loss = mae.Forward(predictions, targets);
        auto huber_loss = huber.Forward(predictions, targets);
        
        std::cout << std::endl << "Loss comparison (with outlier at index 2):" << std::endl;
        std::cout << "  MSE Loss:   " << std::fixed << std::setprecision(4) << mse_loss[0] << std::endl;
        std::cout << "  MAE Loss:   " << std::fixed << std::setprecision(4) << mae_loss[0] << std::endl;
        std::cout << "  Huber Loss: " << std::fixed << std::setprecision(4) << huber_loss[0] << std::endl;
        std::cout << std::endl << "Note: MAE and Huber are more robust to outliers than MSE" << std::endl;
    }
    
    // ===== Parameter Validation =====
    std::cout << std::endl << "=== Parameter Validation ===" << std::endl;
    {
        std::vector<float> pred_data = {0.5f, 0.8f};
        std::vector<float> wrong_data = {0.0f, 1.0f, 0.5f}; // Wrong shape
        std::vector<float> binary_data = {0.0f, 1.0f};
        std::vector<float> invalid_data = {0.0f, 0.5f}; // Invalid for BCE
        
        Tensor predictions(pred_data, {2});
        Tensor wrong_targets(wrong_data, {3});
        Tensor binary_targets(binary_data, {2});
        Tensor invalid_targets(invalid_data, {2});
        
        // Test shape mismatch
        try {
            nn::MSELoss mse;
            mse.Forward(predictions, wrong_targets);
            std::cout << "ERROR: Should have thrown exception for shape mismatch!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught shape mismatch: " << e.what() << std::endl;
        }
        
        // Test BCE with invalid targets
        try {
            nn::BCELoss bce;
            bce.Forward(predictions, invalid_targets);
            std::cout << "ERROR: Should have thrown exception for non-binary targets!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught invalid BCE targets: " << e.what() << std::endl;
        }
        
        // Test Huber with invalid delta
        try {
            nn::HuberLoss huber(-1.0f);
            std::cout << "ERROR: Should have thrown exception for negative delta!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Correctly caught invalid Huber delta: " << e.what() << std::endl;
        }
    }
    
    // ===== ForwardBackward Method =====
    std::cout << std::endl << "=== ForwardBackward Method ===" << std::endl;
    {
        auto [predictions, targets] = GenerateRegressionData(3);
        
        nn::MSELoss mse;
        auto [loss, gradients] = mse.ForwardBackward(predictions, targets);
        
        PrintTensor(predictions, "Predictions");
        PrintTensor(targets, "Targets");
        PrintTensor(loss, "Loss");
        PrintTensor(gradients, "Gradients");
        
        std::cout << "ForwardBackward method computes both loss and gradients efficiently" << std::endl;
    }
    
    // ===== Batch Processing =====
    std::cout << std::endl << "=== Batch Processing ===" << std::endl;
    {
        size_t batch_size = 4;
        size_t n_classes = 3;
        auto [predictions, targets] = GenerateMultiClassData(batch_size, n_classes);
        
        std::cout << "Batch size: " << batch_size << ", Classes: " << n_classes << std::endl;
        
        nn::CrossEntropyLoss ce;
        auto loss = ce.Forward(predictions, targets);
        auto gradients = ce.Backward(predictions, targets);
        
        PrintTensor(loss, "Batch Loss");
        std::cout << "Gradients shape: " << gradients.Shape()[0] << "x" << gradients.Shape()[1] << std::endl;
        
        std::cout << "Cross entropy loss handles batch processing automatically" << std::endl;
    }
    
    std::cout << std::endl << "=== Loss Functions Example Complete ===" << std::endl;
    
    return 0;
} 