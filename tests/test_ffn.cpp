/**
 * Comprehensive tests for FFN (Feed-Forward Network) class
 * Tests all major functionality including Predict, Compile, Train
 */

#include <gtest/gtest.h>
#include "kotml/kotml.hpp"
#include <vector>
#include <memory>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

class FFNTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard test data
        testInput = Tensor({1.0f, 2.0f, 3.0f}, {1, 3});
        batchInput = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
        
        // Simple classification data (XOR-like problem)
        trainInputs = {
            Tensor({0.0f, 0.0f}, {1, 2}),
            Tensor({0.0f, 1.0f}, {1, 2}),
            Tensor({1.0f, 0.0f}, {1, 2}),
            Tensor({1.0f, 1.0f}, {1, 2})
        };
        
        trainTargets = {
            Tensor({0.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1}),
            Tensor({0.0f}, {1, 1})
        };
        
        // Regression data
        regressionInputs = {
            Tensor({1.0f}, {1, 1}),
            Tensor({2.0f}, {1, 1}),
            Tensor({3.0f}, {1, 1}),
            Tensor({4.0f}, {1, 1})
        };
        
        regressionTargets = {
            Tensor({2.0f}, {1, 1}),  // y = 2x
            Tensor({4.0f}, {1, 1}),
            Tensor({6.0f}, {1, 1}),
            Tensor({8.0f}, {1, 1})
        };
    }
    
    Tensor testInput;
    Tensor batchInput;
    std::vector<Tensor> trainInputs;
    std::vector<Tensor> trainTargets;
    std::vector<Tensor> regressionInputs;
    std::vector<Tensor> regressionTargets;
};

// Basic construction tests
TEST_F(FFNTest, Construction) {
    // Basic construction
    FFN simple({3, 5, 2});
    EXPECT_EQ(simple.GetInputSize(), 3);
    EXPECT_EQ(simple.GetOutputSize(), 2);
    EXPECT_EQ(simple.GetNumLayers(), 2);
    EXPECT_EQ(simple.GetHiddenActivation(), ActivationType::Relu);
    EXPECT_EQ(simple.GetOutputActivation(), ActivationType::None);
    EXPECT_EQ(simple.GetDropoutRate(), 0.0f);
    EXPECT_FALSE(simple.IsCompiled());
    
    // Construction with activations
    FFN withActivations({4, 8, 3}, ActivationType::Tanh, ActivationType::Sigmoid, 0.1f);
    EXPECT_EQ(withActivations.GetHiddenActivation(), ActivationType::Tanh);
    EXPECT_EQ(withActivations.GetOutputActivation(), ActivationType::Sigmoid);
    EXPECT_EQ(withActivations.GetDropoutRate(), 0.1f);
    
    // Invalid construction
    EXPECT_THROW(FFN({3}), std::invalid_argument);  // Too few layers
    EXPECT_THROW(FFN({}), std::invalid_argument);   // Empty layers
}

// Forward pass tests
TEST_F(FFNTest, ForwardPass) {
    FFN network({3, 5, 2});
    
    // Single input
    Tensor output = network.Forward(testInput);
    EXPECT_EQ(output.Shape().size(), 2);
    EXPECT_EQ(output.Shape()[0], 1);
    EXPECT_EQ(output.Shape()[1], 2);
    
    // Batch input
    Tensor batchOutput = network.Forward(batchInput);
    EXPECT_EQ(batchOutput.Shape()[0], 2);
    EXPECT_EQ(batchOutput.Shape()[1], 2);
    
    // Different activations
    FFN withSigmoid({3, 4, 1}, ActivationType::Relu, ActivationType::Sigmoid);
    Tensor sigmoidOutput = withSigmoid.Forward(testInput);
    
    // Sigmoid output should be between 0 and 1
    for (size_t i = 0; i < sigmoidOutput.Size(); ++i) {
        EXPECT_GE(sigmoidOutput[i], 0.0f);
        EXPECT_LE(sigmoidOutput[i], 1.0f);
    }
}

// Parameter management tests
TEST_F(FFNTest, Parameters) {
    FFN network({2, 4, 3});
    
    auto params = network.Parameters();
    EXPECT_GT(params.size(), 0);  // Should have parameters
    
    size_t totalParams = network.CountParameters();
    EXPECT_GT(totalParams, 0);
    
    // Expected parameters: (2*4 + 4) + (4*3 + 3) = 8+4 + 12+3 = 27
    EXPECT_EQ(totalParams, 27);
    
    // Zero gradients
    network.ZeroGrad();
    // Should not throw
}

// Training mode tests
TEST_F(FFNTest, TrainingMode) {
    FFN network({3, 5, 2}, ActivationType::Relu, ActivationType::None, 0.2f);
    
    // Default should be training mode
    EXPECT_TRUE(network.IsTraining());
    
    // Set to inference mode
    network.SetTraining(false);
    EXPECT_FALSE(network.IsTraining());
    
    // Set back to training mode
    network.SetTraining(true);
    EXPECT_TRUE(network.IsTraining());
}

// Compilation tests
TEST_F(FFNTest, Compilation) {
    FFN network({2, 4, 1});
    
    // Before compilation
    EXPECT_FALSE(network.IsCompiled());
    EXPECT_EQ(network.GetOptimizer(), nullptr);
    EXPECT_EQ(network.GetLossFunction(), nullptr);
    
    // Valid compilation
    auto optimizer = std::make_unique<SGD>(0.01f);
    auto lossFunction = std::make_unique<MSELoss>();
    network.Compile(std::move(optimizer), std::move(lossFunction));
    
    // After compilation
    EXPECT_TRUE(network.IsCompiled());
    EXPECT_NE(network.GetOptimizer(), nullptr);
    EXPECT_NE(network.GetLossFunction(), nullptr);
    EXPECT_EQ(network.GetOptimizer()->GetName(), "SGD");
    EXPECT_EQ(network.GetLossFunction()->GetName(), "MSELoss");
    
    // Invalid compilation (null pointers)
    FFN network2({2, 3, 1});
    EXPECT_THROW(network2.Compile(nullptr, std::make_unique<MSELoss>()), std::invalid_argument);
    EXPECT_THROW(network2.Compile(std::make_unique<SGD>(0.01f), nullptr), std::invalid_argument);
}

// Prediction tests
TEST_F(FFNTest, Prediction) {
    FFN network({2, 4, 1});
    
    // Before compilation - should work
    auto predictions = network.Predict(trainInputs);
    EXPECT_EQ(predictions.size(), trainInputs.size());
    
    for (const auto& pred : predictions) {
        EXPECT_EQ(pred.Shape()[0], 1);
        EXPECT_EQ(pred.Shape()[1], 1);
    }
    
    // After compilation
    network.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    auto predictions2 = network.Predict(trainInputs);
    EXPECT_EQ(predictions2.size(), trainInputs.size());
    
    // Prediction should set training mode to false
    network.SetTraining(true);
    network.Predict(trainInputs);
    EXPECT_FALSE(network.IsTraining());
}

// Training tests
TEST_F(FFNTest, Training) {
    FFN network({2, 8, 1});
    
    // Cannot train without compilation
    EXPECT_THROW(network.Train(trainInputs, trainTargets, 1), std::runtime_error);
    
    // Compile and train
    network.Compile(std::make_unique<SGD>(0.1f), std::make_unique<MSELoss>());
    
    // Store initial predictions
    auto initialPredictions = network.Predict(trainInputs);
    
    // Train for a few epochs
    auto history = network.Train(trainInputs, trainTargets, 10, 0, nullptr, nullptr, false);
    
    // Check training history
    EXPECT_EQ(history.size(), 10);
    
    // Loss should generally decrease (or at least change)
    EXPECT_NE(history.front(), history.back());
    
    // Predictions should change after training
    auto trainedPredictions = network.Predict(trainInputs);
    bool predictionsChanged = false;
    for (size_t i = 0; i < initialPredictions.size(); ++i) {
        if (std::abs(initialPredictions[i][0] - trainedPredictions[i][0]) > 1e-6f) {
            predictionsChanged = true;
            break;
        }
    }
    EXPECT_TRUE(predictionsChanged);
}

// Regression training test
TEST_F(FFNTest, RegressionTraining) {
    FFN regressor({1, 8, 1});
    regressor.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Train on simple linear function y = 2x
    auto history = regressor.Train(regressionInputs, regressionTargets, 50, 0, nullptr, nullptr, false);
    
    // Test predictions
    auto predictions = regressor.Predict(regressionInputs);
    
    // Should learn approximately y = 2x
    for (size_t i = 0; i < predictions.size(); ++i) {
        float expected = regressionTargets[i][0];
        float predicted = predictions[i][0];
        float error = std::abs(expected - predicted);
        
        // Allow for some training error
        EXPECT_LT(error, 1.0f) << "Input: " << regressionInputs[i][0] 
                               << ", Expected: " << expected 
                               << ", Got: " << predicted;
    }
}

// Classification training test
TEST_F(FFNTest, ClassificationTraining) {
    FFN classifier({2, 16, 1}, ActivationType::Relu, ActivationType::Sigmoid);
    classifier.Compile(std::make_unique<SGD>(0.5f), std::make_unique<BCELoss>());
    
    // Train on XOR-like problem
    auto history = classifier.Train(trainInputs, trainTargets, 100, 0, nullptr, nullptr, false);
    
    // Test classification accuracy
    auto predictions = classifier.Predict(trainInputs);
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float predicted = predictions[i][0];
        float expected = trainTargets[i][0];
        int predictedClass = (predicted > 0.5f) ? 1 : 0;
        int expectedClass = (expected > 0.5f) ? 1 : 0;
        
        if (predictedClass == expectedClass) {
            correct++;
        }
    }
    
    // Should achieve reasonable accuracy (at least 50% for random guessing)
    float accuracy = static_cast<float>(correct) / predictions.size();
    EXPECT_GE(accuracy, 0.5f);
}

// Batch training test
TEST_F(FFNTest, BatchTraining) {
    FFN network({2, 4, 1});
    network.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Train with different batch sizes
    auto history1 = network.Train(trainInputs, trainTargets, 5, 1, nullptr, nullptr, false);  // SGD
    auto history2 = network.Train(trainInputs, trainTargets, 5, 2, nullptr, nullptr, false);  // Mini-batch
    auto history3 = network.Train(trainInputs, trainTargets, 5, 0, nullptr, nullptr, false);  // Full batch
    
    EXPECT_EQ(history1.size(), 5);
    EXPECT_EQ(history2.size(), 5);
    EXPECT_EQ(history3.size(), 5);
}

// Architecture information tests
TEST_F(FFNTest, ArchitectureInfo) {
    FFN network({10, 32, 16, 5}, ActivationType::Tanh, ActivationType::Sigmoid, 0.1f);
    
    // Test layer sizes
    auto layerSizes = network.GetLayerSizes();
    std::vector<size_t> expected = {10, 32, 16, 5};
    EXPECT_EQ(layerSizes, expected);
    
    // Test individual layer access
    EXPECT_EQ(network.GetNumModules(), 6);  // 3 Linear + 2 Tanh + 1 Sigmoid
    
    for (size_t i = 0; i < network.GetNumModules(); ++i) {
        auto* layer = network.GetLayer(i);
        EXPECT_NE(layer, nullptr);
        EXPECT_FALSE(layer->GetName().empty());
    }
    
    // Test out of range access
    EXPECT_THROW(network.GetLayer(network.GetNumModules()), std::out_of_range);
    
    // Test const version
    const auto& constNetwork = network;
    auto* constLayer = constNetwork.GetLayer(0);
    EXPECT_NE(constLayer, nullptr);
}

// Network name and summary tests
TEST_F(FFNTest, NetworkInfo) {
    FFN network({4, 8, 2});
    
    EXPECT_EQ(network.GetName(), "FFN");
    
    // PrintArchitecture should not throw
    EXPECT_NO_THROW(network.PrintArchitecture());
}

// Large network test
TEST_F(FFNTest, LargeNetwork) {
    FFN largeNetwork({100, 256, 128, 64, 32, 10});
    
    EXPECT_EQ(largeNetwork.CountParameters(), 
              (100*256 + 256) + (256*128 + 128) + (128*64 + 64) + (64*32 + 32) + (32*10 + 10));
    
    Tensor largeInput = Tensor::Randn({1, 100});
    Tensor output = largeNetwork.Forward(largeInput);
    
    EXPECT_EQ(output.Shape()[0], 1);
    EXPECT_EQ(output.Shape()[1], 10);
}

// Edge cases and error handling
TEST_F(FFNTest, EdgeCases) {
    FFN network({2, 3, 1});
    
    // Test with zero input
    Tensor zeroInput({2}, 0.0f);
    EXPECT_NO_THROW(network.Forward(zeroInput));
    
    // Test with very large input
    Tensor largeInput({2}, 1e6f);
    EXPECT_NO_THROW(network.Forward(largeInput));
    
    // Test with very small input
    Tensor smallInput({2}, 1e-6f);
    EXPECT_NO_THROW(network.Forward(smallInput));
} 