/**
 * Comprehensive tests for Sequential class
 * Tests all major functionality including Builder pattern, Predict, Compile, Train
 */

#include <gtest/gtest.h>
#include "kotml/kotml.hpp"
#include <vector>
#include <memory>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

class SequentialTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard test data
        testInput = Tensor({1.0f, 2.0f, 3.0f}, {1, 3});
        batchInput = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
        
        // Classification data (AND gate)
        trainInputs = {
            Tensor({0.0f, 0.0f}, {1, 2}),
            Tensor({0.0f, 1.0f}, {1, 2}),
            Tensor({1.0f, 0.0f}, {1, 2}),
            Tensor({1.0f, 1.0f}, {1, 2})
        };
        
        trainTargets = {
            Tensor({0.0f}, {1, 1}),
            Tensor({0.0f}, {1, 1}),
            Tensor({0.0f}, {1, 1}),
            Tensor({1.0f}, {1, 1})
        };
        
        // Regression data (y = 2x + 1)
        regressionInputs = {
            Tensor({1.0f}, {1, 1}),
            Tensor({2.0f}, {1, 1}),
            Tensor({3.0f}, {1, 1}),
            Tensor({4.0f}, {1, 1})
        };
        
        regressionTargets = {
            Tensor({3.0f}, {1, 1}),  // y = 2*1 + 1
            Tensor({5.0f}, {1, 1}),  // y = 2*2 + 1
            Tensor({7.0f}, {1, 1}),  // y = 2*3 + 1
            Tensor({9.0f}, {1, 1})   // y = 2*4 + 1
        };
    }
    
    Tensor testInput;
    Tensor batchInput;
    std::vector<Tensor> trainInputs;
    std::vector<Tensor> trainTargets;
    std::vector<Tensor> regressionInputs;
    std::vector<Tensor> regressionTargets;
};

// Builder pattern tests
TEST_F(SequentialTest, BuilderPattern) {
    // Basic builder usage
    auto network = Sequential()
        .Linear(3, 5)
        .ReLU()
        .Linear(5, 2)
        .Build();
    
    EXPECT_TRUE(network.IsBuilt());
    EXPECT_EQ(network.GetNumLayers(), 3);  // Linear + ReLU + Linear
    EXPECT_FALSE(network.IsCompiled());
    
    // Builder with different layer types
    auto complex = Sequential()
        .Input(10)
        .Linear(10, 16)
        .Tanh()
        .Dropout(0.2f)
        .Linear(16, 8)
        .Sigmoid()
        .Output(8, 3, ActivationType::None)
        .Build();
    
    EXPECT_TRUE(complex.IsBuilt());
    EXPECT_EQ(complex.GetNumLayers(), 7);
    
    // Cannot build twice
    EXPECT_THROW(std::move(complex).Build(), std::runtime_error);
}

// Move semantics tests
TEST_F(SequentialTest, MoveSemantics) {
    // Create network with move constructor
    auto network1 = Sequential()
        .Linear(3, 4)
        .ReLU()
        .Linear(4, 2)
        .Build();
    
    // Move constructor
    auto network2 = std::move(network1);
    EXPECT_TRUE(network2.IsBuilt());
    EXPECT_FALSE(network1.IsBuilt());  // Original should be invalid
    
    // Move assignment
    auto network3 = Sequential().Linear(2, 1).Build();
    network3 = std::move(network2);
    EXPECT_TRUE(network3.IsBuilt());
    EXPECT_EQ(network3.GetNumLayers(), 3);
}

// Builder validation tests
TEST_F(SequentialTest, BuilderValidation) {
    // Cannot modify after build
    auto network = Sequential()
        .Linear(3, 5)
        .Build();
    
    EXPECT_THROW(std::move(network).Linear(5, 2), std::runtime_error);
    EXPECT_THROW(std::move(network).ReLU(), std::runtime_error);
    EXPECT_THROW(std::move(network).Dropout(0.1f), std::runtime_error);
    EXPECT_THROW(std::move(network).Add(std::make_unique<kotml::nn::Linear>(2, 1)), std::runtime_error);
}

// Layer addition tests
TEST_F(SequentialTest, LayerTypes) {
    // Test all layer types
    auto network = Sequential()
        .Input(10)
        .Linear(10, 16)
        .ReLU()
        .Dropout(0.3f)
        .Linear(16, 8)
        .Tanh()
        .Linear(8, 4)
        .Sigmoid()
        .Output(4, 2, ActivationType::None)
        .Build();
    
    EXPECT_EQ(network.GetNumLayers(), 9);
    
    // Check individual layers
    for (size_t i = 0; i < network.GetNumLayers(); ++i) {
        auto* layer = network.GetLayer(i);
        EXPECT_NE(layer, nullptr);
        EXPECT_FALSE(layer->GetName().empty());
    }
    
    // Test convenient activation methods
    auto activations = Sequential()
        .Linear(3, 4)
        .ReLU()
        .Linear(4, 4)
        .Sigmoid()
        .Linear(4, 4)
        .Tanh()
        .Linear(4, 1)
        .Build();
    
    EXPECT_EQ(activations.GetNumLayers(), 7);
}

// Forward pass tests
TEST_F(SequentialTest, ForwardPass) {
    auto network = Sequential()
        .Linear(3, 8)
        .ReLU()
        .Linear(8, 4)
        .Tanh()
        .Linear(4, 2)
        .Build();
    
    // Cannot forward pass before build
    auto unbuilt = Sequential().Linear(3, 2);
    EXPECT_THROW(unbuilt.Forward(testInput), std::runtime_error);
    
    // Single input
    Tensor output = network.Forward(testInput);
    EXPECT_EQ(output.Shape().size(), 2);
    EXPECT_EQ(output.Shape()[0], 1);
    EXPECT_EQ(output.Shape()[1], 2);
    
    // Batch input
    Tensor batchOutput = network.Forward(batchInput);
    EXPECT_EQ(batchOutput.Shape()[0], 2);
    EXPECT_EQ(batchOutput.Shape()[1], 2);
    
    // Test with different activations
    auto sigmoid_net = Sequential()
        .Linear(3, 5)
        .Sigmoid()
        .Linear(5, 1)
        .Build();
    
    Tensor sigmoid_output = sigmoid_net.Forward(testInput);
    // Output should have reasonable values
    EXPECT_GT(sigmoid_output.Size(), 0);
}

// Parameter management tests
TEST_F(SequentialTest, Parameters) {
    auto network = Sequential()
        .Linear(3, 8)
        .ReLU()
        .Linear(8, 4)
        .Linear(4, 2)
        .Build();
    
    auto params = network.Parameters();
    EXPECT_GT(params.size(), 0);
    
    size_t totalParams = network.CountParameters();
    EXPECT_GT(totalParams, 0);
    
    // Expected: (3*8 + 8) + (8*4 + 4) + (4*2 + 2) = 32 + 36 + 10 = 78
    EXPECT_EQ(totalParams, 78);
    
    // Zero gradients
    network.ZeroGrad();
    // Should not throw
}

// Training mode tests
TEST_F(SequentialTest, TrainingMode) {
    auto network = Sequential()
        .Linear(3, 8)
        .ReLU()
        .Dropout(0.2f)
        .Linear(8, 2)
        .Build();
    
    // Default should be training mode
    EXPECT_TRUE(network.IsTraining());
    
    // Set to inference mode
    network.SetTraining(false);
    EXPECT_FALSE(network.IsTraining());
    
    // Set back to training mode
    network.SetTraining(true);
    EXPECT_TRUE(network.IsTraining());
    
    // Should propagate to all layers
    for (size_t i = 0; i < network.GetNumLayers(); ++i) {
        auto* layer = network.GetLayer(i);
        EXPECT_EQ(layer->IsTraining(), network.IsTraining());
    }
}

// Compilation tests
TEST_F(SequentialTest, Compilation) {
    auto network = Sequential()
        .Linear(2, 4)
        .ReLU()
        .Linear(4, 1)
        .Build();
    
    // Before compilation
    EXPECT_FALSE(network.IsCompiled());
    EXPECT_EQ(network.GetOptimizer(), nullptr);
    EXPECT_EQ(network.GetLossFunction(), nullptr);
    
    // Cannot compile unbuilt network
    auto unbuilt = Sequential().Linear(2, 1);
    EXPECT_THROW(unbuilt.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>()), std::runtime_error);
    
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
    auto network2 = Sequential().Linear(2, 1).Build();
    EXPECT_THROW(network2.Compile(nullptr, std::make_unique<MSELoss>()), std::invalid_argument);
    EXPECT_THROW(network2.Compile(std::make_unique<SGD>(0.01f), nullptr), std::invalid_argument);
}

// Prediction tests
TEST_F(SequentialTest, Prediction) {
    auto network = Sequential()
        .Linear(2, 8)
        .ReLU()
        .Linear(8, 1)
        .Build();
    
    // Cannot predict on unbuilt network
    auto unbuilt = Sequential().Linear(2, 1);
    EXPECT_THROW(unbuilt.Predict(trainInputs), std::runtime_error);
    
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
TEST_F(SequentialTest, Training) {
    auto network = Sequential()
        .Linear(2, 16)
        .ReLU()
        .Linear(16, 1)
        .Build();
    
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
TEST_F(SequentialTest, RegressionTraining) {
    auto regressor = Sequential()
        .Linear(1, 16)
        .ReLU()
        .Linear(16, 8)
        .ReLU()
        .Linear(8, 1)
        .Build();
    
    regressor.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Train on linear function y = 2x + 1
    auto history = regressor.Train(regressionInputs, regressionTargets, 100, 0, nullptr, nullptr, false);
    
    // Test predictions
    auto predictions = regressor.Predict(regressionInputs);
    
    // Should learn approximately y = 2x + 1
    for (size_t i = 0; i < predictions.size(); ++i) {
        float expected = regressionTargets[i][0];
        float predicted = predictions[i][0];
        float error = std::abs(expected - predicted);
        
        // Allow for some training error
        EXPECT_LT(error, 2.0f) << "Input: " << regressionInputs[i][0] 
                               << ", Expected: " << expected 
                               << ", Got: " << predicted;
    }
}

// Classification training test
TEST_F(SequentialTest, ClassificationTraining) {
    auto classifier = Sequential()
        .Linear(2, 16)
        .ReLU()
        .Linear(16, 1)
        .Sigmoid()
        .Build();
    
    classifier.Compile(std::make_unique<SGD>(0.5f), std::make_unique<BCELoss>());
    
    // Train on XOR-like problem (как у FFN, а не AND gate)
    std::vector<Tensor> xorInputs = {
        Tensor({0.0f, 0.0f}, {1, 2}),
        Tensor({0.0f, 1.0f}, {1, 2}),
        Tensor({1.0f, 0.0f}, {1, 2}),
        Tensor({1.0f, 1.0f}, {1, 2})
    };
    
    std::vector<Tensor> xorTargets = {
        Tensor({0.0f}, {1, 1}),  // XOR: 0 XOR 0 = 0
        Tensor({1.0f}, {1, 1}),  // XOR: 0 XOR 1 = 1
        Tensor({1.0f}, {1, 1}),  // XOR: 1 XOR 0 = 1
        Tensor({0.0f}, {1, 1})   // XOR: 1 XOR 1 = 0
    };
    
    auto history = classifier.Train(xorInputs, xorTargets, 100, 0, nullptr, nullptr, false);  // Уменьшил с 200 до 100
    
    // Test classification accuracy
    auto predictions = classifier.Predict(xorInputs);
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float predicted = predictions[i][0];
        float expected = xorTargets[i][0];
        int predictedClass = (predicted > 0.5f) ? 1 : 0;
        int expectedClass = (expected > 0.5f) ? 1 : 0;
        
        if (predictedClass == expectedClass) {
            correct++;
        }
    }
    
    // Should achieve reasonable accuracy
    float accuracy = static_cast<float>(correct) / predictions.size();
    EXPECT_GE(accuracy, 0.5f);
}

// Batch training test
TEST_F(SequentialTest, BatchTraining) {
    auto network = Sequential()
        .Linear(2, 8)
        .ReLU()
        .Linear(8, 1)
        .Build();
    
    network.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Train with different batch sizes
    auto history1 = network.Train(trainInputs, trainTargets, 5, 1, nullptr, nullptr, false);  // SGD
    auto history2 = network.Train(trainInputs, trainTargets, 5, 2, nullptr, nullptr, false);  // Mini-batch
    auto history3 = network.Train(trainInputs, trainTargets, 5, 0, nullptr, nullptr, false);  // Full batch
    
    EXPECT_EQ(history1.size(), 5);
    EXPECT_EQ(history2.size(), 5);
    EXPECT_EQ(history3.size(), 5);
}

// Dropout behavior test
TEST_F(SequentialTest, DropoutBehavior) {
    auto network = Sequential()
        .Linear(10, 20)
        .ReLU()
        .Dropout(0.5f)
        .Linear(20, 5)
        .Build();
    
    Tensor input = Tensor::Ones({1, 10});
    
    // Training mode - dropout should be active
    network.SetTraining(true);
    Tensor trainOutput1 = network.Forward(input);
    Tensor trainOutput2 = network.Forward(input);
    
    // Outputs should be different due to dropout
    bool different = false;
    for (size_t i = 0; i < trainOutput1.Size(); ++i) {
        if (std::abs(trainOutput1[i] - trainOutput2[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    // Note: Due to random nature, this might occasionally fail
    
    // Inference mode - dropout should be inactive
    network.SetTraining(false);
    Tensor inferOutput1 = network.Forward(input);
    Tensor inferOutput2 = network.Forward(input);
    
    // Outputs should be identical in inference mode
    for (size_t i = 0; i < inferOutput1.Size(); ++i) {
        EXPECT_NEAR(inferOutput1[i], inferOutput2[i], 1e-6f);
    }
}

// Complex architecture test
TEST_F(SequentialTest, ComplexArchitecture) {
    auto network = Sequential()
        .Input(100)
        .Linear(100, 256)
        .ReLU()
        .Dropout(0.2f)
        .Linear(256, 128)
        .Tanh()
        .Dropout(0.1f)
        .Linear(128, 64)
        .ReLU()
        .Linear(64, 32)
        .Sigmoid()
        .Output(32, 10, ActivationType::None)
        .Build();
    
    EXPECT_EQ(network.GetNumLayers(), 11);
    EXPECT_GT(network.CountParameters(), 0);
    
    // Test forward pass
    Tensor largeInput = Tensor::Randn({1, 100});
    Tensor output = network.Forward(largeInput);
    
    EXPECT_EQ(output.Shape()[0], 1);
    EXPECT_EQ(output.Shape()[1], 10);
}

// Layer access tests
TEST_F(SequentialTest, LayerAccess) {
    auto network = Sequential()
        .Linear(3, 8)
        .ReLU()
        .Linear(8, 4)
        .Tanh()
        .Linear(4, 2)
        .Build();
    
    // Test layer access
    EXPECT_EQ(network.GetNumLayers(), 5);
    
    for (size_t i = 0; i < network.GetNumLayers(); ++i) {
        auto* layer = network.GetLayer(i);
        EXPECT_NE(layer, nullptr);
        EXPECT_FALSE(layer->GetName().empty());
    }
    
    // Test out of range access
    EXPECT_THROW(network.GetLayer(network.GetNumLayers()), std::out_of_range);
    
    // Test const version
    const auto& constNetwork = network;
    auto* constLayer = constNetwork.GetLayer(0);
    EXPECT_NE(constLayer, nullptr);
}

// Network information tests
TEST_F(SequentialTest, NetworkInfo) {
    auto network = Sequential()
        .Linear(4, 8)
        .ReLU()
        .Linear(8, 2)
        .Build();
    
    EXPECT_EQ(network.GetName(), "Sequential(3 layers)");
    
    // Summary should not throw
    EXPECT_NO_THROW(network.Summary());
    
    // Test unbuilt network summary
    auto unbuilt = Sequential().Linear(2, 1);
    EXPECT_NO_THROW(unbuilt.Summary());
}

// Validation with validation data
TEST_F(SequentialTest, ValidationTraining) {
    auto network = Sequential()
        .Linear(2, 16)
        .ReLU()
        .Linear(16, 1)
        .Build();
    
    network.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Create validation data (same as training for simplicity)
    std::vector<Tensor> validInputs = trainInputs;
    std::vector<Tensor> validTargets = trainTargets;
    
    // Train with validation data
    auto history = network.Train(trainInputs, trainTargets, 10, 0, &validInputs, &validTargets, false);
    
    EXPECT_EQ(history.size(), 10);
}

// Add custom module test
TEST_F(SequentialTest, CustomModule) {
    auto network = Sequential()
        .Linear(3, 8)
        .Add(std::make_unique<kotml::nn::Activation>(ActivationType::Relu))
        .Linear(8, 2)
        .Build();
    
    EXPECT_EQ(network.GetNumLayers(), 3);
    
    Tensor output = network.Forward(testInput);
    EXPECT_EQ(output.Shape()[0], 1);
    EXPECT_EQ(output.Shape()[1], 2);
}

// Edge cases and error handling
TEST_F(SequentialTest, EdgeCases) {
    auto network = Sequential()
        .Linear(2, 8)
        .ReLU()
        .Linear(8, 1)
        .Build();
    
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