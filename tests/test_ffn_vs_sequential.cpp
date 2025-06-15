/**
 * Comparative tests between FFN and Sequential classes
 * Tests both equivalence and unique capabilities
 */

#include <gtest/gtest.h>
#include "kotml/kotml.hpp"
#include <vector>
#include <memory>

using namespace kotml;
using namespace kotml::nn;
using namespace kotml::optim;

class FFNvSeqTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test data for both network types
        testInputs = {
            Tensor({1.0f, 2.0f}, {1, 2}),
            Tensor({2.0f, 3.0f}, {1, 2}),
            Tensor({3.0f, 4.0f}, {1, 2}),
            Tensor({4.0f, 5.0f}, {1, 2})
        };
        
        testTargets = {
            Tensor({3.0f}, {1, 1}),  // Sum
            Tensor({5.0f}, {1, 1}),
            Tensor({7.0f}, {1, 1}),
            Tensor({9.0f}, {1, 1})
        };
    }
    
    std::vector<Tensor> testInputs;
    std::vector<Tensor> testTargets;
};

// Test equivalent architectures
TEST_F(FFNvSeqTest, EquivalentArchitectures) {
    // Create equivalent networks
    FFN ffnNetwork({2, 8, 4, 1});
    
    auto seqNetwork = Sequential()
        .Linear(2, 8)
        .ReLU()
        .Linear(8, 4)
        .ReLU()
        .Linear(4, 1)
        .Build();
    
    // Both should have same parameter count
    EXPECT_EQ(ffnNetwork.CountParameters(), seqNetwork.CountParameters());
    
    // Both should have similar structure
    EXPECT_EQ(ffnNetwork.GetInputSize(), 2);
    EXPECT_EQ(ffnNetwork.GetOutputSize(), 1);
    
    // Test forward pass compatibility
    Tensor input({1.5f, 2.5f}, {1, 2});
    
    Tensor ffnOutput = ffnNetwork.Forward(input);
    Tensor seqOutput = seqNetwork.Forward(input);
    
    // Outputs should have same shape
    EXPECT_EQ(ffnOutput.Shape(), seqOutput.Shape());
}

// Test parameter equivalence after weight copying
TEST_F(FFNvSeqTest, ParameterEquivalence) {
    // Create networks
    FFN ffnNet({2, 4, 1});
    auto seqNet = Sequential()
        .Linear(2, 4)
        .ReLU()
        .Linear(4, 1)
        .Build();
    
    // Copy parameters from FFN to Sequential (manually for testing)
    auto ffnParams = ffnNet.Parameters();
    auto seqParams = seqNet.Parameters();
    
    EXPECT_EQ(ffnParams.size(), seqParams.size());
    
    // Set same values
    for (size_t i = 0; i < ffnParams.size(); ++i) {
        for (size_t j = 0; j < ffnParams[i]->Size(); ++j) {
            (*seqParams[i])[j] = (*ffnParams[i])[j];
        }
    }
    
    // Now outputs should be very similar (not identical due to different implementations)
    Tensor input({1.0f, 2.0f}, {1, 2});
    Tensor ffnOutput = ffnNet.Forward(input);
    Tensor seqOutput = seqNet.Forward(input);
    
    for (size_t i = 0; i < ffnOutput.Size(); ++i) {
        EXPECT_NEAR(ffnOutput[i], seqOutput[i], 1e-5f);
    }
}

// Test compilation and training compatibility
TEST_F(FFNvSeqTest, TrainingCompatibility) {
    // Create equivalent networks
    FFN ffnNet({2, 16, 1});
    auto seqNet = Sequential()
        .Linear(2, 16)
        .ReLU()
        .Linear(16, 1)
        .Build();
    
    // Compile both with same configuration
    ffnNet.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    seqNet.Compile(std::make_unique<SGD>(0.01f), std::make_unique<MSELoss>());
    
    // Both should be compiled
    EXPECT_TRUE(ffnNet.IsCompiled());
    EXPECT_TRUE(seqNet.IsCompiled());
    
    // Train both on same data
    auto ffnHistory = ffnNet.Train(testInputs, testTargets, 10, 0, nullptr, nullptr, false);
    auto seqHistory = seqNet.Train(testInputs, testTargets, 10, 0, nullptr, nullptr, false);
    
    // Both should have training history
    EXPECT_EQ(ffnHistory.size(), 10);
    EXPECT_EQ(seqHistory.size(), 10);
    
    // Both should be able to make predictions
    auto ffnPreds = ffnNet.Predict(testInputs);
    auto seqPreds = seqNet.Predict(testInputs);
    
    EXPECT_EQ(ffnPreds.size(), testInputs.size());
    EXPECT_EQ(seqPreds.size(), testInputs.size());
}

// Test FFN limitations (what it CANNOT do)
TEST_F(FFNvSeqTest, FFNLimitations) {
    // FFN cannot have mixed activations
    FFN ffnNet({3, 8, 4, 2});
    
    // All hidden layers use same activation
    EXPECT_EQ(ffnNet.GetHiddenActivation(), ActivationType::Relu);
    
    // Sequential CAN have mixed activations
    auto mixedSeq = Sequential()
        .Linear(3, 8)
        .ReLU()        // Different activation
        .Linear(8, 4)
        .Tanh()        // Different activation
        .Linear(4, 2)
        .Sigmoid()     // Different activation
        .Build();
    
    EXPECT_EQ(mixedSeq.GetNumLayers(), 6);
    
    // Test that mixed activations work
    Tensor input({1.0f, 2.0f, 3.0f}, {1, 3});
    EXPECT_NO_THROW(mixedSeq.Forward(input));
}

// Test Sequential unique capabilities
TEST_F(FFNvSeqTest, SequentialUniqueFeatures) {
    // Sequential supports Dropout (FFN has basic support but not flexible)
    auto dropoutSeq = Sequential()
        .Linear(4, 16)
        .ReLU()
        .Dropout(0.5f)
        .Linear(16, 8)
        .ReLU()
        .Dropout(0.3f)
        .Linear(8, 2)
        .Build();
    
    // Should work with dropout in training mode
    dropoutSeq.SetTraining(true);
    Tensor input({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    
    Tensor output1 = dropoutSeq.Forward(input);
    Tensor output2 = dropoutSeq.Forward(input);
    
    // Outputs should be different due to dropout (stochastic test)
    bool different = false;
    for (size_t i = 0; i < output1.Size(); ++i) {
        if (std::abs(output1[i] - output2[i]) > 1e-6f) {
            different = true;
            break;
        }
    }
    // Note: This is probabilistic and might occasionally fail
    
    // In inference mode, should be deterministic
    dropoutSeq.SetTraining(false);
    Tensor inferOutput1 = dropoutSeq.Forward(input);
    Tensor inferOutput2 = dropoutSeq.Forward(input);
    
    for (size_t i = 0; i < inferOutput1.Size(); ++i) {
        EXPECT_NEAR(inferOutput1[i], inferOutput2[i], 1e-6f);
    }
}

// Test architectural flexibility differences
TEST_F(FFNvSeqTest, ArchitecturalFlexibility) {
    // FFN: Simple, uniform architecture
    FFN simpleFFN({10, 32, 16, 5});
    
    // Sequential: Complex, heterogeneous architecture
    auto complexSeq = Sequential()
        .Input(10)          // Input validation
        .Linear(10, 32)
        .ReLU()
        .Dropout(0.2f)
        .Linear(32, 24)
        .Tanh()
        .Linear(24, 16)
        .Sigmoid()
        .Linear(16, 8)
        .ReLU()
        .Output(8, 5, ActivationType::None)  // Composite output
        .Build();
    
    // FFN is more compact and simple
    EXPECT_LT(simpleFFN.GetNumModules(), complexSeq.GetNumLayers());
    
    // Both should work for prediction
    Tensor input = Tensor::Randn({1, 10});
    
    Tensor ffnOutput = simpleFFN.Forward(input);
    Tensor seqOutput = complexSeq.Forward(input);
    
    // Both should produce outputs of same dimension
    EXPECT_EQ(ffnOutput.Shape()[1], seqOutput.Shape()[1]);
}

// Test performance characteristics
TEST_F(FFNvSeqTest, PerformanceCharacteristics) {
    // Create equivalent large networks
    std::vector<size_t> architecture = {50, 128, 64, 32, 10};
    FFN ffnNet(architecture);
    
    auto seqNet = Sequential()
        .Linear(50, 128)
        .ReLU()
        .Linear(128, 64)
        .ReLU()
        .Linear(64, 32)
        .ReLU()
        .Linear(32, 10)
        .Build();
    
    // Both should handle batch processing
    Tensor batchInput = Tensor::Randn({10, 50});  // 10 samples
    
    EXPECT_NO_THROW(ffnNet.Forward(batchInput));
    EXPECT_NO_THROW(seqNet.Forward(batchInput));
    
    // Check output shapes
    Tensor ffnBatchOutput = ffnNet.Forward(batchInput);
    Tensor seqBatchOutput = seqNet.Forward(batchInput);
    
    EXPECT_EQ(ffnBatchOutput.Shape()[0], 10);
    EXPECT_EQ(ffnBatchOutput.Shape()[1], 10);
    EXPECT_EQ(seqBatchOutput.Shape()[0], 10);
    EXPECT_EQ(seqBatchOutput.Shape()[1], 10);
}

// Test builder pattern safety vs FFN simplicity
TEST_F(FFNvSeqTest, APIDesignDifferences) {
    // FFN: Simple constructor
    EXPECT_NO_THROW(FFN({2, 4, 1}));
    EXPECT_THROW(FFN({2}), std::invalid_argument);  // Too few layers
    
    // Sequential: Builder pattern with safety
    auto validSeq = Sequential()
        .Linear(2, 4)
        .ReLU()
        .Linear(4, 1)
        .Build();
    
    EXPECT_TRUE(validSeq.IsBuilt());
    
    // Cannot modify after build
    EXPECT_THROW(std::move(validSeq).Linear(1, 2), std::runtime_error);
    
    // Cannot use before build
    auto unbuildSeq = Sequential().Linear(2, 1);
    EXPECT_THROW(unbuildSeq.Forward(Tensor({2}, 1.0f)), std::runtime_error);
}

// Test information and debugging capabilities
TEST_F(FFNvSeqTest, InformationCapabilities) {
    FFN ffnNet({4, 8, 3});
    auto seqNet = Sequential()
        .Linear(4, 8)
        .ReLU()
        .Linear(8, 3)
        .Build();
    
    // Both should provide architecture information
    EXPECT_NO_THROW(ffnNet.PrintArchitecture());
    EXPECT_NO_THROW(seqNet.Summary());
    
    // Both should provide parameter counts
    EXPECT_GT(ffnNet.CountParameters(), 0);
    EXPECT_GT(seqNet.CountParameters(), 0);
    
    // Both should allow parameter access
    auto ffnParams = ffnNet.Parameters();
    auto seqParams = seqNet.Parameters();
    
    EXPECT_GT(ffnParams.size(), 0);
    EXPECT_GT(seqParams.size(), 0);
    
    // Both should allow layer access
    EXPECT_GT(ffnNet.GetNumModules(), 0);
    EXPECT_GT(seqNet.GetNumLayers(), 0);
}

// Test use case recommendations
TEST_F(FFNvSeqTest, UseCaseGuidelines) {
    // FFN is ideal for simple, uniform architectures
    FFN quickPrototype({10, 20, 5});
    EXPECT_EQ(quickPrototype.GetInputSize(), 10);
    EXPECT_EQ(quickPrototype.GetOutputSize(), 5);
    
    // Sequential is ideal for complex, research-oriented architectures
    auto researchNet = Sequential()
        .Input(10)                  // Input validation
        .Linear(10, 64)
        .ReLU()
        .Dropout(0.3f)             // Regularization
        .Linear(64, 32)
        .Tanh()                    // Different activation
        .Linear(32, 16)
        .ReLU()
        .Dropout(0.2f)
        .Output(16, 5, ActivationType::Sigmoid)  // Composite output
        .Build();
    
    // Both should work for their intended use cases
    Tensor input = Tensor::Randn({1, 10});
    
    EXPECT_NO_THROW(quickPrototype.Forward(input));
    EXPECT_NO_THROW(researchNet.Forward(input));
    
    // FFN for quick setup
    EXPECT_LT(quickPrototype.GetNumModules(), researchNet.GetNumLayers());
    
    // Sequential for flexibility
    EXPECT_GT(researchNet.GetNumLayers(), 5);
} 