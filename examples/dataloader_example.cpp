#include "kotml/kotml.hpp"
#include "kotml/tensor.hpp"
#include "kotml/data/data.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace kotml;

// Helper function to print tensor info
void PrintTensorInfo(const Tensor& tensor, const std::string& name) {
    std::cout << name << ": shape=[";
    for (size_t i = 0; i < tensor.Ndim(); ++i) {
        std::cout << tensor.Shape()[i];
        if (i < tensor.Ndim() - 1) std::cout << ", ";
    }
    std::cout << "], size=" << tensor.Size();
    
    if (tensor.Size() <= 20) {
        std::cout << ", data=[";
        for (size_t i = 0; i < std::min(10ul, tensor.Size()); ++i) {
            std::cout << std::fixed << std::setprecision(3) << tensor[i];
            if (i < std::min(10ul, tensor.Size()) - 1) std::cout << ", ";
        }
        if (tensor.Size() > 10) std::cout << "...";
        std::cout << "]";
    }
    std::cout << std::endl;
}

// Helper function to print batch info
void PrintBatchInfo(const std::pair<Tensor, Tensor>& batch, const std::string& name) {
    std::cout << "=== " << name << " ===" << std::endl;
    PrintTensorInfo(batch.first, "Inputs");
    PrintTensorInfo(batch.second, "Targets");
    std::cout << std::endl;
}

// Demonstrate basic dataset usage
void DemonstrateDatasets() {
    std::cout << "=== Dataset Examples ===" << std::endl;
    
    // 1. TensorDataset
    std::cout << "1. TensorDataset:" << std::endl;
    {
        // Create sample data
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> target_data = {2.0f, 4.0f, 6.0f};
        
        Tensor inputs(input_data, {3, 2});  // 3 samples, 2 features each
        Tensor targets(target_data, {3});   // 3 targets
        
        data::TensorDataset dataset(inputs, targets);
        
        std::cout << "  Dataset: " << dataset.GetName() << std::endl;
        std::cout << "  Size: " << dataset.Size() << std::endl;
        std::cout << "  Input shape: [";
        for (size_t dim : dataset.GetInputShape()) std::cout << dim << " ";
        std::cout << "]" << std::endl;
        std::cout << "  Target shape: [";
        for (size_t dim : dataset.GetTargetShape()) std::cout << dim << " ";
        std::cout << "]" << std::endl;
        
        // Get individual samples
        for (size_t i = 0; i < dataset.Size(); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "  Sample " << i << ": input=[" << input[0] << ", " << input[1] 
                      << "], target=" << target[0] << std::endl;
        }
    }
    
    // 2. SyntheticDataset - Linear Regression
    std::cout << std::endl << "2. SyntheticDataset - Linear Regression:" << std::endl;
    {
        data::SyntheticDataset dataset(data::SyntheticDataset::DataType::LINEAR_REGRESSION, 
                                      5, 1, 1, 0.1f, 42);
        
        std::cout << "  Dataset: " << dataset.GetName() << std::endl;
        std::cout << "  Size: " << dataset.Size() << std::endl;
        
        for (size_t i = 0; i < dataset.Size(); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "  Sample " << i << ": x=" << std::fixed << std::setprecision(3) 
                      << input[0] << ", y=" << target[0] << std::endl;
        }
    }
    
    // 3. SyntheticDataset - Binary Classification
    std::cout << std::endl << "3. SyntheticDataset - Binary Classification:" << std::endl;
    {
        data::SyntheticDataset dataset(data::SyntheticDataset::DataType::BINARY_CLASSIFICATION, 
                                      5, 2, 2, 0.0f, 42);
        
        std::cout << "  Dataset: " << dataset.GetName() << std::endl;
        
        for (size_t i = 0; i < dataset.Size(); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "  Sample " << i << ": x=[" << std::fixed << std::setprecision(3) 
                      << input[0] << ", " << input[1] << "], class=[" 
                      << target[0] << ", " << target[1] << "]" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

// Demonstrate DataLoader functionality
void DemonstrateDataLoader() {
    std::cout << "=== DataLoader Examples ===" << std::endl;
    
    // Create a larger synthetic dataset
    auto dataset = std::make_shared<data::SyntheticDataset>(
        data::SyntheticDataset::DataType::LINEAR_REGRESSION, 100, 1, 1, 0.1f, 42);
    
    // 1. Basic DataLoader usage
    std::cout << "1. Basic DataLoader (batch_size=8, shuffle=true):" << std::endl;
    {
        data::DataLoader loader(dataset, 8, true, false, 42);
        
        std::cout << "  Dataset size: " << loader.GetDatasetSize() << std::endl;
        std::cout << "  Batch size: " << loader.GetBatchSize() << std::endl;
        std::cout << "  Number of batches: " << loader.NumBatches() << std::endl;
        std::cout << "  Shuffling: " << (loader.IsShuffling() ? "Yes" : "No") << std::endl;
        
        // Show first few batches
        for (size_t i = 0; i < std::min(3ul, loader.NumBatches()); ++i) {
            auto batch = loader.GetBatch(i);
            std::cout << "  Batch " << i << ": inputs shape=[" << batch.first.Shape()[0] 
                      << ", " << batch.first.Shape()[1] << "], targets shape=[" 
                      << batch.second.Shape()[0] << ", " << batch.second.Shape()[1] << "]" << std::endl;
        }
    }
    
    // 2. Range-based for loop
    std::cout << std::endl << "2. Range-based for loop iteration:" << std::endl;
    {
        data::DataLoader loader(dataset, 16, false, false, 42); // No shuffle for predictable output
        
        size_t batch_count = 0;
        for (auto [inputs, targets] : loader) {
            std::cout << "  Batch " << batch_count << ": " << inputs.Shape()[0] << " samples" << std::endl;
            batch_count++;
            if (batch_count >= 3) break; // Show only first 3 batches
        }
    }
    
    // 3. Drop last incomplete batch
    std::cout << std::endl << "3. Drop last incomplete batch:" << std::endl;
    {
        data::DataLoader loader_keep(dataset, 30, false, false, 42);
        data::DataLoader loader_drop(dataset, 30, false, true, 42);
        
        std::cout << "  Keep last: " << loader_keep.NumBatches() << " batches" << std::endl;
        std::cout << "  Drop last: " << loader_drop.NumBatches() << " batches" << std::endl;
        
        // Show last batch sizes
        if (loader_keep.NumBatches() > 0) {
            auto last_batch = loader_keep.GetBatch(loader_keep.NumBatches() - 1);
            std::cout << "  Last batch size (keep): " << last_batch.first.Shape()[0] << std::endl;
        }
    }
    
    std::cout << std::endl;
}

// Demonstrate train/validation splitting
void DemonstrateTrainValSplit() {
    std::cout << "=== Train/Validation Split ===" << std::endl;
    
    // Create dataset
    auto dataset = std::make_shared<data::SyntheticDataset>(
        data::SyntheticDataset::DataType::POLYNOMIAL_REGRESSION, 200, 1, 1, 0.1f, 42);
    
    data::DataLoader original_loader(dataset, 32, true, false, 42);
    
    // Split into train/val
    auto [train_loader, val_loader] = original_loader.TrainValSplit(0.8f, 42);
    
    std::cout << "Original dataset size: " << original_loader.GetDatasetSize() << std::endl;
    std::cout << "Train dataset size: " << train_loader->GetDatasetSize() << std::endl;
    std::cout << "Validation dataset size: " << val_loader->GetDatasetSize() << std::endl;
    std::cout << "Train batches: " << train_loader->NumBatches() << std::endl;
    std::cout << "Validation batches: " << val_loader->NumBatches() << std::endl;
    
    // Show that train shuffles but validation doesn't
    std::cout << "Train shuffling: " << (train_loader->IsShuffling() ? "Yes" : "No") << std::endl;
    std::cout << "Validation shuffling: " << (val_loader->IsShuffling() ? "Yes" : "No") << std::endl;
    
    std::cout << std::endl;
}

// Demonstrate utility functions
void DemonstrateUtilities() {
    std::cout << "=== Utility Functions ===" << std::endl;
    
    // 1. CreateTensorLoader
    std::cout << "1. CreateTensorLoader:" << std::endl;
    {
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> target_data = {2.0f, 4.0f, 6.0f, 8.0f};
        
        Tensor inputs(input_data, {4, 2});
        Tensor targets(target_data, {4});
        
        auto loader = data::utils::CreateTensorLoader(inputs, targets, 2, false, false, 42);
        
        std::cout << "  Created loader with " << loader->NumBatches() << " batches" << std::endl;
        
        for (size_t i = 0; i < loader->NumBatches(); ++i) {
            auto batch = loader->GetBatch(i);
            std::cout << "  Batch " << i << ": " << batch.first.Shape()[0] << " samples" << std::endl;
        }
    }
    
    // 2. CreateSyntheticLoader
    std::cout << std::endl << "2. CreateSyntheticLoader:" << std::endl;
    {
        auto loader = data::utils::CreateSyntheticLoader(
            data::SyntheticDataset::DataType::SINE_WAVE, 50, 1, 1, 0.05f, 10, true, 42);
        
        std::cout << "  Created synthetic loader: " << loader->GetDataset().GetName() << std::endl;
        std::cout << "  Batches: " << loader->NumBatches() << std::endl;
        
        // Show first batch
        auto batch = loader->GetBatch(0);
        std::cout << "  First batch: " << batch.first.Shape()[0] << " samples" << std::endl;
    }
    
    // 3. CreateTrainValLoaders
    std::cout << std::endl << "3. CreateTrainValLoaders:" << std::endl;
    {
        std::vector<float> input_data(200);
        std::vector<float> target_data(100);
        
        // Generate simple data
        for (size_t i = 0; i < 100; ++i) {
            input_data[i * 2] = static_cast<float>(i);
            input_data[i * 2 + 1] = static_cast<float>(i) * 0.5f;
            target_data[i] = static_cast<float>(i) * 2.0f;
        }
        
        Tensor inputs(input_data, {100, 2});
        Tensor targets(target_data, {100});
        
        auto [train_loader, val_loader] = data::utils::CreateTrainValLoaders(
            inputs, targets, 0.7f, 16, true, 42);
        
        std::cout << "  Train size: " << train_loader->GetDatasetSize() << std::endl;
        std::cout << "  Val size: " << val_loader->GetDatasetSize() << std::endl;
        std::cout << "  Train batches: " << train_loader->NumBatches() << std::endl;
        std::cout << "  Val batches: " << val_loader->NumBatches() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate different synthetic data types
void DemonstrateSyntheticDataTypes() {
    std::cout << "=== Synthetic Data Types ===" << std::endl;
    
    struct DataTypeDemo {
        data::SyntheticDataset::DataType type;
        std::string name;
        size_t inputDim;
        size_t outputDim;
        size_t samples;
    };
    
    std::vector<DataTypeDemo> demos = {
        {data::SyntheticDataset::DataType::LINEAR_REGRESSION, "Linear Regression", 1, 1, 5},
        {data::SyntheticDataset::DataType::POLYNOMIAL_REGRESSION, "Polynomial Regression", 1, 1, 5},
        {data::SyntheticDataset::DataType::SINE_WAVE, "Sine Wave", 1, 1, 5},
        {data::SyntheticDataset::DataType::BINARY_CLASSIFICATION, "Binary Classification", 2, 2, 5},
        {data::SyntheticDataset::DataType::MULTICLASS_CLASSIFICATION, "Multiclass Classification", 2, 3, 5},
        {data::SyntheticDataset::DataType::SPIRAL, "Spiral Classification", 2, 2, 5}
    };
    
    for (const auto& demo : demos) {
        std::cout << demo.name << ":" << std::endl;
        
        data::SyntheticDataset dataset(demo.type, demo.samples, demo.inputDim, 
                                      demo.outputDim, 0.1f, 42);
        
        for (size_t i = 0; i < std::min(3ul, demo.samples); ++i) {
            auto [input, target] = dataset.GetItem(i);
            
            std::cout << "  Sample " << i << ": input=[";
            for (size_t j = 0; j < input.Size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << input[j];
                if (j < input.Size() - 1) std::cout << ", ";
            }
            std::cout << "], target=[";
            for (size_t j = 0; j < target.Size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << target[j];
                if (j < target.Size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
    }
}

// Demonstrate performance and memory efficiency
void DemonstratePerformance() {
    std::cout << "=== Performance Demonstration ===" << std::endl;
    
    // Create large dataset
    size_t large_size = 10000;
    auto dataset = std::make_shared<data::SyntheticDataset>(
        data::SyntheticDataset::DataType::LINEAR_REGRESSION, large_size, 10, 1, 0.1f, 42);
    
    data::DataLoader loader(dataset, 64, true, false, 42);
    
    std::cout << "Large dataset: " << large_size << " samples, " 
              << loader.NumBatches() << " batches" << std::endl;
    
    // Time batch loading
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t total_samples = 0;
    for (auto [inputs, targets] : loader) {
        total_samples += inputs.Shape()[0];
        // Simulate some processing
        volatile float sum = 0.0f;
        for (size_t i = 0; i < std::min(10ul, inputs.Size()); ++i) {
            sum += inputs[i];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Processed " << total_samples << " samples in " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Average: " << static_cast<float>(duration.count()) / loader.NumBatches() 
              << " ms per batch" << std::endl;
    
    std::cout << std::endl;
}

// Demonstrate error handling
void DemonstrateErrorHandling() {
    std::cout << "=== Error Handling ===" << std::endl;
    
    // Test various error conditions
    std::vector<std::string> tests = {
        "Empty dataset",
        "Zero batch size", 
        "Invalid train ratio",
        "Out of bounds batch index",
        "Mismatched tensor shapes"
    };
    
    for (const auto& test : tests) {
        std::cout << "Testing: " << test << std::endl;
        
        try {
            if (test == "Empty dataset") {
                // This would require creating an empty dataset, which our current implementation prevents
                std::cout << "  (Prevented by design - datasets cannot be empty)" << std::endl;
                
            } else if (test == "Zero batch size") {
                auto dataset = std::make_shared<data::SyntheticDataset>(
                    data::SyntheticDataset::DataType::LINEAR_REGRESSION, 10, 1, 1, 0.1f, 42);
                data::DataLoader loader(dataset, 0); // This should throw
                
            } else if (test == "Invalid train ratio") {
                auto dataset = std::make_shared<data::SyntheticDataset>(
                    data::SyntheticDataset::DataType::LINEAR_REGRESSION, 10, 1, 1, 0.1f, 42);
                data::DataLoader loader(dataset, 4);
                loader.TrainValSplit(1.5f); // Invalid ratio
                
            } else if (test == "Out of bounds batch index") {
                auto dataset = std::make_shared<data::SyntheticDataset>(
                    data::SyntheticDataset::DataType::LINEAR_REGRESSION, 10, 1, 1, 0.1f, 42);
                data::DataLoader loader(dataset, 4);
                loader.GetBatch(100); // Out of bounds
                
            } else if (test == "Mismatched tensor shapes") {
                std::vector<float> inputs = {1.0f, 2.0f, 3.0f, 4.0f};
                std::vector<float> targets = {1.0f, 2.0f}; // Wrong size
                Tensor input_tensor(inputs, {4});
                Tensor target_tensor(targets, {2});
                data::TensorDataset dataset(input_tensor, target_tensor); // Should throw
            }
            
            std::cout << "  ERROR: Should have thrown an exception!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Correctly caught: " << e.what() << std::endl;
        }
        
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== KotML DataLoader Example ===" << std::endl << std::endl;
    
    try {
        DemonstrateDatasets();
        DemonstrateDataLoader();
        DemonstrateTrainValSplit();
        DemonstrateUtilities();
        DemonstrateSyntheticDataTypes();
        DemonstratePerformance();
        DemonstrateErrorHandling();
        
        std::cout << "=== DataLoader Example Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 