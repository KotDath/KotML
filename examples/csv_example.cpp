#include "kotml/kotml.hpp"
#include "kotml/tensor.hpp"
#include "kotml/data/data.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace kotml;

// Helper function to create sample CSV files for demonstration
void CreateSampleCSVFiles() {
    // 1. Simple regression dataset
    {
        std::ofstream file("sample_regression.csv");
        file << "x1,x2,x3,y\n";
        
        for (int i = 0; i < 100; ++i) {
            float x1 = static_cast<float>(i) / 10.0f;
            float x2 = std::sin(x1);
            float x3 = std::cos(x1);
            float y = 2.0f * x1 + 1.5f * x2 - 0.8f * x3 + 0.1f * (rand() % 100 - 50) / 50.0f;
            
            file << x1 << "," << x2 << "," << x3 << "," << y << "\n";
        }
    }
    
    // 2. Classification dataset
    {
        std::ofstream file("sample_classification.csv");
        file << "feature1,feature2,feature3,feature4,class\n";
        
        for (int i = 0; i < 200; ++i) {
            float f1 = static_cast<float>(rand() % 1000 - 500) / 100.0f;
            float f2 = static_cast<float>(rand() % 1000 - 500) / 100.0f;
            float f3 = static_cast<float>(rand() % 1000 - 500) / 100.0f;
            float f4 = static_cast<float>(rand() % 1000 - 500) / 100.0f;
            
            // Simple decision boundary: sum of features
            int class_id = (f1 + f2 + f3 + f4 > 0) ? 1 : 0;
            
            file << f1 << "," << f2 << "," << f3 << "," << f4 << "," << class_id << "\n";
        }
    }
    
    // 3. Dataset without headers
    {
        std::ofstream file("sample_no_header.csv");
        
        for (int i = 0; i < 50; ++i) {
            float x = static_cast<float>(i) / 5.0f;
            float y = x * x + 0.1f * (rand() % 100 - 50) / 50.0f;
            
            file << x << "," << y << "\n";
        }
    }
    
    // 4. Dataset with semicolon delimiter
    {
        std::ofstream file("sample_semicolon.csv");
        file << "input1;input2;target1;target2\n";
        
        for (int i = 0; i < 30; ++i) {
            float i1 = static_cast<float>(i) / 10.0f;
            float i2 = std::sin(i1);
            float t1 = i1 + i2;
            float t2 = i1 - i2;
            
            file << i1 << ";" << i2 << ";" << t1 << ";" << t2 << "\n";
        }
    }
    
    std::cout << "Sample CSV files created successfully!" << std::endl;
}

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

// Demonstrate basic CSV dataset usage
void DemonstrateBasicCSVUsage() {
    std::cout << "=== Basic CSV Dataset Usage ===" << std::endl;
    
    // 1. Automatic column detection
    std::cout << "1. Automatic column detection (regression dataset):" << std::endl;
    try {
        data::CSVDataset dataset("sample_regression.csv");
        dataset.PrintInfo();
        
        std::cout << "  Input column names: ";
        auto inputNames = dataset.GetInputColumnNames();
        for (size_t i = 0; i < inputNames.size(); ++i) {
            std::cout << inputNames[i];
            if (i < inputNames.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "  Target column names: ";
        auto targetNames = dataset.GetTargetColumnNames();
        for (size_t i = 0; i < targetNames.size(); ++i) {
            std::cout << targetNames[i];
            if (i < targetNames.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Show first few samples
        std::cout << "  First 3 samples:" << std::endl;
        for (size_t i = 0; i < std::min(3ul, dataset.Size()); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "    Sample " << i << ": input=[";
            for (size_t j = 0; j < input.Size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << input[j];
                if (j < input.Size() - 1) std::cout << ", ";
            }
            std::cout << "], target=" << target[0] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate manual column specification
void DemonstrateManualColumnSelection() {
    std::cout << "=== Manual Column Selection ===" << std::endl;
    
    std::cout << "1. Select specific columns for input and target:" << std::endl;
    try {
        // Use columns 0,2 as input and column 3 as target
        std::vector<size_t> inputCols = {0, 2};  // x1, x3
        std::vector<size_t> targetCols = {3};    // y
        
        data::CSVDataset dataset("sample_regression.csv", inputCols, targetCols);
        dataset.PrintInfo();
        
        // Show first few samples
        std::cout << "  First 3 samples:" << std::endl;
        for (size_t i = 0; i < std::min(3ul, dataset.Size()); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "    Sample " << i << ": input=[";
            for (size_t j = 0; j < input.Size(); ++j) {
                std::cout << std::fixed << std::setprecision(3) << input[j];
                if (j < input.Size() - 1) std::cout << ", ";
            }
            std::cout << "], target=" << target[0] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate different CSV formats
void DemonstrateDifferentFormats() {
    std::cout << "=== Different CSV Formats ===" << std::endl;
    
    // 1. No header
    std::cout << "1. CSV without headers:" << std::endl;
    try {
        data::CSVDataset dataset("sample_no_header.csv", false); // No header
        dataset.PrintInfo();
        
        // Show first few samples
        for (size_t i = 0; i < std::min(3ul, dataset.Size()); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "    Sample " << i << ": input=" << input[0] 
                      << ", target=" << target[0] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    // 2. Different delimiter
    std::cout << std::endl << "2. CSV with semicolon delimiter:" << std::endl;
    try {
        // Manual column specification: first 2 columns as input, last 2 as targets
        std::vector<size_t> inputCols = {0, 1};   // input1, input2
        std::vector<size_t> targetCols = {2, 3};  // target1, target2
        
        data::CSVDataset dataset("sample_semicolon.csv", inputCols, targetCols, true, ';');
        dataset.PrintInfo();
        
        // Show first few samples
        for (size_t i = 0; i < std::min(3ul, dataset.Size()); ++i) {
            auto [input, target] = dataset.GetItem(i);
            std::cout << "    Sample " << i << ": input=[" << input[0] << ", " << input[1] 
                      << "], target=[" << target[0] << ", " << target[1] << "]" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate CSV DataLoader
void DemonstrateCSVDataLoader() {
    std::cout << "=== CSV DataLoader ===" << std::endl;
    
    // 1. Basic CSV loader
    std::cout << "1. Basic CSV DataLoader:" << std::endl;
    try {
        auto loader = data::utils::CreateCSVLoader("sample_regression.csv", 16, true);
        
        std::cout << "  Dataset size: " << loader->GetDatasetSize() << std::endl;
        std::cout << "  Batch size: " << loader->GetBatchSize() << std::endl;
        std::cout << "  Number of batches: " << loader->NumBatches() << std::endl;
        
        // Show first batch
        auto batch = loader->GetBatch(0);
        std::cout << "  First batch shape: inputs=[" << batch.first.Shape()[0] 
                  << ", " << batch.first.Shape()[1] << "], targets=[" 
                  << batch.second.Shape()[0] << ", " << batch.second.Shape()[1] << "]" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    // 2. Manual column specification
    std::cout << std::endl << "2. CSV DataLoader with manual column selection:" << std::endl;
    try {
        std::vector<size_t> inputCols = {0, 1};  // x1, x2
        std::vector<size_t> targetCols = {3};    // y
        
        auto loader = data::utils::CreateCSVLoader("sample_regression.csv", 
                                                  inputCols, targetCols, 8, false);
        
        std::cout << "  Dataset size: " << loader->GetDatasetSize() << std::endl;
        std::cout << "  Input features: " << inputCols.size() << std::endl;
        std::cout << "  Target features: " << targetCols.size() << std::endl;
        
        // Iterate through first few batches
        size_t batchCount = 0;
        for (auto [inputs, targets] : *loader) {
            std::cout << "  Batch " << batchCount << ": " << inputs.Shape()[0] << " samples" << std::endl;
            batchCount++;
            if (batchCount >= 3) break;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate train/validation splitting with CSV
void DemonstrateCSVTrainValSplit() {
    std::cout << "=== CSV Train/Validation Split ===" << std::endl;
    
    try {
        auto [trainLoader, valLoader] = data::utils::CreateCSVTrainValLoaders(
            "sample_classification.csv", 0.8f, 32, true);
        
        std::cout << "Original dataset: sample_classification.csv" << std::endl;
        std::cout << "Train dataset size: " << trainLoader->GetDatasetSize() << std::endl;
        std::cout << "Validation dataset size: " << valLoader->GetDatasetSize() << std::endl;
        std::cout << "Train batches: " << trainLoader->NumBatches() << std::endl;
        std::cout << "Validation batches: " << valLoader->NumBatches() << std::endl;
        
        // Show sample from each
        auto trainBatch = trainLoader->GetBatch(0);
        auto valBatch = valLoader->GetBatch(0);
        
        std::cout << "Train batch shape: inputs=[" << trainBatch.first.Shape()[0] 
                  << ", " << trainBatch.first.Shape()[1] << "], targets=[" 
                  << trainBatch.second.Shape()[0] << ", " << trainBatch.second.Shape()[1] << "]" << std::endl;
        
        std::cout << "Val batch shape: inputs=[" << valBatch.first.Shape()[0] 
                  << ", " << valBatch.first.Shape()[1] << "], targets=[" 
                  << valBatch.second.Shape()[0] << ", " << valBatch.second.Shape()[1] << "]" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

// Demonstrate error handling
void DemonstrateErrorHandling() {
    std::cout << "=== Error Handling ===" << std::endl;
    
    std::vector<std::string> tests = {
        "Non-existent file",
        "Invalid column indices",
        "Empty input columns",
        "File with insufficient columns"
    };
    
    for (const auto& test : tests) {
        std::cout << "Testing: " << test << std::endl;
        
        try {
            if (test == "Non-existent file") {
                data::CSVDataset dataset("non_existent_file.csv");
                
            } else if (test == "Invalid column indices") {
                std::vector<size_t> inputCols = {0, 10};  // Column 10 doesn't exist
                std::vector<size_t> targetCols = {1};
                data::CSVDataset dataset("sample_regression.csv", inputCols, targetCols);
                
            } else if (test == "Empty input columns") {
                std::vector<size_t> inputCols = {};  // Empty
                std::vector<size_t> targetCols = {1};
                data::CSVDataset dataset("sample_regression.csv", inputCols, targetCols);
                
            } else if (test == "File with insufficient columns") {
                // Try to auto-detect on a file with only 1 column (need at least 2)
                std::ofstream tempFile("temp_single_column.csv");
                tempFile << "single_column\n1\n2\n3\n";
                tempFile.close();
                
                data::CSVDataset dataset("temp_single_column.csv");
            }
            
            std::cout << "  ERROR: Should have thrown an exception!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "  Correctly caught: " << e.what() << std::endl;
        }
        
        std::cout << std::endl;
    }
}

// Demonstrate performance with larger CSV file
void DemonstratePerformance() {
    std::cout << "=== Performance with Larger Dataset ===" << std::endl;
    
    // Create a larger CSV file
    std::cout << "Creating large CSV file (10,000 samples)..." << std::endl;
    {
        std::ofstream file("large_dataset.csv");
        file << "f1,f2,f3,f4,f5,target\n";
        
        for (int i = 0; i < 10000; ++i) {
            float f1 = static_cast<float>(i) / 1000.0f;
            float f2 = std::sin(f1);
            float f3 = std::cos(f1);
            float f4 = f1 * f1;
            float f5 = std::sqrt(std::abs(f1));
            float target = f1 + 0.5f * f2 - 0.3f * f3 + 0.1f * f4 + 0.2f * f5;
            
            file << f1 << "," << f2 << "," << f3 << "," << f4 << "," << f5 << "," << target << "\n";
        }
    }
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Load dataset
        data::CSVDataset dataset("large_dataset.csv");
        
        auto loadEnd = std::chrono::high_resolution_clock::now();
        auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - start);
        
        std::cout << "Dataset loaded in " << loadTime.count() << " ms" << std::endl;
        std::cout << "Dataset size: " << dataset.Size() << " samples" << std::endl;
        
        // Create DataLoader and iterate
        auto loader = data::utils::CreateCSVLoader("large_dataset.csv", 64, true);
        
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        size_t totalSamples = 0;
        for (auto [inputs, targets] : *loader) {
            totalSamples += inputs.Shape()[0];
            // Simulate some processing
            volatile float sum = 0.0f;
            for (size_t i = 0; i < std::min(10ul, inputs.Size()); ++i) {
                sum += inputs[i];
            }
        }
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        std::cout << "Processed " << totalSamples << " samples in " << iterTime.count() << " ms" << std::endl;
        std::cout << "Processing rate: " << static_cast<float>(totalSamples) / iterTime.count() * 1000 
                  << " samples/second" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "=== KotML CSV Dataset Example ===" << std::endl << std::endl;
    
    try {
        // Create sample files first
        CreateSampleCSVFiles();
        std::cout << std::endl;
        
        // Run demonstrations
        DemonstrateBasicCSVUsage();
        DemonstrateManualColumnSelection();
        DemonstrateDifferentFormats();
        DemonstrateCSVDataLoader();
        DemonstrateCSVTrainValSplit();
        DemonstratePerformance();
        DemonstrateErrorHandling();
        
        std::cout << "=== CSV Example Complete ===" << std::endl;
        
        // Clean up temporary files
        std::remove("temp_single_column.csv");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 