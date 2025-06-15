#pragma once

#include "kotml/data/dataset.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace kotml {
namespace data {

/**
 * Subset dataset that wraps another dataset with specific indices
 * Used internally by DataLoader for train/val splits
 */
class SubsetDataset : public Dataset {
private:
    std::shared_ptr<Dataset> m_baseDataset;
    std::vector<size_t> m_indices;
    
public:
    SubsetDataset(std::shared_ptr<Dataset> baseDataset, const std::vector<size_t>& indices)
        : m_baseDataset(baseDataset), m_indices(indices) {
        
        if (!baseDataset) {
            throw std::invalid_argument("Base dataset cannot be null");
        }
        
        if (indices.empty()) {
            throw std::invalid_argument("Indices cannot be empty");
        }
        
        // Validate indices
        for (size_t idx : indices) {
            if (idx >= baseDataset->Size()) {
                throw std::out_of_range("Index " + std::to_string(idx) + 
                                      " out of range for base dataset");
            }
        }
    }
    
    std::pair<Tensor, Tensor> GetItem(size_t index) const override {
        ValidateIndex(index);
        return m_baseDataset->GetItem(m_indices[index]);
    }
    
    size_t Size() const override { return m_indices.size(); }
    
    std::string GetName() const override { 
        return "SubsetOf" + m_baseDataset->GetName(); 
    }
    
    std::vector<size_t> GetInputShape() const override { 
        return m_baseDataset->GetInputShape(); 
    }
    
    std::vector<size_t> GetTargetShape() const override { 
        return m_baseDataset->GetTargetShape(); 
    }
    
    const std::vector<size_t>& GetIndices() const { return m_indices; }
    
    std::shared_ptr<Dataset> GetBaseDataset() const { return m_baseDataset; }
};

/**
 * DataLoader for batch loading of datasets
 * Supports shuffling, different batch sizes, and drop_last option
 */
class DataLoader {
private:
    std::shared_ptr<Dataset> m_dataset;
    size_t m_batchSize;
    bool m_shuffle;
    bool m_dropLast;
    unsigned int m_seed;
    
    // Internal state
    mutable std::vector<size_t> m_indices;
    mutable std::mt19937 m_generator;
    mutable bool m_needsReshuffle;
    
public:
    /**
     * Create DataLoader
     * @param dataset Dataset to load from
     * @param batchSize Size of each batch
     * @param shuffle Whether to shuffle data each epoch
     * @param dropLast Whether to drop the last incomplete batch
     * @param seed Random seed for shuffling
     */
    DataLoader(std::shared_ptr<Dataset> dataset, size_t batchSize = 32, 
               bool shuffle = true, bool dropLast = false, unsigned int seed = 42)
        : m_dataset(dataset), m_batchSize(batchSize), m_shuffle(shuffle), 
          m_dropLast(dropLast), m_seed(seed), m_generator(seed), m_needsReshuffle(true) {
        
        if (!dataset) {
            throw std::invalid_argument("Dataset cannot be null");
        }
        
        if (batchSize == 0) {
            throw std::invalid_argument("Batch size must be positive");
        }
        
        if (dataset->Empty()) {
            throw std::invalid_argument("Dataset cannot be empty");
        }
        
        // Initialize indices
        m_indices.resize(m_dataset->Size());
        std::iota(m_indices.begin(), m_indices.end(), 0);
    }
    
    /**
     * Get number of batches per epoch
     */
    size_t NumBatches() const {
        size_t datasetSize = m_dataset->Size();
        if (m_dropLast) {
            return datasetSize / m_batchSize;
        } else {
            return (datasetSize + m_batchSize - 1) / m_batchSize; // Ceiling division
        }
    }
    
    /**
     * Get batch size
     */
    size_t GetBatchSize() const { return m_batchSize; }
    
    /**
     * Get dataset size
     */
    size_t GetDatasetSize() const { return m_dataset->Size(); }
    
    /**
     * Check if shuffling is enabled
     */
    bool IsShuffling() const { return m_shuffle; }
    
    /**
     * Check if drop_last is enabled
     */
    bool IsDropLast() const { return m_dropLast; }
    
    /**
     * Get dataset reference
     */
    const Dataset& GetDataset() const { return *m_dataset; }
    
    /**
     * Set new batch size
     */
    void SetBatchSize(size_t batchSize) {
        if (batchSize == 0) {
            throw std::invalid_argument("Batch size must be positive");
        }
        m_batchSize = batchSize;
    }
    
    /**
     * Enable/disable shuffling
     */
    void SetShuffle(bool shuffle) {
        m_shuffle = shuffle;
        if (shuffle) {
            m_needsReshuffle = true;
        }
    }
    
    /**
     * Set random seed
     */
    void SetSeed(unsigned int seed) {
        m_seed = seed;
        m_generator.seed(seed);
        m_needsReshuffle = true;
    }
    
    /**
     * Get a specific batch by index
     * @param batchIndex Index of the batch (0 to NumBatches()-1)
     * @return Pair of (batch_inputs, batch_targets) tensors
     */
    std::pair<Tensor, Tensor> GetBatch(size_t batchIndex) const {
        if (batchIndex >= NumBatches()) {
            throw std::out_of_range("Batch index " + std::to_string(batchIndex) + 
                                  " out of range [0, " + std::to_string(NumBatches()) + ")");
        }
        
        // Shuffle if needed
        if (m_shuffle && m_needsReshuffle) {
            std::shuffle(m_indices.begin(), m_indices.end(), m_generator);
            m_needsReshuffle = false;
        }
        
        // Calculate batch boundaries
        size_t startIdx = batchIndex * m_batchSize;
        size_t endIdx = std::min(startIdx + m_batchSize, m_dataset->Size());
        size_t actualBatchSize = endIdx - startIdx;
        
        // Get first sample to determine shapes
        auto [firstInput, firstTarget] = m_dataset->GetItem(m_indices[startIdx]);
        
        // Calculate batch tensor shapes
        std::vector<size_t> inputBatchShape = {actualBatchSize};
        std::vector<size_t> targetBatchShape = {actualBatchSize};
        
        for (size_t dim : firstInput.Shape()) {
            inputBatchShape.push_back(dim);
        }
        for (size_t dim : firstTarget.Shape()) {
            targetBatchShape.push_back(dim);
        }
        
        // Create batch tensors
        Tensor batchInputs = Tensor::Zeros(inputBatchShape);
        Tensor batchTargets = Tensor::Zeros(targetBatchShape);
        
        // Fill batch tensors
        for (size_t i = 0; i < actualBatchSize; ++i) {
            auto [input, target] = m_dataset->GetItem(m_indices[startIdx + i]);
            
            // Copy input data
            if (firstInput.Ndim() == 1) {
                // 1D input: copy to batch[i, :]
                for (size_t j = 0; j < input.Size(); ++j) {
                    batchInputs.At({i, j}) = input[j];
                }
            } else {
                // Handle higher dimensions if needed
                throw std::runtime_error("GetBatch: Unsupported input tensor dimensionality");
            }
            
            // Copy target data
            if (firstTarget.Ndim() == 1) {
                // 1D target: copy to batch[i, :]
                for (size_t j = 0; j < target.Size(); ++j) {
                    batchTargets.At({i, j}) = target[j];
                }
            } else {
                // Handle higher dimensions if needed
                throw std::runtime_error("GetBatch: Unsupported target tensor dimensionality");
            }
        }
        
        return {batchInputs, batchTargets};
    }
    
    /**
     * Iterator class for range-based for loops
     */
    class Iterator {
    private:
        const DataLoader* m_loader;
        size_t m_currentBatch;
        
    public:
        Iterator(const DataLoader* loader, size_t batchIndex) 
            : m_loader(loader), m_currentBatch(batchIndex) {}
        
        std::pair<Tensor, Tensor> operator*() const {
            return m_loader->GetBatch(m_currentBatch);
        }
        
        Iterator& operator++() {
            ++m_currentBatch;
            return *this;
        }
        
        bool operator!=(const Iterator& other) const {
            return m_currentBatch != other.m_currentBatch;
        }
        
        bool operator==(const Iterator& other) const {
            return m_currentBatch == other.m_currentBatch;
        }
    };
    
    /**
     * Begin iterator for range-based for loops
     */
    Iterator begin() const {
        // Trigger reshuffle for new epoch
        if (m_shuffle) {
            m_needsReshuffle = true;
        }
        return Iterator(this, 0);
    }
    
    /**
     * End iterator for range-based for loops
     */
    Iterator end() const {
        return Iterator(this, NumBatches());
    }
    
    /**
     * Reset the dataloader (force reshuffle on next iteration)
     */
    void Reset() const {
        m_needsReshuffle = true;
    }
    
    /**
     * Get all data as a single batch (useful for small datasets)
     */
    std::pair<Tensor, Tensor> GetAllData() const {
        // Temporarily set batch size to dataset size
        size_t originalBatchSize = m_batchSize;
        const_cast<DataLoader*>(this)->m_batchSize = m_dataset->Size();
        
        auto result = GetBatch(0);
        
        // Restore original batch size
        const_cast<DataLoader*>(this)->m_batchSize = originalBatchSize;
        
        return result;
    }
    
    /**
     * Split dataset into train/validation sets
     * @param trainRatio Ratio of data to use for training (0.0 to 1.0)
     * @param seed Random seed for splitting
     * @return Pair of (train_loader, val_loader)
     */
    std::pair<std::unique_ptr<DataLoader>, std::unique_ptr<DataLoader>> 
    TrainValSplit(float trainRatio = 0.8f, unsigned int seed = 42) const {
        
        if (trainRatio <= 0.0f || trainRatio >= 1.0f) {
            throw std::invalid_argument("Train ratio must be between 0 and 1");
        }
        
        size_t datasetSize = m_dataset->Size();
        size_t trainSize = static_cast<size_t>(datasetSize * trainRatio);
        size_t valSize = datasetSize - trainSize;
        
        // Create shuffled indices for splitting
        std::vector<size_t> splitIndices(datasetSize);
        std::iota(splitIndices.begin(), splitIndices.end(), 0);
        
        std::mt19937 splitGen(seed);
        std::shuffle(splitIndices.begin(), splitIndices.end(), splitGen);
        
        // Create subset datasets
        auto trainDataset = std::make_shared<SubsetDataset>(m_dataset, 
            std::vector<size_t>(splitIndices.begin(), splitIndices.begin() + trainSize));
        auto valDataset = std::make_shared<SubsetDataset>(m_dataset,
            std::vector<size_t>(splitIndices.begin() + trainSize, splitIndices.end()));
        
        // Create loaders
        auto trainLoader = std::make_unique<DataLoader>(trainDataset, m_batchSize, 
                                                       m_shuffle, m_dropLast, m_seed);
        auto valLoader = std::make_unique<DataLoader>(valDataset, m_batchSize, 
                                                     false, m_dropLast, m_seed); // No shuffle for validation
        
        return {std::move(trainLoader), std::move(valLoader)};
    }
};

/**
 * Utility functions for creating common data loaders
 */
namespace utils {

/**
 * Create a DataLoader from tensors
 */
inline std::unique_ptr<DataLoader> CreateTensorLoader(const Tensor& inputs, const Tensor& targets,
                                                     size_t batchSize = 32, bool shuffle = true,
                                                     bool dropLast = false, unsigned int seed = 42) {
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    return std::make_unique<DataLoader>(dataset, batchSize, shuffle, dropLast, seed);
}

/**
 * Create a DataLoader from synthetic data
 */
inline std::unique_ptr<DataLoader> CreateSyntheticLoader(SyntheticDataset::DataType dataType,
                                                        size_t numSamples, size_t inputDim = 1,
                                                        size_t outputDim = 1, float noiseLevel = 0.1f,
                                                        size_t batchSize = 32, bool shuffle = true,
                                                        unsigned int seed = 42) {
    auto dataset = std::make_shared<SyntheticDataset>(dataType, numSamples, inputDim, 
                                                     outputDim, noiseLevel, seed);
    return std::make_unique<DataLoader>(dataset, batchSize, shuffle, false, seed);
}

/**
 * Create train/validation loaders from tensors
 */
inline std::pair<std::unique_ptr<DataLoader>, std::unique_ptr<DataLoader>>
CreateTrainValLoaders(const Tensor& inputs, const Tensor& targets, float trainRatio = 0.8f,
                     size_t batchSize = 32, bool shuffle = true, unsigned int seed = 42) {
    auto loader = CreateTensorLoader(inputs, targets, batchSize, shuffle, false, seed);
    return loader->TrainValSplit(trainRatio, seed);
}

/**
 * Create a DataLoader from CSV file with automatic column detection
 */
inline std::unique_ptr<DataLoader> CreateCSVLoader(const std::string& filename,
                                                  size_t batchSize = 32, bool shuffle = true,
                                                  bool hasHeader = true, char delimiter = ',',
                                                  size_t skipRows = 0, unsigned int seed = 42) {
    auto dataset = std::make_shared<CSVDataset>(filename, hasHeader, delimiter, skipRows);
    return std::make_unique<DataLoader>(dataset, batchSize, shuffle, false, seed);
}

/**
 * Create a DataLoader from CSV file with manual column specification
 */
inline std::unique_ptr<DataLoader> CreateCSVLoader(const std::string& filename,
                                                  const std::vector<size_t>& inputColumns,
                                                  const std::vector<size_t>& targetColumns,
                                                  size_t batchSize = 32, bool shuffle = true,
                                                  bool hasHeader = true, char delimiter = ',',
                                                  size_t skipRows = 0, unsigned int seed = 42) {
    auto dataset = std::make_shared<CSVDataset>(filename, inputColumns, targetColumns, 
                                               hasHeader, delimiter, skipRows);
    return std::make_unique<DataLoader>(dataset, batchSize, shuffle, false, seed);
}

/**
 * Create train/validation loaders from CSV file
 */
inline std::pair<std::unique_ptr<DataLoader>, std::unique_ptr<DataLoader>>
CreateCSVTrainValLoaders(const std::string& filename, float trainRatio = 0.8f,
                        size_t batchSize = 32, bool shuffle = true,
                        bool hasHeader = true, char delimiter = ',',
                        size_t skipRows = 0, unsigned int seed = 42) {
    auto loader = CreateCSVLoader(filename, batchSize, shuffle, hasHeader, delimiter, skipRows, seed);
    return loader->TrainValSplit(trainRatio, seed);
}

/**
 * Create train/validation loaders from CSV file with manual column specification
 */
inline std::pair<std::unique_ptr<DataLoader>, std::unique_ptr<DataLoader>>
CreateCSVTrainValLoaders(const std::string& filename,
                        const std::vector<size_t>& inputColumns,
                        const std::vector<size_t>& targetColumns,
                        float trainRatio = 0.8f, size_t batchSize = 32, bool shuffle = true,
                        bool hasHeader = true, char delimiter = ',',
                        size_t skipRows = 0, unsigned int seed = 42) {
    auto loader = CreateCSVLoader(filename, inputColumns, targetColumns, batchSize, shuffle, 
                                 hasHeader, delimiter, skipRows, seed);
    return loader->TrainValSplit(trainRatio, seed);
}

} // namespace utils

} // namespace data
} // namespace kotml 