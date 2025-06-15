#pragma once

/**
 * KotML Data Module
 * 
 * This module provides comprehensive data handling capabilities for machine learning:
 * - Dataset classes for different data sources (in-memory, synthetic, CSV files)
 * - DataLoader for batch processing with shuffling
 * - Synthetic data generators for testing
 * - CSV file reading with automatic column detection
 * - Train/validation splitting utilities
 * 
 * Key Features:
 * - Memory-efficient batch loading
 * - Automatic shuffling and reproducible random seeds
 * - Support for various data types (regression, classification)
 * - CSV file parsing with header support and custom delimiters
 * - Easy integration with training loops
 * - Extensible architecture for custom datasets
 */

#include "kotml/data/dataset.hpp"
#include "kotml/data/dataloader.hpp"

namespace kotml {
namespace data {

// Re-export commonly used types for convenience
using TensorDataset = kotml::data::TensorDataset;
using SyntheticDataset = kotml::data::SyntheticDataset;
using CSVDataset = kotml::data::CSVDataset;
using DataLoader = kotml::data::DataLoader;
using SubsetDataset = kotml::data::SubsetDataset;

// Re-export utility functions
namespace utils {
    using kotml::data::utils::CreateTensorLoader;
    using kotml::data::utils::CreateSyntheticLoader;
    using kotml::data::utils::CreateTrainValLoaders;
    using kotml::data::utils::CreateCSVLoader;
    using kotml::data::utils::CreateCSVTrainValLoaders;
}

} // namespace data
} // namespace kotml 