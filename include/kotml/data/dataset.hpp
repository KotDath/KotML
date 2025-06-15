#pragma once

#include "kotml/tensor.hpp"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

namespace kotml {
namespace data {

/**
 * Base class for all datasets
 * Provides common interface for data access and manipulation
 */
class Dataset {
public:
    Dataset() = default;
    virtual ~Dataset() = default;
    
    /**
     * Get a single sample by index
     * @param index Sample index
     * @return Pair of (input, target) tensors
     */
    virtual std::pair<Tensor, Tensor> GetItem(size_t index) const = 0;
    
    /**
     * Get the total number of samples in the dataset
     * @return Number of samples
     */
    virtual size_t Size() const = 0;
    
    /**
     * Get dataset name/description
     */
    virtual std::string GetName() const = 0;
    
    /**
     * Get input shape for a single sample
     */
    virtual std::vector<size_t> GetInputShape() const = 0;
    
    /**
     * Get target shape for a single sample
     */
    virtual std::vector<size_t> GetTargetShape() const = 0;
    
    /**
     * Check if dataset is empty
     */
    bool Empty() const { return Size() == 0; }
    
    /**
     * Validate index bounds
     */
    void ValidateIndex(size_t index) const {
        if (index >= Size()) {
            throw std::out_of_range("Dataset index " + std::to_string(index) + 
                                  " out of range [0, " + std::to_string(Size()) + ")");
        }
    }
};

/**
 * In-memory dataset that stores all data in tensors
 * Suitable for small to medium datasets that fit in memory
 */
class TensorDataset : public Dataset {
private:
    Tensor m_inputs;
    Tensor m_targets;
    size_t m_numSamples;
    std::vector<size_t> m_inputShape;
    std::vector<size_t> m_targetShape;
    
public:
    /**
     * Create dataset from input and target tensors
     * @param inputs Input tensor (first dimension is batch size)
     * @param targets Target tensor (first dimension is batch size)
     */
    TensorDataset(const Tensor& inputs, const Tensor& targets) 
        : m_inputs(inputs), m_targets(targets) {
        
        if (inputs.Empty() || targets.Empty()) {
            throw std::invalid_argument("Input and target tensors cannot be empty");
        }
        
        if (inputs.Shape()[0] != targets.Shape()[0]) {
            throw std::invalid_argument("Input and target tensors must have the same batch size");
        }
        
        m_numSamples = inputs.Shape()[0];
        
        // Extract single sample shapes (remove batch dimension)
        m_inputShape = std::vector<size_t>(inputs.Shape().begin() + 1, inputs.Shape().end());
        m_targetShape = std::vector<size_t>(targets.Shape().begin() + 1, targets.Shape().end());
        
        // Handle 1D case (single values)
        if (m_inputShape.empty()) m_inputShape = {1};
        if (m_targetShape.empty()) m_targetShape = {1};
    }
    
    std::pair<Tensor, Tensor> GetItem(size_t index) const override {
        ValidateIndex(index);
        
        // Extract single sample from batch
        Tensor input_sample = ExtractSample(m_inputs, index);
        Tensor target_sample = ExtractSample(m_targets, index);
        
        return {input_sample, target_sample};
    }
    
    size_t Size() const override { return m_numSamples; }
    
    std::string GetName() const override { return "TensorDataset"; }
    
    std::vector<size_t> GetInputShape() const override { return m_inputShape; }
    
    std::vector<size_t> GetTargetShape() const override { return m_targetShape; }
    
    /**
     * Get all inputs as a single tensor
     */
    const Tensor& GetInputs() const { return m_inputs; }
    
    /**
     * Get all targets as a single tensor
     */
    const Tensor& GetTargets() const { return m_targets; }

private:
    /**
     * Extract a single sample from a batched tensor
     */
    Tensor ExtractSample(const Tensor& batched_tensor, size_t index) const {
        if (batched_tensor.Ndim() == 1) {
            // 1D case: return single value as 1D tensor
            std::vector<float> data = {batched_tensor[index]};
            return Tensor(data, {1});
        } else if (batched_tensor.Ndim() == 2) {
            // 2D case: extract row
            size_t cols = batched_tensor.Shape()[1];
            std::vector<float> sample_data(cols);
            
            for (size_t j = 0; j < cols; ++j) {
                sample_data[j] = batched_tensor.At({index, j});
            }
            
            return Tensor(sample_data, {cols});
        } else {
            throw std::runtime_error("ExtractSample: Unsupported tensor dimensionality");
        }
    }
};

/**
 * Synthetic dataset generator for testing and experimentation
 * Generates data on-the-fly using mathematical functions
 */
class SyntheticDataset : public Dataset {
public:
    enum class DataType {
        LINEAR_REGRESSION,      // y = ax + b + noise
        POLYNOMIAL_REGRESSION,  // y = ax² + bx + c + noise
        BINARY_CLASSIFICATION,  // Binary classification with 2D features
        MULTICLASS_CLASSIFICATION, // Multi-class classification
        SINE_WAVE,             // y = sin(x) + noise
        SPIRAL                 // 2D spiral classification
    };
    
private:
    DataType m_dataType;
    size_t m_numSamples;
    size_t m_inputDim;
    size_t m_outputDim;
    float m_noiseLevel;
    unsigned int m_seed;
    
    // Parameters for different data types
    std::vector<float> m_parameters;
    
public:
    /**
     * Create synthetic dataset
     * @param dataType Type of synthetic data to generate
     * @param numSamples Number of samples to generate
     * @param inputDim Input dimensionality
     * @param outputDim Output dimensionality (for classification: number of classes)
     * @param noiseLevel Amount of noise to add (0.0 = no noise)
     * @param seed Random seed for reproducibility
     */
    SyntheticDataset(DataType dataType, size_t numSamples, size_t inputDim = 1, 
                    size_t outputDim = 1, float noiseLevel = 0.1f, unsigned int seed = 42)
        : m_dataType(dataType), m_numSamples(numSamples), m_inputDim(inputDim),
          m_outputDim(outputDim), m_noiseLevel(noiseLevel), m_seed(seed) {
        
        if (numSamples == 0) {
            throw std::invalid_argument("Number of samples must be positive");
        }
        
        // Initialize parameters based on data type
        InitializeParameters();
    }
    
    std::pair<Tensor, Tensor> GetItem(size_t index) const override {
        ValidateIndex(index);
        return GenerateSample(index);
    }
    
    size_t Size() const override { return m_numSamples; }
    
    std::string GetName() const override { 
        switch (m_dataType) {
            case DataType::LINEAR_REGRESSION: return "LinearRegressionDataset";
            case DataType::POLYNOMIAL_REGRESSION: return "PolynomialRegressionDataset";
            case DataType::BINARY_CLASSIFICATION: return "BinaryClassificationDataset";
            case DataType::MULTICLASS_CLASSIFICATION: return "MulticlassClassificationDataset";
            case DataType::SINE_WAVE: return "SineWaveDataset";
            case DataType::SPIRAL: return "SpiralDataset";
            default: return "SyntheticDataset";
        }
    }
    
    std::vector<size_t> GetInputShape() const override { 
        return {m_inputDim}; 
    }
    
    std::vector<size_t> GetTargetShape() const override { 
        if (m_dataType == DataType::BINARY_CLASSIFICATION || 
            m_dataType == DataType::MULTICLASS_CLASSIFICATION ||
            m_dataType == DataType::SPIRAL) {
            return {m_outputDim}; // One-hot encoded
        }
        return {m_outputDim}; 
    }
    
    /**
     * Generate all samples at once and return as TensorDataset
     */
    std::unique_ptr<TensorDataset> GenerateAll() const {
        std::vector<float> all_inputs;
        std::vector<float> all_targets;
        
        for (size_t i = 0; i < m_numSamples; ++i) {
            auto [input, target] = GenerateSample(i);
            
            // Append input data
            for (size_t j = 0; j < input.Size(); ++j) {
                all_inputs.push_back(input[j]);
            }
            
            // Append target data
            for (size_t j = 0; j < target.Size(); ++j) {
                all_targets.push_back(target[j]);
            }
        }
        
        Tensor inputs(all_inputs, {m_numSamples, m_inputDim});
        Tensor targets(all_targets, {m_numSamples, m_outputDim});
        
        return std::make_unique<TensorDataset>(inputs, targets);
    }

private:
    void InitializeParameters() {
        switch (m_dataType) {
            case DataType::LINEAR_REGRESSION:
                m_parameters = {2.5f, 1.3f}; // slope, intercept
                break;
            case DataType::POLYNOMIAL_REGRESSION:
                m_parameters = {0.5f, -1.2f, 2.0f}; // a, b, c for ax² + bx + c
                break;
            case DataType::SINE_WAVE:
                m_parameters = {1.0f, 1.0f, 0.0f}; // amplitude, frequency, phase
                break;
            default:
                break;
        }
    }
    
    std::pair<Tensor, Tensor> GenerateSample(size_t index) const {
        // Use index as part of seed for deterministic but varied generation
        std::mt19937 gen(m_seed + index);
        std::normal_distribution<float> noise(0.0f, m_noiseLevel);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        
        std::vector<float> input_data(m_inputDim);
        std::vector<float> target_data(m_outputDim);
        
        switch (m_dataType) {
            case DataType::LINEAR_REGRESSION: {
                // Generate x in range [-2, 2]
                input_data[0] = uniform(gen) * 2.0f;
                float x = input_data[0];
                target_data[0] = m_parameters[0] * x + m_parameters[1] + noise(gen);
                break;
            }
            
            case DataType::POLYNOMIAL_REGRESSION: {
                input_data[0] = uniform(gen) * 2.0f;
                float x = input_data[0];
                target_data[0] = m_parameters[0] * x * x + m_parameters[1] * x + 
                               m_parameters[2] + noise(gen);
                break;
            }
            
            case DataType::SINE_WAVE: {
                input_data[0] = uniform(gen) * 4.0f * 3.14159f; // 0 to 4π
                float x = input_data[0];
                target_data[0] = m_parameters[0] * std::sin(m_parameters[1] * x + m_parameters[2]) + 
                               noise(gen);
                break;
            }
            
            case DataType::BINARY_CLASSIFICATION: {
                // Generate 2D points
                for (size_t i = 0; i < m_inputDim; ++i) {
                    input_data[i] = uniform(gen) * 4.0f - 2.0f; // [-2, 2]
                }
                
                // Simple decision boundary: x₁ + x₂ > 0
                float decision = 0.0f;
                for (size_t i = 0; i < m_inputDim; ++i) {
                    decision += input_data[i];
                }
                
                // One-hot encoding
                if (decision > 0) {
                    target_data[0] = 0.0f;
                    target_data[1] = 1.0f;
                } else {
                    target_data[0] = 1.0f;
                    target_data[1] = 0.0f;
                }
                break;
            }
            
            case DataType::MULTICLASS_CLASSIFICATION: {
                // Generate points and assign to classes based on distance from origin
                for (size_t i = 0; i < m_inputDim; ++i) {
                    input_data[i] = uniform(gen) * 4.0f - 2.0f;
                }
                
                float distance = 0.0f;
                for (size_t i = 0; i < m_inputDim; ++i) {
                    distance += input_data[i] * input_data[i];
                }
                distance = std::sqrt(distance);
                
                // Assign class based on distance
                size_t class_id = static_cast<size_t>(distance * m_outputDim / 3.0f);
                class_id = std::min(class_id, m_outputDim - 1);
                
                // One-hot encoding
                std::fill(target_data.begin(), target_data.end(), 0.0f);
                target_data[class_id] = 1.0f;
                break;
            }
            
            case DataType::SPIRAL: {
                // Generate 2D spiral data
                float t = static_cast<float>(index) / m_numSamples * 4.0f * 3.14159f;
                float r = t / (4.0f * 3.14159f);
                
                input_data[0] = r * std::cos(t) + noise(gen) * 0.1f;
                input_data[1] = r * std::sin(t) + noise(gen) * 0.1f;
                
                // Alternate between classes
                size_t class_id = (index / (m_numSamples / m_outputDim)) % m_outputDim;
                std::fill(target_data.begin(), target_data.end(), 0.0f);
                target_data[class_id] = 1.0f;
                break;
            }
        }
        
        return {Tensor(input_data, {m_inputDim}), Tensor(target_data, {m_outputDim})};
    }
};

/**
 * CSV Dataset for reading data from CSV files
 * Supports automatic header detection, column selection, and data type conversion
 */
class CSVDataset : public Dataset {
private:
    std::string m_filename;
    std::vector<std::vector<float>> m_data;
    std::vector<size_t> m_inputColumns;
    std::vector<size_t> m_targetColumns;
    std::vector<std::string> m_headers;
    size_t m_numSamples;
    bool m_hasHeader;
    char m_delimiter;
    
public:
    /**
     * Create CSV dataset
     * @param filename Path to CSV file
     * @param inputColumns Indices of columns to use as input features (0-based)
     * @param targetColumns Indices of columns to use as targets (0-based)
     * @param hasHeader Whether the first row contains column headers
     * @param delimiter CSV delimiter character (default: ',')
     * @param skipRows Number of rows to skip after header (default: 0)
     */
    CSVDataset(const std::string& filename, 
               const std::vector<size_t>& inputColumns,
               const std::vector<size_t>& targetColumns,
               bool hasHeader = true,
               char delimiter = ',',
               size_t skipRows = 0) 
        : m_filename(filename), m_inputColumns(inputColumns), 
          m_targetColumns(targetColumns), m_hasHeader(hasHeader), 
          m_delimiter(delimiter) {
        
        if (inputColumns.empty()) {
            throw std::invalid_argument("Input columns cannot be empty");
        }
        
        if (targetColumns.empty()) {
            throw std::invalid_argument("Target columns cannot be empty");
        }
        
        LoadCSV(skipRows);
    }
    
    /**
     * Create CSV dataset with automatic column detection
     * Uses all columns except the last one as input, last column as target
     * @param filename Path to CSV file
     * @param hasHeader Whether the first row contains column headers
     * @param delimiter CSV delimiter character
     * @param skipRows Number of rows to skip after header
     */
    CSVDataset(const std::string& filename,
               bool hasHeader = true,
               char delimiter = ',',
               size_t skipRows = 0)
        : m_filename(filename), m_hasHeader(hasHeader), m_delimiter(delimiter) {
        
        LoadCSVAutoDetect(skipRows);
    }
    
    std::pair<Tensor, Tensor> GetItem(size_t index) const override {
        ValidateIndex(index);
        
        std::vector<float> inputData(m_inputColumns.size());
        std::vector<float> targetData(m_targetColumns.size());
        
        // Extract input features
        for (size_t i = 0; i < m_inputColumns.size(); ++i) {
            inputData[i] = m_data[index][m_inputColumns[i]];
        }
        
        // Extract targets
        for (size_t i = 0; i < m_targetColumns.size(); ++i) {
            targetData[i] = m_data[index][m_targetColumns[i]];
        }
        
        return {Tensor(inputData, {m_inputColumns.size()}), 
                Tensor(targetData, {m_targetColumns.size()})};
    }
    
    size_t Size() const override { return m_numSamples; }
    
    std::string GetName() const override { 
        return "CSVDataset(" + m_filename + ")"; 
    }
    
    std::vector<size_t> GetInputShape() const override { 
        return {m_inputColumns.size()}; 
    }
    
    std::vector<size_t> GetTargetShape() const override { 
        return {m_targetColumns.size()}; 
    }
    
    /**
     * Get column headers (if available)
     */
    const std::vector<std::string>& GetHeaders() const { return m_headers; }
    
    /**
     * Get input column names
     */
    std::vector<std::string> GetInputColumnNames() const {
        std::vector<std::string> names;
        if (!m_headers.empty()) {
            for (size_t col : m_inputColumns) {
                if (col < m_headers.size()) {
                    names.push_back(m_headers[col]);
                } else {
                    names.push_back("Column_" + std::to_string(col));
                }
            }
        } else {
            for (size_t col : m_inputColumns) {
                names.push_back("Column_" + std::to_string(col));
            }
        }
        return names;
    }
    
    /**
     * Get target column names
     */
    std::vector<std::string> GetTargetColumnNames() const {
        std::vector<std::string> names;
        if (!m_headers.empty()) {
            for (size_t col : m_targetColumns) {
                if (col < m_headers.size()) {
                    names.push_back(m_headers[col]);
                } else {
                    names.push_back("Column_" + std::to_string(col));
                }
            }
        } else {
            for (size_t col : m_targetColumns) {
                names.push_back("Column_" + std::to_string(col));
            }
        }
        return names;
    }
    
    /**
     * Get basic statistics about the dataset
     */
    void PrintInfo() const {
        std::cout << "CSV Dataset Information:" << std::endl;
        std::cout << "  File: " << m_filename << std::endl;
        std::cout << "  Samples: " << m_numSamples << std::endl;
        std::cout << "  Input features: " << m_inputColumns.size() << std::endl;
        std::cout << "  Target features: " << m_targetColumns.size() << std::endl;
        std::cout << "  Has header: " << (m_hasHeader ? "Yes" : "No") << std::endl;
        std::cout << "  Delimiter: '" << m_delimiter << "'" << std::endl;
        
        if (!m_headers.empty()) {
            std::cout << "  Headers: ";
            for (size_t i = 0; i < m_headers.size(); ++i) {
                std::cout << m_headers[i];
                if (i < m_headers.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "  Input columns: ";
        for (size_t i = 0; i < m_inputColumns.size(); ++i) {
            std::cout << m_inputColumns[i];
            if (i < m_inputColumns.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        std::cout << "  Target columns: ";
        for (size_t i = 0; i < m_targetColumns.size(); ++i) {
            std::cout << m_targetColumns[i];
            if (i < m_targetColumns.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }

private:
    void LoadCSV(size_t skipRows) {
        std::ifstream file(m_filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + m_filename);
        }
        
        std::string line;
        size_t lineNumber = 0;
        
        // Read header if present
        if (m_hasHeader && std::getline(file, line)) {
            m_headers = ParseCSVLine(line);
            lineNumber++;
        }
        
        // Skip additional rows if requested
        for (size_t i = 0; i < skipRows && std::getline(file, line); ++i) {
            lineNumber++;
        }
        
        // Read data
        m_data.clear();
        while (std::getline(file, line)) {
            if (line.empty()) continue; // Skip empty lines
            
            auto values = ParseCSVLine(line);
            if (values.empty()) continue;
            
            // Convert to float and validate columns
            std::vector<float> row;
            try {
                for (const auto& value : values) {
                    row.push_back(ParseFloat(value));
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing line " + std::to_string(lineNumber + 1) + 
                                       ": " + std::string(e.what()));
            }
            
            // Validate column indices
            ValidateColumnIndices(row.size(), lineNumber + 1);
            
            m_data.push_back(row);
            lineNumber++;
        }
        
        if (m_data.empty()) {
            throw std::runtime_error("No valid data found in CSV file: " + m_filename);
        }
        
        m_numSamples = m_data.size();
    }
    
    void LoadCSVAutoDetect(size_t skipRows) {
        std::ifstream file(m_filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + m_filename);
        }
        
        std::string line;
        
        // Read header if present
        if (m_hasHeader && std::getline(file, line)) {
            m_headers = ParseCSVLine(line);
        }
        
        // Skip additional rows
        for (size_t i = 0; i < skipRows && std::getline(file, line); ++i) {
            // Skip
        }
        
        // Read first data line to determine number of columns
        if (!std::getline(file, line)) {
            throw std::runtime_error("No data found in CSV file: " + m_filename);
        }
        
        auto firstRow = ParseCSVLine(line);
        size_t numColumns = firstRow.size();
        
        if (numColumns < 2) {
            throw std::runtime_error("CSV file must have at least 2 columns (input and target)");
        }
        
        // Auto-detect: all columns except last are inputs, last column is target
        m_inputColumns.clear();
        m_targetColumns.clear();
        
        for (size_t i = 0; i < numColumns - 1; ++i) {
            m_inputColumns.push_back(i);
        }
        m_targetColumns.push_back(numColumns - 1);
        
        // Reset file and load normally
        file.clear();
        file.seekg(0);
        LoadCSV(skipRows);
    }
    
    std::vector<std::string> ParseCSVLine(const std::string& line) const {
        std::vector<std::string> result;
        std::stringstream ss(line);
        std::string cell;
        
        bool inQuotes = false;
        std::string currentCell;
        
        for (char c : line) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == m_delimiter && !inQuotes) {
                result.push_back(Trim(currentCell));
                currentCell.clear();
            } else {
                currentCell += c;
            }
        }
        
        // Add the last cell
        result.push_back(Trim(currentCell));
        
        return result;
    }
    
    float ParseFloat(const std::string& str) const {
        if (str.empty()) {
            return 0.0f; // Default value for empty cells
        }
        
        try {
            return std::stof(str);
        } catch (const std::exception&) {
            throw std::runtime_error("Cannot convert '" + str + "' to float");
        }
    }
    
    std::string Trim(const std::string& str) const {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        
        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }
    
    void ValidateColumnIndices(size_t numColumns, size_t lineNumber) const {
        for (size_t col : m_inputColumns) {
            if (col >= numColumns) {
                throw std::runtime_error("Input column index " + std::to_string(col) + 
                                       " out of range at line " + std::to_string(lineNumber) +
                                       " (available columns: 0-" + std::to_string(numColumns - 1) + ")");
            }
        }
        
        for (size_t col : m_targetColumns) {
            if (col >= numColumns) {
                throw std::runtime_error("Target column index " + std::to_string(col) + 
                                       " out of range at line " + std::to_string(lineNumber) +
                                       " (available columns: 0-" + std::to_string(numColumns - 1) + ")");
            }
        }
    }
};

} // namespace data
} // namespace kotml 