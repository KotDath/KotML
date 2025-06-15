#include "kotml/tensor.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <functional>

namespace kotml {

// Constructors
Tensor::Tensor() : m_requiresGrad(false) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool requiresGrad)
    : m_shape(shape), m_requiresGrad(requiresGrad) {
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
    if (shape.empty()) {
        total_size = 0;
    }
    m_data.resize(total_size, 0.0f);
    if (m_requiresGrad) {
        m_grad.resize(total_size, 0.0f);
    }
    computeStrides();
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requiresGrad)
    : m_data(data), m_shape(shape), m_requiresGrad(requiresGrad) {
    size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
    if (data.size() != expected_size) {
        throw std::invalid_argument("Data size doesn't match shape");
    }
    if (m_requiresGrad) {
        m_grad.resize(m_data.size(), 0.0f);
    }
    computeStrides();
}

Tensor::Tensor(std::initializer_list<float> data, const std::vector<size_t>& shape, bool requiresGrad)
    : m_data(data), m_shape(shape), m_requiresGrad(requiresGrad) {
    size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
    if (m_data.size() != expected_size) {
        throw std::invalid_argument("Data size doesn't match shape");
    }
    if (m_requiresGrad) {
        m_grad.resize(m_data.size(), 0.0f);
    }
    computeStrides();
}

// Copy and move
Tensor::Tensor(const Tensor& other)
    : m_data(other.m_data), m_shape(other.m_shape), m_strides(other.m_strides),
      m_grad(other.m_grad), m_requiresGrad(other.m_requiresGrad) {}

Tensor::Tensor(Tensor&& other) noexcept
    : m_data(std::move(other.m_data)), m_shape(std::move(other.m_shape)),
      m_strides(std::move(other.m_strides)), m_grad(std::move(other.m_grad)),
      m_requiresGrad(other.m_requiresGrad) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_grad = other.m_grad;
        m_requiresGrad = other.m_requiresGrad;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        m_data = std::move(other.m_data);
        m_shape = std::move(other.m_shape);
        m_strides = std::move(other.m_strides);
        m_grad = std::move(other.m_grad);
        m_requiresGrad = other.m_requiresGrad;
    }
    return *this;
}

// Helper methods
void Tensor::computeStrides() {
    m_strides.resize(m_shape.size());
    if (!m_shape.empty()) {
        m_strides.back() = 1;
        for (int i = static_cast<int>(m_shape.size()) - 2; i >= 0; --i) {
            m_strides[i] = m_strides[i + 1] * m_shape[i + 1];
        }
    }
}

size_t Tensor::computeIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != m_shape.size()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= m_shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * m_strides[i];
    }
    return index;
}

// Dimensions
size_t Tensor::Size() const {
    return m_data.size();
}

// Element access
float& Tensor::operator[](size_t index) {
    if (index >= m_data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return m_data[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= m_data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return m_data[index];
}

float& Tensor::At(const std::vector<size_t>& indices) {
    return m_data[computeIndex(indices)];
}

const float& Tensor::At(const std::vector<size_t>& indices) const {
    return m_data[computeIndex(indices)];
}

// Arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Tensor shapes don't match for addition");
    }
    
    bool resultRequiresGrad = m_requiresGrad || other.m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] + other.m_data[i];
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([](const Tensor& grad_output) -> std::vector<Tensor> {
            // For addition: grad_a = grad_output, grad_b = grad_output
            return {grad_output, grad_output};
        }, {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)});
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Tensor shapes don't match for subtraction");
    }
    
    bool resultRequiresGrad = m_requiresGrad || other.m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] - other.m_data[i];
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([](const Tensor& grad_output) -> std::vector<Tensor> {
            // For subtraction: grad_a = grad_output, grad_b = -grad_output
            Tensor neg_grad_output = grad_output * (-1.0f);
            return {grad_output, neg_grad_output};
        }, {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)});
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Tensor shapes don't match for multiplication");
    }
    
    bool resultRequiresGrad = m_requiresGrad || other.m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] * other.m_data[i];
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        // Capture values by copying them
        std::vector<float> this_data = m_data;
        std::vector<float> other_data = other.m_data;
        std::vector<size_t> shape = m_shape;
        
        result.SetGradFn([this_data, other_data, shape](const Tensor& grad_output) -> std::vector<Tensor> {
            // For multiplication: grad_a = grad_output * b, grad_b = grad_output * a
            Tensor grad_a(shape, false);
            Tensor grad_b(shape, false);
            
            for (size_t i = 0; i < grad_output.Size(); ++i) {
                grad_a.Data()[i] = grad_output.Data()[i] * other_data[i];
                grad_b.Data()[i] = grad_output.Data()[i] * this_data[i];
            }
            
            return {grad_a, grad_b};
        }, {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)});
    }
    
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (m_shape != other.m_shape) {
        throw std::invalid_argument("Tensor shapes don't match for division");
    }
    
    bool resultRequiresGrad = m_requiresGrad || other.m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        // IEEE 754 compliant division - returns inf/-inf for division by zero
        result.m_data[i] = m_data[i] / other.m_data[i];
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        // Capture values by copying them
        std::vector<float> this_data = m_data;
        std::vector<float> other_data = other.m_data;
        std::vector<size_t> shape = m_shape;
        
        result.SetGradFn([this_data, other_data, shape](const Tensor& grad_output) -> std::vector<Tensor> {
            // For division: grad_a = grad_output / b, grad_b = -grad_output * a / (b^2)
            Tensor grad_a(shape, false);
            Tensor grad_b(shape, false);
            
            for (size_t i = 0; i < grad_output.Size(); ++i) {
                grad_a.Data()[i] = grad_output.Data()[i] / other_data[i];
                grad_b.Data()[i] = -grad_output.Data()[i] * this_data[i] / (other_data[i] * other_data[i]);
            }
            
            return {grad_a, grad_b};
        }, {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)});
    }
    
    return result;
}

// Scalar operations
Tensor Tensor::operator+(float scalar) const {
    bool resultRequiresGrad = m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] + scalar;
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([](const Tensor& grad_output) -> std::vector<Tensor> {
            // For scalar addition: grad_tensor = grad_output
            return {grad_output};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    bool resultRequiresGrad = m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] - scalar;
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([](const Tensor& grad_output) -> std::vector<Tensor> {
            // For scalar subtraction: grad_tensor = grad_output
            return {grad_output};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    bool resultRequiresGrad = m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] = m_data[i] * scalar;
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([scalar](const Tensor& grad_output) -> std::vector<Tensor> {
            // For scalar multiplication: grad_tensor = grad_output * scalar
            return {grad_output * scalar};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

Tensor Tensor::operator/(float scalar) const {
    bool resultRequiresGrad = m_requiresGrad;
    Tensor result(m_shape, resultRequiresGrad);
    for (size_t i = 0; i < m_data.size(); ++i) {
        // IEEE 754 compliant division - returns inf/-inf for division by zero
        result.m_data[i] = m_data[i] / scalar;
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        result.SetGradFn([scalar](const Tensor& grad_output) -> std::vector<Tensor> {
            // For scalar division: grad_tensor = grad_output / scalar
            return {grad_output / scalar};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

// Assignment operations
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other;
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other;
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other;
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other;
    return *this;
}

// Linear algebra
Tensor Tensor::Matmul(const Tensor& other) const {
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    if (m_shape[1] != other.m_shape[0]) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    size_t m = m_shape[0];
    size_t n = m_shape[1];
    size_t p = other.m_shape[1];
    
    bool resultRequiresGrad = m_requiresGrad || other.m_requiresGrad;
    Tensor result({m, p}, resultRequiresGrad);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += m_data[i * n + k] * other.m_data[k * p + j];
            }
            result.m_data[i * p + j] = sum;
        }
    }
    
    // Set up automatic differentiation
    if (resultRequiresGrad) {
        // Capture values and shapes by copying them
        std::vector<float> this_data = m_data;
        std::vector<float> other_data = other.m_data;
        std::vector<size_t> this_shape = m_shape;
        std::vector<size_t> other_shape = other.m_shape;
        
        result.SetGradFn([this_data, other_data, this_shape, other_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            // For matrix multiplication C = A @ B:
            // grad_A = grad_output @ B^T
            // grad_B = A^T @ grad_output
            
            size_t m = this_shape[0];
            size_t n = this_shape[1];
            size_t p = other_shape[1];
            
            Tensor grad_A(this_shape, false);
            Tensor grad_B(other_shape, false);
            
            // grad_A = grad_output @ B^T
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < p; ++k) {
                        sum += grad_output.Data()[i * p + k] * other_data[j * p + k];
                    }
                    grad_A.Data()[i * n + j] = sum;
                }
            }
            
            // grad_B = A^T @ grad_output
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < p; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < m; ++k) {
                        sum += this_data[k * n + i] * grad_output.Data()[k * p + j];
                    }
                    grad_B.Data()[i * p + j] = sum;
                }
            }
            
            return {grad_A, grad_B};
        }, {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)});
    }
    
    return result;
}

Tensor Tensor::Reshape(const std::vector<size_t>& newShape) const {
    size_t new_size = std::accumulate(newShape.begin(), newShape.end(), 1UL, std::multiplies<size_t>());
    if (new_size != m_data.size()) {
        throw std::invalid_argument("New shape size doesn't match tensor size");
    }
    
    Tensor result(m_data, newShape, m_requiresGrad);
    if (m_requiresGrad) {
        result.m_grad = m_grad;
    }
    return result;
}

Tensor Tensor::Transpose() const {
    if (m_shape.size() != 2) {
        throw std::invalid_argument("Transpose currently only supports 2D tensors");
    }
    
    size_t rows = m_shape[0];
    size_t cols = m_shape[1];
    
    Tensor result({cols, rows}, m_requiresGrad);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.m_data[j * rows + i] = m_data[i * cols + j];
        }
    }
    
    return result;
}

// Reduction operations
Tensor Tensor::Sum() const {
    float total = std::accumulate(m_data.begin(), m_data.end(), 0.0f);
    Tensor result({1}, m_requiresGrad);
    result.m_data[0] = total;
    
    // Set up automatic differentiation
    if (m_requiresGrad) {
        std::vector<size_t> input_shape = m_shape;
        
        result.SetGradFn([input_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            // For sum: grad_input[i] = grad_output for all i
            Tensor grad_input(input_shape, false);
            float grad_value = grad_output.Data()[0];
            std::fill(grad_input.Data().begin(), grad_input.Data().end(), grad_value);
            return {grad_input};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

Tensor Tensor::Sum(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(m_shape.size())) {
        throw std::invalid_argument("Axis out of bounds");
    }
    
    std::vector<size_t> new_shape = m_shape;
    new_shape.erase(new_shape.begin() + axis);
    if (new_shape.empty()) {
        new_shape = {1};
    }
    
    Tensor result(new_shape, m_requiresGrad);
    std::fill(result.m_data.begin(), result.m_data.end(), 0.0f);
    
    // General implementation for any dimensionality
    size_t axis_size = m_shape[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    
    // Calculate outer size (product of dimensions before axis)
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
        outer_size *= m_shape[i];
    }
    
    // Calculate inner size (product of dimensions after axis)
    for (size_t i = axis + 1; i < m_shape.size(); ++i) {
        inner_size *= m_shape[i];
    }
    
    // Perform summation
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            float sum = 0.0f;
            for (size_t ax = 0; ax < axis_size; ++ax) {
                size_t src_idx = outer * axis_size * inner_size + ax * inner_size + inner;
                sum += m_data[src_idx];
            }
            size_t dst_idx = outer * inner_size + inner;
            result.m_data[dst_idx] = sum;
        }
    }
    
    return result;
}

Tensor Tensor::Mean() const {
    float total = std::accumulate(m_data.begin(), m_data.end(), 0.0f);
    Tensor result({1}, m_requiresGrad);
    result.m_data[0] = total / static_cast<float>(m_data.size());
    
    // Set up automatic differentiation
    if (m_requiresGrad) {
        size_t input_size = m_data.size();
        std::vector<size_t> input_shape = m_shape;
        
        result.SetGradFn([input_size, input_shape](const Tensor& grad_output) -> std::vector<Tensor> {
            // For mean: grad_input[i] = grad_output / input_size
            float grad_value = grad_output.Data()[0] / static_cast<float>(input_size);
            Tensor grad_input(input_shape, false);
            std::fill(grad_input.Data().begin(), grad_input.Data().end(), grad_value);
            return {grad_input};
        }, {const_cast<Tensor*>(this)});
    }
    
    return result;
}

Tensor Tensor::Mean(int axis) const {
    Tensor sum_result = Sum(axis);
    float divisor = static_cast<float>(m_shape[axis]);
    return sum_result / divisor;
}

// Automatic differentiation
void Tensor::Backward() {
    if (!m_requiresGrad) {
        throw std::runtime_error("Tensor doesn't require gradients");
    }
    
    // Initialize gradient for scalar tensor
    if (m_data.size() == 1 && m_grad[0] == 0.0f) {
        m_grad[0] = 1.0f;
    }
    
    // Use a static set to track processed tensors to avoid infinite recursion
    static thread_local std::unordered_set<const Tensor*> processing;
    
    // Check if we're already processing this tensor
    if (processing.count(this)) {
        return;
    }
    
    processing.insert(this);
    
    // Compute gradients for parent tensors
    if (m_gradFn) {
        Tensor grad_output(m_grad, m_shape, false);
        auto parent_grads = m_gradFn(grad_output);
        
        for (size_t i = 0; i < m_gradParents.size() && i < parent_grads.size(); ++i) {
            if (m_gradParents[i] && m_gradParents[i]->m_requiresGrad) {
                // Accumulate gradients
                for (size_t j = 0; j < m_gradParents[i]->m_grad.size(); ++j) {
                    m_gradParents[i]->m_grad[j] += parent_grads[i].m_data[j];
                }
                // Recursive backward call for parent tensors
                m_gradParents[i]->Backward();
            }
        }
    }
    
    processing.erase(this);
}

void Tensor::ZeroGrad() {
    std::fill(m_grad.begin(), m_grad.end(), 0.0f);
}

void Tensor::SetGradFn(const GradFunction& gradFn, 
                       const std::vector<Tensor*>& parents) {
    m_gradFn = gradFn;
    m_gradParents = parents;
}

// Utilities
void Tensor::Fill(float value) {
    std::fill(m_data.begin(), m_data.end(), value);
}

void Tensor::RandomNormal(float mean, float std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (auto& val : m_data) {
        val = dist(gen);
    }
}

void Tensor::RandomUniform(float min, float max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (auto& val : m_data) {
        val = dist(gen);
    }
}

// Output
void Tensor::Print() const {
    std::cout << ToString() << std::endl;
}

std::string Tensor::ToString() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < m_shape.size(); ++i) {
        oss << m_shape[i];
        if (i < m_shape.size() - 1) oss << ", ";
    }
    oss << "], data=[";
    
    size_t max_elements = std::min(m_data.size(), size_t(10));
    for (size_t i = 0; i < max_elements; ++i) {
        oss << m_data[i];
        if (i < max_elements - 1) oss << ", ";
    }
    if (m_data.size() > max_elements) {
        oss << "...";
    }
    oss << "])";
    
    return oss.str();
}

// Static factory methods
Tensor Tensor::Zeros(const std::vector<size_t>& shape, bool requiresGrad) {
    Tensor result(shape, requiresGrad);
    result.Fill(0.0f);
    return result;
}

Tensor Tensor::Ones(const std::vector<size_t>& shape, bool requiresGrad) {
    Tensor result(shape, requiresGrad);
    result.Fill(1.0f);
    return result;
}

Tensor Tensor::Eye(size_t n, bool requiresGrad) {
    Tensor result({n, n}, requiresGrad);
    result.Fill(0.0f);
    for (size_t i = 0; i < n; ++i) {
        result.m_data[i * n + i] = 1.0f;
    }
    return result;
}

Tensor Tensor::Randn(const std::vector<size_t>& shape, bool requiresGrad) {
    Tensor result(shape, requiresGrad);
    result.RandomNormal(0.0f, 1.0f);
    return result;
}

Tensor Tensor::Rand(const std::vector<size_t>& shape, bool requiresGrad) {
    Tensor result(shape, requiresGrad);
    result.RandomUniform(0.0f, 1.0f);
    return result;
}

// Operators for scalars on the left
Tensor operator+(float scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor result(tensor.Shape(), tensor.RequiresGrad());
    for (size_t i = 0; i < tensor.Size(); ++i) {
        result[i] = scalar - tensor[i];
    }
    return result;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor result(tensor.Shape(), tensor.RequiresGrad());
    for (size_t i = 0; i < tensor.Size(); ++i) {
        // IEEE 754 compliant division - returns inf/-inf for division by zero
        result[i] = scalar / tensor[i];
    }
    return result;
}

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << tensor.ToString();
    return os;
}

} // namespace kotml 