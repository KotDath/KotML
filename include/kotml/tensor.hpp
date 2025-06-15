#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <initializer_list>
#include <iostream>

namespace kotml {

// Forward declaration for gradient functions
class Tensor;
using GradFunction = std::function<std::vector<Tensor>(const Tensor&)>;

class Tensor {
private:
    std::vector<float> m_data;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    
    // For automatic differentiation
    std::vector<float> m_grad;
    bool m_requiresGrad;
    std::vector<Tensor*> m_gradParents;
    GradFunction m_gradFn;
    
    void computeStrides();
    size_t computeIndex(const std::vector<size_t>& indices) const;
    
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape, bool requiresGrad = false);
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requiresGrad = false);
    Tensor(std::initializer_list<float> data, const std::vector<size_t>& shape, bool requiresGrad = false);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor() = default;
    
    // Data access
    const std::vector<float>& Data() const { return m_data; }
    std::vector<float>& Data() { return m_data; }
    const std::vector<size_t>& Shape() const { return m_shape; }
    const std::vector<size_t>& Strides() const { return m_strides; }
    
    // Dimensions
    size_t Size() const;
    size_t Ndim() const { return m_shape.size(); }
    bool Empty() const { return m_data.empty(); }
    
    // Element access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& At(const std::vector<size_t>& indices);
    const float& At(const std::vector<size_t>& indices) const;
    
    // Arithmetic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    // Scalar operations
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    
    // Assignment operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // Linear algebra
    Tensor Matmul(const Tensor& other) const;
    Tensor Reshape(const std::vector<size_t>& newShape) const;
    Tensor Transpose() const;
    
    // Reduction operations
    Tensor Sum() const;
    Tensor Sum(int axis) const;
    Tensor Mean() const;
    Tensor Mean(int axis) const;
    
    // Automatic differentiation
    bool RequiresGrad() const { return m_requiresGrad; }
    void SetRequiresGrad(bool requiresGrad) { 
        m_requiresGrad = requiresGrad; 
        if (requiresGrad && m_grad.size() != m_data.size()) {
            m_grad.resize(m_data.size(), 0.0f);
        } else if (!requiresGrad) {
            m_grad.clear();
        }
    }
    
    const std::vector<float>& Grad() const { return m_grad; }
    std::vector<float>& Grad() { return m_grad; }
    
    void Backward();
    void ZeroGrad();
    
    // Set gradient function (for internal use)
    void SetGradFn(const GradFunction& gradFn, 
                   const std::vector<Tensor*>& parents);
    
    // Debug methods for autograd
    bool HasGradFn() const { return static_cast<bool>(m_gradFn); }
    size_t NumGradParents() const { return m_gradParents.size(); }
    
    // Utilities
    void Fill(float value);
    void RandomNormal(float mean = 0.0f, float std = 1.0f);
    void RandomUniform(float min = 0.0f, float max = 1.0f);
    
    // Output
    void Print() const;
    std::string ToString() const;
    
    // Static factory methods
    static Tensor Zeros(const std::vector<size_t>& shape, bool requiresGrad = false);
    static Tensor Ones(const std::vector<size_t>& shape, bool requiresGrad = false);
    static Tensor Eye(size_t n, bool requiresGrad = false);
    static Tensor Randn(const std::vector<size_t>& shape, bool requiresGrad = false);
    static Tensor Rand(const std::vector<size_t>& shape, bool requiresGrad = false);
};

// Operators for scalars on the left
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(float scalar, const Tensor& tensor);

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace kotml 