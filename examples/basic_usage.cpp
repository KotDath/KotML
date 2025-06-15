#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;

int main() {
    std::cout << "=== KotML Basic Usage Demo ===" << std::endl;
    
    // Creating tensors
    std::cout << "\n1. Creating tensors:" << std::endl;
    
    // Creating tensor from data
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "Tensor a: " << a << std::endl;
    
    // Creating zero tensor
    Tensor zeros = Tensor::Zeros({2, 2});
    std::cout << "Zero tensor: " << zeros << std::endl;
    
    // Creating identity matrix
    Tensor eye = Tensor::Eye(3);
    std::cout << "Identity matrix: " << eye << std::endl;
    
    // 2. Arithmetic operations
    std::cout << "\n2. Arithmetic operations:" << std::endl;
    
    Tensor b({2, 3}, {6, 5, 4, 3, 2, 1});
    std::cout << "Tensor b: " << b << std::endl;
    
    Tensor sum = a + b;
    std::cout << "a + b: " << sum << std::endl;
    
    Tensor diff = a - b;
    std::cout << "a - b: " << diff << std::endl;
    
    Tensor prod = a * b;
    std::cout << "a * b (element-wise): " << prod << std::endl;
    
    // Operations with scalars
    Tensor scaled = a * 2.0f;
    std::cout << "a * 2: " << scaled << std::endl;
    
    // 3. Linear algebra
    std::cout << "\n3. Linear algebra:" << std::endl;
    
    Tensor x({3, 2}, {1, 2, 3, 4, 5, 6});
    Tensor y({2, 4}, {1, 0, 1, 0, 0, 1, 0, 1});
    
    std::cout << "Matrix x: " << x << std::endl;
    std::cout << "Matrix y: " << y << std::endl;
    
    Tensor matmul = x.Matmul(y);
    std::cout << "x @ y: " << matmul << std::endl;
    
    Tensor transposed = x.Transpose();
    std::cout << "x transposed: " << transposed << std::endl;
    
    // 4. Shape operations
    std::cout << "\n4. Shape operations:" << std::endl;
    
    Tensor original({2, 6}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::cout << "Original: " << original << std::endl;
    
    Tensor reshaped = original.Reshape({3, 4});
    std::cout << "Reshaped to 3x4: " << reshaped << std::endl;
    
    // 5. Reduction operations
    std::cout << "\n5. Reduction operations:" << std::endl;
    
    Tensor data({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "Data: " << data << std::endl;
    
    Tensor sum_all = data.Sum();
    std::cout << "Sum of all elements: " << sum_all << std::endl;
    
    Tensor sum_axis0 = data.Sum(0);
    std::cout << "Sum along axis 0: " << sum_axis0 << std::endl;
    
    Tensor sum_axis1 = data.Sum(1);
    std::cout << "Sum along axis 1: " << sum_axis1 << std::endl;
    
    Tensor mean_all = data.Mean();
    std::cout << "Mean of all elements: " << mean_all << std::endl;
    
    // 6. Automatic differentiation
    std::cout << "\n6. Automatic differentiation:" << std::endl;
    
    // Creating tensors with gradient tracking
    Tensor x_grad({2.0f, 3.0f}, {2}, true);
    Tensor y_grad({1.0f, 4.0f}, {2}, true);
    
    std::cout << "x (with grad): " << x_grad << std::endl;
    std::cout << "y (with grad): " << y_grad << std::endl;
    
    // Computing z = x * y + x
    Tensor z_intermediate = x_grad * y_grad;
    Tensor z_grad = z_intermediate + x_grad;
    
    std::cout << "z = x * y + x: " << z_grad << std::endl;
    
    // Computing sum to get scalar
    Tensor scalar_result = z_grad.Sum();
    std::cout << "Sum result: " << scalar_result << std::endl;
    
    // Backward propagation
    scalar_result.Backward();
    
    std::cout << "Gradient x: ";
    for (size_t i = 0; i < x_grad.Size(); ++i) {
        std::cout << x_grad.Grad()[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Gradient y: ";
    for (size_t i = 0; i < y_grad.Size(); ++i) {
        std::cout << y_grad.Grad()[i] << " ";
    }
    std::cout << std::endl;
    
    // 7. Random tensors
    std::cout << "\n7. Random tensors:" << std::endl;
    
    Tensor random_uniform = Tensor::Randn({2, 3});
    random_uniform.RandomUniform(0.0f, 1.0f);
    std::cout << "Random uniform [0,1]: " << random_uniform << std::endl;
    
    Tensor random_normal = Tensor::Randn({2, 3});
    std::cout << "Random normal (0,1): " << random_normal << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    return 0;
} 