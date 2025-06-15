#include "kotml/tensor.hpp"
#include <iostream>

using namespace kotml;

int main() {
    std::cout << "=== Simple Automatic Differentiation Demo ===" << std::endl;
    
    // Creating simple tensors with gradients
    Tensor x({2.0f}, {1}, true);
    std::cout << "x = " << x << std::endl;
    
    // Simple function: y = x^2 (implemented as x * x)
    Tensor y = x * x;
    std::cout << "y = x^2 = " << y << std::endl;
    
    // Initialize gradient of output tensor
    y.Grad()[0] = 1.0f;
    
    // Manual gradient computation for demonstration
    // dy/dx = 2*x, so gradient should be 2*2 = 4
    x.Grad()[0] = 2.0f * x[0] * y.Grad()[0];  // Manual gradient computation
    
    std::cout << "Gradient dy/dx = " << x.Grad()[0] << " (expected 4)" << std::endl;
    
    // More complex example
    std::cout << "\n=== More Complex Example ===" << std::endl;
    
    // Creating tensors with gradients
    Tensor a({2.0f}, {1}, true);
    Tensor b({3.0f}, {1}, true);
    
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    
    // Complex function: z = a * b + a
    // First compute a * b
    Tensor ab = a * b;
    std::cout << "a * b = " << ab << std::endl;
    
    // Then add a
    Tensor z = ab + a;
    std::cout << "z = a * b + a = " << z << std::endl;
    
    // Initialize gradient of final result
    z.Grad()[0] = 1.0f;
    
    // Manual gradient computation
    // dz/da = b + 1 = 3 + 1 = 4
    // dz/db = a = 2
    
    // Gradients for intermediate results
    ab.Grad()[0] = z.Grad()[0];  // dz/d(ab) = 1
    
    // Gradients for input variables
    a.Grad()[0] = ab.Grad()[0] * b[0] + z.Grad()[0];  // da: from ab and from direct term
    b.Grad()[0] = ab.Grad()[0] * a[0];  // db: only from ab
    
    std::cout << "Gradient dz/da = " << a.Grad()[0] << " (expected 5)" << std::endl;
    std::cout << "Gradient dz/db = " << b.Grad()[0] << " (expected 3)" << std::endl;
    
    // Reduction operations demonstration
    std::cout << "\n=== Reduction Operations ===" << std::endl;
    
    // Creating multi-element tensor
    Tensor vec({1.0f, 2.0f, 3.0f}, {3}, true);
    std::cout << "Vector: " << vec << std::endl;
    
    // Sum operation
    Tensor sum_result = vec.Sum();
    std::cout << "Sum: " << sum_result << std::endl;
    
    // Initialize gradient for sum
    sum_result.Grad()[0] = 1.0f;
    
    // Manual gradient computation for sum
    // d(sum)/d(vec[i]) = 1 for all i
    for (size_t i = 0; i < vec.Size(); ++i) {
        vec.Grad()[i] = sum_result.Grad()[0];
    }
    
    std::cout << "Gradients for sum: ";
    for (size_t i = 0; i < vec.Size(); ++i) {
        std::cout << vec.Grad()[i] << " ";
    }
    std::cout << "(expected all 1s)" << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    return 0;
} 