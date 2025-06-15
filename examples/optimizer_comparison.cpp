/**
 * Пример сравнения оптимизаторов SGD и Adam
 * Демонстрирует различия в поведении оптимизаторов на простой квадратичной функции
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "kotml/tensor.hpp"
#include "kotml/optim/sgd.hpp"
#include "kotml/optim/adam.hpp"

using namespace kotml;
using namespace kotml::optim;

// Функция для вычисления квадратичной функции потерь
// loss = 0.5 * sum((x - target)^2)
float ComputeLoss(const Tensor& x, const Tensor& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < x.Size(); ++i) {
        float diff = x[i] - target[i];
        loss += 0.5f * diff * diff;
    }
    return loss;
}

// Вычисление градиентов для квадратичной функции потерь
// grad = x - target
void ComputeGradients(Tensor& x, const Tensor& target) {
    for (size_t i = 0; i < x.Size(); ++i) {
        x.Grad()[i] = x[i] - target[i];
    }
}

// Функция для запуска оптимизации с заданным оптимизатором
void RunOptimization(const std::string& optimizer_name, 
                    Optimizer& optimizer, 
                    Tensor& param, 
                    const Tensor& target,
                    int max_iterations = 100) {
    
    std::cout << "\n=== " << optimizer_name << " Optimization ===" << std::endl;
    std::cout << "Initial parameters: [" << param[0] << ", " << param[1] << "]" << std::endl;
    std::cout << "Target values: [" << target[0] << ", " << target[1] << "]" << std::endl;
    
    float initial_loss = ComputeLoss(param, target);
    std::cout << "Initial loss: " << initial_loss << std::endl;
    
    std::cout << "\nOptimization progress:" << std::endl;
    std::cout << std::setw(4) << "Iter" << std::setw(12) << "Loss" 
              << std::setw(12) << "Param[0]" << std::setw(12) << "Param[1]" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    // Добавляем параметр в оптимизатор
    optimizer.AddParameter(param);
    
    for (int i = 0; i < max_iterations; ++i) {
        // Обнуляем градиенты
        optimizer.ZeroGrad();
        
        // Вычисляем градиенты
        ComputeGradients(param, target);
        
        // Выполняем шаг оптимизации
        optimizer.Step();
        
        // Вычисляем текущую функцию потерь
        float current_loss = ComputeLoss(param, target);
        
        // Выводим прогресс каждые 10 итераций
        if (i % 10 == 0 || i == max_iterations - 1) {
            std::cout << std::setw(4) << i << std::setw(12) << std::fixed << std::setprecision(6) 
                      << current_loss << std::setw(12) << param[0] << std::setw(12) << param[1] << std::endl;
        }
        
        // Останавливаемся, если достигли достаточной точности
        if (current_loss < 1e-6f) {
            std::cout << "Converged at iteration " << i << std::endl;
            break;
        }
    }
    
    float final_loss = ComputeLoss(param, target);
    std::cout << "\nFinal loss: " << final_loss << std::endl;
    std::cout << "Final parameters: [" << param[0] << ", " << param[1] << "]" << std::endl;
    std::cout << "Distance to target: " << std::sqrt(std::pow(param[0] - target[0], 2) + 
                                                    std::pow(param[1] - target[1], 2)) << std::endl;
}

int main() {
    std::cout << "=== Optimizer Comparison Demo ===" << std::endl;
    std::cout << "Comparing SGD and Adam optimizers on a simple quadratic function" << std::endl;
    std::cout << "Objective: minimize 0.5 * sum((x - target)^2)" << std::endl;
    
    // Целевые значения
    Tensor target(std::vector<float>{1.0f, 2.0f}, std::vector<size_t>{2}, false);
    
    // === SGD Optimization ===
    {
        // Начальные параметры для SGD
        Tensor param_sgd(std::vector<float>{5.0f, -3.0f}, std::vector<size_t>{2}, true);
        
        // Создаем SGD оптимизатор
        SGD sgd_optimizer(0.1f);  // learning rate = 0.1
        
        RunOptimization("SGD", sgd_optimizer, param_sgd, target);
    }
    
    // === Adam Optimization ===
    {
        // Начальные параметры для Adam (те же самые)
        Tensor param_adam(std::vector<float>{5.0f, -3.0f}, std::vector<size_t>{2}, true);
        
        // Создаем Adam оптимизатор
        Adam adam_optimizer(0.1f);  // learning rate = 0.1
        
        RunOptimization("Adam", adam_optimizer, param_adam, target);
    }
    
    // === SGD with Momentum ===
    {
        // Начальные параметры для SGD с momentum
        Tensor param_sgd_momentum(std::vector<float>{5.0f, -3.0f}, std::vector<size_t>{2}, true);
        
        // Создаем SGD оптимизатор с momentum
        SGD sgd_momentum_optimizer(0.1f, 0.9f);  // learning rate = 0.1, momentum = 0.9
        
        RunOptimization("SGD with Momentum", sgd_momentum_optimizer, param_sgd_momentum, target);
    }
    
    // === Сравнение конфигураций ===
    std::cout << "\n=== Optimizer Configurations ===" << std::endl;
    
    SGD sgd_config(0.01f, 0.9f, 0.0f, 0.001f, true, 1.0f);
    std::cout << "SGD Config: " << sgd_config.GetConfig() << std::endl;
    
    Adam adam_config(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, true, 1.0f);
    std::cout << "Adam Config: " << adam_config.GetConfig() << std::endl;
    
    // === Демонстрация различных learning rates ===
    std::cout << "\n=== Learning Rate Comparison (SGD) ===" << std::endl;
    
    std::vector<float> learning_rates = {0.01f, 0.1f, 0.5f};
    
    for (float lr : learning_rates) {
        std::cout << "\n--- Learning Rate: " << lr << " ---" << std::endl;
        
        Tensor param_lr(std::vector<float>{5.0f, -3.0f}, std::vector<size_t>{2}, true);
        SGD sgd_lr(lr);
        sgd_lr.AddParameter(param_lr);
        
        float initial_loss = ComputeLoss(param_lr, target);
        
        // Выполняем 20 итераций
        for (int i = 0; i < 20; ++i) {
            sgd_lr.ZeroGrad();
            ComputeGradients(param_lr, target);
            sgd_lr.Step();
        }
        
        float final_loss = ComputeLoss(param_lr, target);
        std::cout << "Initial loss: " << initial_loss << ", Final loss: " << final_loss << std::endl;
        std::cout << "Final params: [" << param_lr[0] << ", " << param_lr[1] << "]" << std::endl;
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "- SGD: Simple gradient descent, predictable behavior" << std::endl;
    std::cout << "- SGD with Momentum: Accelerated convergence, smoother updates" << std::endl;
    std::cout << "- Adam: Adaptive learning rates, good for various problems" << std::endl;
    std::cout << "- Learning rate significantly affects convergence speed and stability" << std::endl;
    
    return 0;
} 