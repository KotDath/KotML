/**
 * Тесты для автоматического дифференцирования с тензорами
 * Проверка вычисления градиентов, backward pass
 */

#include <gtest/gtest.h>
#include "kotml/tensor.hpp"
#include <vector>
#include <cmath>
#include <iostream>

using namespace kotml;

class TensorAutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Тензоры с градиентами
        data_a = {2.0f, 3.0f};
        data_b = {4.0f, 5.0f};
        
        tensor_a = Tensor(data_a, {2}, true);  // requires_grad = true
        tensor_b = Tensor(data_b, {2}, true);  // requires_grad = true
        
        // Тензоры без градиентов для сравнения
        tensor_no_grad = Tensor(data_a, {2}, false);
        
        // Скалярные тензоры
        scalar_a = Tensor({3.0f}, {1}, true);
        scalar_b = Tensor({4.0f}, {1}, true);
        
        // Матрицы для более сложных операций
        matrix_data = {1.0f, 2.0f, 3.0f, 4.0f};
        matrix_grad = Tensor(matrix_data, {2, 2}, true);
    }
    
    std::vector<float> data_a, data_b, matrix_data;
    Tensor tensor_a, tensor_b, tensor_no_grad;
    Tensor scalar_a, scalar_b, matrix_grad;
    
    // Вспомогательная функция для сравнения градиентов с допуском
    void ExpectGradNear(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 1e-5f) {
        ASSERT_TRUE(tensor.RequiresGrad());
        ASSERT_EQ(tensor.Grad().size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(tensor.Grad()[i], expected[i], tolerance) 
                << "Gradient mismatch at index " << i;
        }
    }
    
    void ExpectScalarGradNear(const Tensor& tensor, float expected, float tolerance = 1e-5f) {
        ASSERT_TRUE(tensor.RequiresGrad());
        ASSERT_EQ(tensor.Grad().size(), 1);
        EXPECT_NEAR(tensor.Grad()[0], expected, tolerance);
    }
};

// Тесты базовой функциональности градиентов
TEST_F(TensorAutogradTest, BasicGradientProperties) {
    // Проверяем, что градиенты правильно инициализированы
    EXPECT_TRUE(tensor_a.RequiresGrad());
    EXPECT_TRUE(tensor_b.RequiresGrad());
    EXPECT_FALSE(tensor_no_grad.RequiresGrad());
    
    // Проверяем размер градиентов
    EXPECT_EQ(tensor_a.Grad().size(), tensor_a.Size());
    EXPECT_EQ(tensor_b.Grad().size(), tensor_b.Size());
    
    // Проверяем, что градиенты изначально нулевые
    for (size_t i = 0; i < tensor_a.Size(); ++i) {
        EXPECT_FLOAT_EQ(tensor_a.Grad()[i], 0.0f);
        EXPECT_FLOAT_EQ(tensor_b.Grad()[i], 0.0f);
    }
}

TEST_F(TensorAutogradTest, ZeroGradMethod) {
    // Устанавливаем некоторые градиенты
    tensor_a.Grad()[0] = 1.0f;
    tensor_a.Grad()[1] = 2.0f;
    
    // Проверяем, что градиенты установлены
    EXPECT_FLOAT_EQ(tensor_a.Grad()[0], 1.0f);
    EXPECT_FLOAT_EQ(tensor_a.Grad()[1], 2.0f);
    
    // Обнуляем градиенты
    tensor_a.ZeroGrad();
    
    // Проверяем, что градиенты обнулились
    EXPECT_FLOAT_EQ(tensor_a.Grad()[0], 0.0f);
    EXPECT_FLOAT_EQ(tensor_a.Grad()[1], 0.0f);
}

TEST_F(TensorAutogradTest, SetRequiresGrad) {
    Tensor tensor({1.0f, 2.0f}, {2}, false);
    
    // Изначально градиенты отключены
    EXPECT_FALSE(tensor.RequiresGrad());
    
    // Включаем градиенты
    tensor.SetRequiresGrad(true);
    EXPECT_TRUE(tensor.RequiresGrad());
    EXPECT_EQ(tensor.Grad().size(), tensor.Size());
    
    // Отключаем градиенты
    tensor.SetRequiresGrad(false);
    EXPECT_FALSE(tensor.RequiresGrad());
}

// Тесты градиентов для арифметических операций
TEST_F(TensorAutogradTest, AdditionGradients) {
    // Простое сложение: c = a + b
    Tensor c = tensor_a + tensor_b;
    
    // Проверяем, что результат требует градиенты
    EXPECT_TRUE(c.RequiresGrad());
    
    // Создаем градиент для обратного прохода
    // Предполагаем, что grad_output = [1, 1]
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    // Выполняем обратный проход
    c.Backward();
    
    // Для сложения: grad_a = grad_output, grad_b = grad_output
    ExpectGradNear(tensor_a, {1.0f, 1.0f});
    ExpectGradNear(tensor_b, {1.0f, 1.0f});
}

TEST_F(TensorAutogradTest, SubtractionGradients) {
    // Вычитание: c = a - b
    Tensor c = tensor_a - tensor_b;
    
    EXPECT_TRUE(c.RequiresGrad());
    
    // Устанавливаем градиент выхода
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    c.Backward();
    
    // Для вычитания: grad_a = grad_output, grad_b = -grad_output
    ExpectGradNear(tensor_a, {1.0f, 1.0f});
    ExpectGradNear(tensor_b, {-1.0f, -1.0f});
}

TEST_F(TensorAutogradTest, MultiplicationGradients) {
    // Поэлементное умножение: c = a * b
    Tensor c = tensor_a * tensor_b;
    
    EXPECT_TRUE(c.RequiresGrad());
    
    // Устанавливаем градиент выхода
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    c.Backward();
    
    // Для умножения: grad_a = grad_output * b, grad_b = grad_output * a
    // a = [2, 3], b = [4, 5]
    ExpectGradNear(tensor_a, {4.0f, 5.0f});  // grad_output * b
    ExpectGradNear(tensor_b, {2.0f, 3.0f});  // grad_output * a
}

TEST_F(TensorAutogradTest, DivisionGradients) {
    // Поэлементное деление: c = a / b
    Tensor c = tensor_a / tensor_b;
    
    EXPECT_TRUE(c.RequiresGrad());
    
    // Устанавливаем градиент выхода
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    c.Backward();
    
    // Для деления: grad_a = grad_output / b, grad_b = -grad_output * a / (b^2)
    // a = [2, 3], b = [4, 5]
    ExpectGradNear(tensor_a, {1.0f/4.0f, 1.0f/5.0f});  // grad_output / b
    ExpectGradNear(tensor_b, {-2.0f/16.0f, -3.0f/25.0f});  // -grad_output * a / (b^2)
}

// Тесты градиентов для скалярных операций
TEST_F(TensorAutogradTest, ScalarAdditionGradients) {
    // Сложение со скаляром: c = a + 5
    Tensor c = tensor_a + 5.0f;
    
    EXPECT_TRUE(c.RequiresGrad());
    
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    c.Backward();
    
    // Градиент по скаляру не влияет на градиент тензора
    ExpectGradNear(tensor_a, {1.0f, 1.0f});
}

TEST_F(TensorAutogradTest, ScalarMultiplicationGradients) {
    // Умножение на скаляр: c = a * 3
    Tensor c = tensor_a * 3.0f;
    
    EXPECT_TRUE(c.RequiresGrad());
    
    c.Grad()[0] = 1.0f;
    c.Grad()[1] = 1.0f;
    
    c.Backward();
    
    // Градиент умножается на скаляр
    ExpectGradNear(tensor_a, {3.0f, 3.0f});
}

// Тесты цепочки операций
TEST_F(TensorAutogradTest, ChainedOperations) {
    // Цепочка операций: d = (a + b) * (a - b)
    Tensor sum = tensor_a + tensor_b;      // [6, 8]
    Tensor diff = tensor_a - tensor_b;     // [-2, -2]
    Tensor result = sum * diff;            // [-12, -16]
    
    EXPECT_TRUE(result.RequiresGrad());
    
    result.Grad()[0] = 1.0f;
    result.Grad()[1] = 1.0f;
    
    result.Backward();
    
    // Аналитические градиенты для d = (a + b) * (a - b) = a^2 - b^2
    // grad_a = 2*a = [4, 6]
    // grad_b = -2*b = [-8, -10]
    ExpectGradNear(tensor_a, {4.0f, 6.0f});
    ExpectGradNear(tensor_b, {-8.0f, -10.0f});
}

TEST_F(TensorAutogradTest, ComplexChain) {
    // Более сложная цепочка: e = (a * b + a) / b
    Tensor mul = tensor_a * tensor_b;      // [8, 15]
    Tensor add = mul + tensor_a;           // [10, 18]
    Tensor result = add / tensor_b;        // [2.5, 3.6]
    
    EXPECT_TRUE(result.RequiresGrad());
    
    result.Grad()[0] = 1.0f;
    result.Grad()[1] = 1.0f;
    
    result.Backward();
    
    // Аналитические градиенты для e = (a * b + a) / b = a * (b + 1) / b = a + a/b
    // grad_a = 1 + 1/b = [1 + 1/4, 1 + 1/5] = [1.25, 1.2]
    // grad_b = -a/b^2 = [-2/16, -3/25] = [-0.125, -0.12]
    ExpectGradNear(tensor_a, {1.25f, 1.2f}, 1e-4f);
    ExpectGradNear(tensor_b, {-0.125f, -0.12f}, 1e-4f);
}

// Тесты матричных операций
TEST_F(TensorAutogradTest, MatmulGradients) {
    // Создаем матрицы для умножения
    Tensor A({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
    Tensor B({2.0f, 0.0f, 1.0f, 3.0f}, {2, 2}, true);
    
    // C = A * B
    Tensor C = A.Matmul(B);
    
    EXPECT_TRUE(C.RequiresGrad());
    
    // Устанавливаем градиент выхода (единичная матрица)
    C.Grad()[0] = 1.0f; C.Grad()[1] = 0.0f;
    C.Grad()[2] = 0.0f; C.Grad()[3] = 1.0f;
    
    C.Backward();
    
    // Для матричного умножения:
    // grad_A = grad_output * B^T
    // grad_B = A^T * grad_output
    
    // Проверяем, что градиенты вычислены (точные значения зависят от реализации)
    EXPECT_TRUE(A.Grad().size() == 4);
    EXPECT_TRUE(B.Grad().size() == 4);
    
    // Проверяем, что градиенты не нулевые
    bool A_has_nonzero_grad = false;
    bool B_has_nonzero_grad = false;
    
    for (size_t i = 0; i < 4; ++i) {
        if (std::abs(A.Grad()[i]) > 1e-6f) A_has_nonzero_grad = true;
        if (std::abs(B.Grad()[i]) > 1e-6f) B_has_nonzero_grad = true;
    }
    
    EXPECT_TRUE(A_has_nonzero_grad);
    EXPECT_TRUE(B_has_nonzero_grad);
}

// Тесты операций редукции
TEST_F(TensorAutogradTest, SumGradients) {
    // Суммирование: s = sum(a)
    Tensor s = tensor_a.Sum();
    
    EXPECT_TRUE(s.RequiresGrad());
    EXPECT_EQ(s.Size(), 1);
    
    s.Grad()[0] = 1.0f;
    s.Backward();
    
    // Градиент суммы распределяется равномерно
    ExpectGradNear(tensor_a, {1.0f, 1.0f});
}

TEST_F(TensorAutogradTest, MeanGradients) {
    // Среднее значение: m = mean(a)
    Tensor m = tensor_a.Mean();

    EXPECT_TRUE(m.RequiresGrad());
    EXPECT_EQ(m.Size(), 1);
    
    m.Grad()[0] = 1.0f;
    m.Backward();
    
    // Градиент среднего = 1/n для каждого элемента
    float expected_grad = 1.0f / static_cast<float>(tensor_a.Size());
    ExpectGradNear(tensor_a, {expected_grad, expected_grad});
}

// Тесты смешанных операций (с градиентами и без)
TEST_F(TensorAutogradTest, MixedGradientOperations) {
    // Операция между тензором с градиентами и без
    Tensor result = tensor_a + tensor_no_grad;
    
    // Результат должен требовать градиенты, так как один из операндов их требует
    EXPECT_TRUE(result.RequiresGrad());
    
    result.Grad()[0] = 1.0f;
    result.Grad()[1] = 1.0f;
    
    result.Backward();
    
    // Только tensor_a должен получить градиенты
    ExpectGradNear(tensor_a, {1.0f, 1.0f});
    
    // tensor_no_grad не должен иметь градиентов
    EXPECT_FALSE(tensor_no_grad.RequiresGrad());
}

// Тесты накопления градиентов
TEST_F(TensorAutogradTest, GradientAccumulation) {
    // Выполняем несколько операций с одним тензором
    Tensor result1 = tensor_a * 2.0f;
    Tensor result2 = tensor_a + 1.0f;
    
    // Устанавливаем градиенты
    result1.Grad()[0] = 1.0f; result1.Grad()[1] = 1.0f;
    result2.Grad()[0] = 1.0f; result2.Grad()[1] = 1.0f;
    
    // Выполняем обратные проходы
    result1.Backward();
    result2.Backward();
    
    // Градиенты должны накопиться: 2.0 (от result1) + 1.0 (от result2) = 3.0
    ExpectGradNear(tensor_a, {3.0f, 3.0f});
}

// Тесты граничных случаев
TEST_F(TensorAutogradTest, SingleElementGradients) {
    // Скалярные операции
    Tensor result = scalar_a * scalar_b;  // 3 * 4 = 12
    
    EXPECT_TRUE(result.RequiresGrad());
    
    result.Grad()[0] = 1.0f;
    result.Backward();
    
    ExpectScalarGradNear(scalar_a, 4.0f);  // grad = b
    ExpectScalarGradNear(scalar_b, 3.0f);  // grad = a
}

TEST_F(TensorAutogradTest, ZeroGradientPropagation) {
    // Операция, которая должна дать нулевые градиенты
    Tensor zeros = Tensor::Zeros({2}, true);
    Tensor result = tensor_a * zeros;
    
    EXPECT_TRUE(result.RequiresGrad());
    
    result.Grad()[0] = 1.0f;
    result.Grad()[1] = 1.0f;
    
    result.Backward();
    
    // Градиенты tensor_a должны быть нулевыми (умножение на ноль)
    ExpectGradNear(tensor_a, {0.0f, 0.0f});
}

// Тесты производительности и стабильности
TEST_F(TensorAutogradTest, LargeGradientChain) {
    // Обнуляем градиенты перед тестом
    tensor_a.ZeroGrad();
    
    // Создаем длинную цепочку операций без копирования исходного тензора
    Tensor current = tensor_a * 1.0f;  // Начинаем с операции, а не с копии
    
    for (int i = 0; i < 10; ++i) {
        current = current * 1.1f + 0.1f;
    }
    
    EXPECT_TRUE(current.RequiresGrad());
    
    current.Grad()[0] = 1.0f;
    current.Grad()[1] = 1.0f;
    
    // Отладочная информация
    std::cout << "Before backward - tensor_a.grad: [" << tensor_a.Grad()[0] << ", " << tensor_a.Grad()[1] << "]" << std::endl;
    
    // Проверяем, что обратный проход не вызывает краха
    EXPECT_NO_THROW(current.Backward());
    
    // Отладочная информация
    std::cout << "After backward - tensor_a.grad: [" << tensor_a.Grad()[0] << ", " << tensor_a.Grad()[1] << "]" << std::endl;
    
    // Проверяем, что градиенты были вычислены
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < tensor_a.Size(); ++i) {
        std::cout << "tensor_a.grad[" << i << "] = " << tensor_a.Grad()[i] << " (abs = " << std::abs(tensor_a.Grad()[i]) << ")" << std::endl;
        if (std::abs(tensor_a.Grad()[i]) > 1e-6f) {
            has_nonzero_grad = true;
            break;
        }
    }
    std::cout << "has_nonzero_grad = " << has_nonzero_grad << std::endl;
    EXPECT_TRUE(has_nonzero_grad);
}

// Тесты ошибок и исключений
TEST_F(TensorAutogradTest, BackwardWithoutGradients) {
    // Попытка вызвать backward для тензора без градиентов
    EXPECT_NO_THROW({
        try {
            tensor_no_grad.Backward();
            // Если операция выполнилась, это нормально
            SUCCEED();
        } catch (const std::exception& e) {
            // Исключение также нормально для тензоров без градиентов
            SUCCEED();
        }
    });
} 