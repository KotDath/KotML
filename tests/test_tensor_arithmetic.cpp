/**
 * Тесты для арифметических операций с тензорами
 * Проверка операций +, -, *, / для тензоров и скаляров
 */

#include <gtest/gtest.h>
#include "kotml/tensor.hpp"
#include <vector>
#include <cmath>

using namespace kotml;

class TensorArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Тестовые данные
        data_a = {1.0f, 2.0f, 3.0f, 4.0f};
        data_b = {2.0f, 3.0f, 4.0f, 5.0f};
        shape_1d = {4};
        
        data_2d_a = {1.0f, 2.0f, 3.0f, 4.0f};
        data_2d_b = {5.0f, 6.0f, 7.0f, 8.0f};
        shape_2d = {2, 2};
        
        // Создаем тензоры
        tensor_a = Tensor(data_a, shape_1d);
        tensor_b = Tensor(data_b, shape_1d);
        tensor_2d_a = Tensor(data_2d_a, shape_2d);
        tensor_2d_b = Tensor(data_2d_b, shape_2d);
    }
    
    std::vector<float> data_a, data_b, data_2d_a, data_2d_b;
    std::vector<size_t> shape_1d, shape_2d;
    Tensor tensor_a, tensor_b, tensor_2d_a, tensor_2d_b;
    
    // Вспомогательная функция для сравнения тензоров с допуском
    void ExpectTensorNear(const Tensor& actual, const std::vector<float>& expected, float tolerance = 1e-6f) {
        ASSERT_EQ(actual.Size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
        }
    }
};

// Тесты сложения тензоров
TEST_F(TensorArithmeticTest, TensorAddition) {
    Tensor result = tensor_a + tensor_b;
    
    // Ожидаемый результат: [1+2, 2+3, 3+4, 4+5] = [3, 5, 7, 9]
    std::vector<float> expected = {3.0f, 5.0f, 7.0f, 9.0f};
    ExpectTensorNear(result, expected);
    
    // Проверяем размерности
    EXPECT_EQ(result.Shape(), shape_1d);
}

TEST_F(TensorArithmeticTest, TensorAddition2D) {
    Tensor result = tensor_2d_a + tensor_2d_b;
    
    // Ожидаемый результат: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    ExpectTensorNear(result, expected);
    
    EXPECT_EQ(result.Shape(), shape_2d);
}

// Тесты вычитания тензоров
TEST_F(TensorArithmeticTest, TensorSubtraction) {
    Tensor result = tensor_b - tensor_a;
    
    // Ожидаемый результат: [2-1, 3-2, 4-3, 5-4] = [1, 1, 1, 1]
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, TensorSubtractionNegative) {
    Tensor result = tensor_a - tensor_b;
    
    // Ожидаемый результат: [1-2, 2-3, 3-4, 4-5] = [-1, -1, -1, -1]
    std::vector<float> expected = {-1.0f, -1.0f, -1.0f, -1.0f};
    ExpectTensorNear(result, expected);
}

// Тесты умножения тензоров (поэлементное)
TEST_F(TensorArithmeticTest, TensorMultiplication) {
    Tensor result = tensor_a * tensor_b;
    
    // Ожидаемый результат: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
    std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};
    ExpectTensorNear(result, expected);
}

// Тесты деления тензоров
TEST_F(TensorArithmeticTest, TensorDivision) {
    Tensor result = tensor_b / tensor_a;
    
    // Ожидаемый результат: [2/1, 3/2, 4/3, 5/4] = [2, 1.5, 1.333..., 1.25]
    std::vector<float> expected = {2.0f, 1.5f, 4.0f/3.0f, 1.25f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, TensorDivisionByZero) {
    Tensor zeros = Tensor::Zeros({4});
    Tensor ones = Tensor::Ones({4});
    
    Tensor result = ones / zeros;
    
    // Проверяем, что результат содержит бесконечности
    for (size_t i = 0; i < result.Size(); ++i) {
        EXPECT_TRUE(std::isinf(result[i]));
    }
}

// Тесты скалярных операций
TEST_F(TensorArithmeticTest, ScalarAddition) {
    Tensor result = tensor_a + 10.0f;
    
    // Ожидаемый результат: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
    std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, ScalarSubtraction) {
    Tensor result = tensor_a - 1.0f;
    
    // Ожидаемый результат: [1-1, 2-1, 3-1, 4-1] = [0, 1, 2, 3]
    std::vector<float> expected = {0.0f, 1.0f, 2.0f, 3.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, ScalarMultiplication) {
    Tensor result = tensor_a * 2.0f;
    
    // Ожидаемый результат: [1*2, 2*2, 3*2, 4*2] = [2, 4, 6, 8]
    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, ScalarDivision) {
    Tensor result = tensor_a / 2.0f;
    
    // Ожидаемый результат: [1/2, 2/2, 3/2, 4/2] = [0.5, 1, 1.5, 2]
    std::vector<float> expected = {0.5f, 1.0f, 1.5f, 2.0f};
    ExpectTensorNear(result, expected);
}

// Тесты левосторонних скалярных операций
TEST_F(TensorArithmeticTest, LeftScalarAddition) {
    Tensor result = 10.0f + tensor_a;
    
    // Ожидаемый результат: [10+1, 10+2, 10+3, 10+4] = [11, 12, 13, 14]
    std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, LeftScalarSubtraction) {
    Tensor result = 10.0f - tensor_a;
    
    // Ожидаемый результат: [10-1, 10-2, 10-3, 10-4] = [9, 8, 7, 6]
    std::vector<float> expected = {9.0f, 8.0f, 7.0f, 6.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, LeftScalarMultiplication) {
    Tensor result = 3.0f * tensor_a;
    
    // Ожидаемый результат: [3*1, 3*2, 3*3, 3*4] = [3, 6, 9, 12]
    std::vector<float> expected = {3.0f, 6.0f, 9.0f, 12.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, LeftScalarDivision) {
    Tensor result = 12.0f / tensor_a;
    
    // Ожидаемый результат: [12/1, 12/2, 12/3, 12/4] = [12, 6, 4, 3]
    std::vector<float> expected = {12.0f, 6.0f, 4.0f, 3.0f};
    ExpectTensorNear(result, expected);
}

// Тесты операций присваивания
TEST_F(TensorArithmeticTest, AdditionAssignment) {
    Tensor result = tensor_a;  // Копируем
    result += tensor_b;
    
    // Ожидаемый результат: [1+2, 2+3, 3+4, 4+5] = [3, 5, 7, 9]
    std::vector<float> expected = {3.0f, 5.0f, 7.0f, 9.0f};
    ExpectTensorNear(result, expected);
    
    // Проверяем, что оригинальный тензор не изменился
    ExpectTensorNear(tensor_a, data_a);
}

TEST_F(TensorArithmeticTest, SubtractionAssignment) {
    Tensor result = tensor_b;  // Копируем
    result -= tensor_a;
    
    // Ожидаемый результат: [2-1, 3-2, 4-3, 5-4] = [1, 1, 1, 1]
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, MultiplicationAssignment) {
    Tensor result = tensor_a;  // Копируем
    result *= tensor_b;
    
    // Ожидаемый результат: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
    std::vector<float> expected = {2.0f, 6.0f, 12.0f, 20.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, DivisionAssignment) {
    Tensor result = tensor_b;  // Копируем
    result /= tensor_a;
    
    // Ожидаемый результат: [2/1, 3/2, 4/3, 5/4] = [2, 1.5, 1.333..., 1.25]
    std::vector<float> expected = {2.0f, 1.5f, 4.0f/3.0f, 1.25f};
    ExpectTensorNear(result, expected);
}

// Тесты граничных случаев
TEST_F(TensorArithmeticTest, EmptyTensorOperations) {
    Tensor empty1, empty2;
    std::cout << "empty1: " << empty1 << std::endl;
    std::cout << "empty2: " << empty2 << std::endl;
    // Операции с пустыми тензорами должны возвращать пустые тензоры
    Tensor result_add = empty1 + empty2;
    std::cout << "result_add: " << result_add << std::endl;
    Tensor result_sub = empty1 - empty2;
    Tensor result_mul = empty1 * empty2;
    Tensor result_div = empty1 / empty2;
    
    EXPECT_TRUE(result_add.Empty());
    EXPECT_TRUE(result_sub.Empty());
    EXPECT_TRUE(result_mul.Empty());
    EXPECT_TRUE(result_div.Empty());
}

TEST_F(TensorArithmeticTest, SingleElementTensor) {
    std::vector<float> data_a = {5.0f};
    std::vector<float> data_b = {3.0f};
    std::vector<size_t> shape = {1};
    
    Tensor single_a(data_a, shape);
    Tensor single_b(data_b, shape);
    
    Tensor add_result = single_a + single_b;
    Tensor sub_result = single_a - single_b;
    Tensor mul_result = single_a * single_b;
    Tensor div_result = single_a / single_b;
    
    EXPECT_FLOAT_EQ(add_result[0], 8.0f);
    EXPECT_FLOAT_EQ(sub_result[0], 2.0f);
    EXPECT_FLOAT_EQ(mul_result[0], 15.0f);
    EXPECT_FLOAT_EQ(div_result[0], 5.0f/3.0f);
}

// Тесты цепочки операций
TEST_F(TensorArithmeticTest, ChainedOperations) {
    // (a + b) * 2 - 1
    Tensor result = (tensor_a + tensor_b) * 2.0f - 1.0f;
    
    // Ожидаемый результат: ((1+2)*2-1, (2+3)*2-1, (3+4)*2-1, (4+5)*2-1) = (5, 9, 13, 17)
    std::vector<float> expected = {5.0f, 9.0f, 13.0f, 17.0f};
    ExpectTensorNear(result, expected);
}

TEST_F(TensorArithmeticTest, ComplexExpression) {
    // a * b + a / b - 2 * a
    Tensor result = tensor_a * tensor_b + tensor_a / tensor_b - 2.0f * tensor_a;
    
    // Для каждого элемента: a[i] * b[i] + a[i] / b[i] - 2 * a[i]
    std::vector<float> expected;
    for (size_t i = 0; i < data_a.size(); ++i) {
        float val = data_a[i] * data_b[i] + data_a[i] / data_b[i] - 2.0f * data_a[i];
        expected.push_back(val);
    }
    
    ExpectTensorNear(result, expected);
}

// Тесты производительности и точности
TEST_F(TensorArithmeticTest, LargeTensorOperations) {
    // Создаем большие тензоры для проверки производительности
    std::vector<size_t> large_shape = {100, 100};  // 10,000 элементов
    Tensor large_a = Tensor::Ones(large_shape);
    Tensor large_b = Tensor::Ones(large_shape) * 2.0f;
    
    // Выполняем операции
    Tensor result_add = large_a + large_b;
    Tensor result_mul = large_a * large_b;
    
    // Проверяем несколько элементов
    EXPECT_FLOAT_EQ(result_add[0], 3.0f);
    EXPECT_FLOAT_EQ(result_add[5000], 3.0f);
    EXPECT_FLOAT_EQ(result_mul[0], 2.0f);
    EXPECT_FLOAT_EQ(result_mul[9999], 2.0f);
    
    EXPECT_EQ(result_add.Size(), 10000);
    EXPECT_EQ(result_mul.Size(), 10000);
}

// Тесты с различными типами данных
TEST_F(TensorArithmeticTest, FloatingPointPrecision) {
    // Тестируем операции с очень маленькими и очень большими числами
    std::vector<float> small_data = {1e-7f, 1e-6f};
    std::vector<float> large_data = {1e6f, 1e7f};
    std::vector<size_t> shape = {2};
    
    Tensor small(small_data, shape);
    Tensor large(large_data, shape);
    
    Tensor result = small + large;
    
    // Результат должен быть близок к large, так как small очень мал
    EXPECT_NEAR(result[0], 1e6f, 1e-5f);
    EXPECT_NEAR(result[1], 1e7f, 1e-4f);
} 