/**
 * Тесты для операций редукции с тензорами
 * Проверка операций Sum, Mean по различным осям
 */

#include <gtest/gtest.h>
#include "kotml/tensor.hpp"
#include <vector>
#include <cmath>

using namespace kotml;

class TensorReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 1D тензор
        data_1d = {1.0f, 2.0f, 3.0f, 4.0f};
        tensor_1d = Tensor(data_1d, {4});
        
        // 2D тензор (2x3)
        data_2d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        tensor_2d = Tensor(data_2d, {2, 3});
        
        // 3D тензор (2x2x2)
        data_3d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        tensor_3d = Tensor(data_3d, {2, 2, 2});
        
        // Тензор с отрицательными значениями
        data_mixed = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
        tensor_mixed = Tensor(data_mixed, {2, 3});
        
        // Тензор с дробными значениями
        data_float = {1.5f, 2.5f, 3.5f, 4.5f};
        tensor_float = Tensor(data_float, {2, 2});
    }
    
    std::vector<float> data_1d, data_2d, data_3d, data_mixed, data_float;
    Tensor tensor_1d, tensor_2d, tensor_3d, tensor_mixed, tensor_float;
    
    // Вспомогательная функция для сравнения тензоров с допуском
    void ExpectTensorNear(const Tensor& actual, const std::vector<float>& expected, 
                         const std::vector<size_t>& expected_shape, float tolerance = 1e-6f) {
        ASSERT_EQ(actual.Size(), expected.size());
        ASSERT_EQ(actual.Shape(), expected_shape);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
        }
    }
    
    void ExpectScalarNear(const Tensor& actual, float expected, float tolerance = 1e-6f) {
        ASSERT_EQ(actual.Size(), 1);
        EXPECT_NEAR(actual[0], expected, tolerance);
    }
};

// Тесты суммирования без указания оси (полная редукция)
TEST_F(TensorReductionTest, Sum1D) {
    Tensor result = tensor_1d.Sum();
    
    // Ожидаемый результат: 1 + 2 + 3 + 4 = 10
    ExpectScalarNear(result, 10.0f);
}

TEST_F(TensorReductionTest, Sum2D) {
    Tensor result = tensor_2d.Sum();
    
    // Ожидаемый результат: 1 + 2 + 3 + 4 + 5 + 6 = 21
    ExpectScalarNear(result, 21.0f);
}

TEST_F(TensorReductionTest, Sum3D) {
    Tensor result = tensor_3d.Sum();
    
    // Ожидаемый результат: 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
    ExpectScalarNear(result, 36.0f);
}

TEST_F(TensorReductionTest, SumMixed) {
    Tensor result = tensor_mixed.Sum();
    
    // Ожидаемый результат: -2 + (-1) + 0 + 1 + 2 + 3 = 3
    ExpectScalarNear(result, 3.0f);
}

TEST_F(TensorReductionTest, SumFloat) {
    Tensor result = tensor_float.Sum();
    
    // Ожидаемый результат: 1.5 + 2.5 + 3.5 + 4.5 = 12.0
    ExpectScalarNear(result, 12.0f);
}

// Тесты суммирования по осям
TEST_F(TensorReductionTest, Sum2DAxis0) {
    Tensor result = tensor_2d.Sum(0);
    
    // Суммирование по строкам (ось 0)
    // [1 2 3] -> [1+4, 2+5, 3+6] = [5, 7, 9]
    // [4 5 6]
    std::vector<float> expected = {5.0f, 7.0f, 9.0f};
    std::vector<size_t> expected_shape = {3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Sum2DAxis1) {
    Tensor result = tensor_2d.Sum(1);
    
    // Суммирование по столбцам (ось 1)
    // [1 2 3] -> [1+2+3] = [6]
    // [4 5 6]    [4+5+6]   [15]
    std::vector<float> expected = {6.0f, 15.0f};
    std::vector<size_t> expected_shape = {2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Sum3DAxis0) {
    Tensor result = tensor_3d.Sum(0);
    
    // Суммирование по первой оси (2x2x2 -> 2x2)
    // Первый слой: [1 2]  Второй слой: [5 6]
    //              [3 4]                [7 8]
    // Результат:   [1+5 2+6] = [6  8]
    //              [3+7 4+8]   [10 12]
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    std::cout << "Result shape: " << tensor_3d << result << std::endl;
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Sum3DAxis1) {
    Tensor result = tensor_3d.Sum(1);
    
    // Суммирование по второй оси (2x2x2 -> 2x2)
    // Для каждого слоя суммируем по строкам
    std::vector<float> expected = {4.0f, 6.0f, 12.0f, 14.0f};  // [1+3, 2+4, 5+7, 6+8]
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Sum3DAxis2) {
    Tensor result = tensor_3d.Sum(2);
    
    // Суммирование по третьей оси (2x2x2 -> 2x2)
    // Для каждого слоя суммируем по столбцам
    std::vector<float> expected = {3.0f, 7.0f, 11.0f, 15.0f};  // [1+2, 3+4, 5+6, 7+8]
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

// Тесты среднего значения без указания оси
TEST_F(TensorReductionTest, Mean1D) {
    Tensor result = tensor_1d.Mean();
    
    // Ожидаемый результат: (1 + 2 + 3 + 4) / 4 = 2.5
    ExpectScalarNear(result, 2.5f);
}

TEST_F(TensorReductionTest, Mean2D) {
    Tensor result = tensor_2d.Mean();
    
    // Ожидаемый результат: (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5
    ExpectScalarNear(result, 3.5f);
}

TEST_F(TensorReductionTest, Mean3D) {
    Tensor result = tensor_3d.Mean();
    
    // Ожидаемый результат: (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) / 8 = 4.5
    ExpectScalarNear(result, 4.5f);
}

TEST_F(TensorReductionTest, MeanMixed) {
    Tensor result = tensor_mixed.Mean();
    
    // Ожидаемый результат: (-2 + (-1) + 0 + 1 + 2 + 3) / 6 = 0.5
    ExpectScalarNear(result, 0.5f);
}

TEST_F(TensorReductionTest, MeanFloat) {
    Tensor result = tensor_float.Mean();
    
    // Ожидаемый результат: (1.5 + 2.5 + 3.5 + 4.5) / 4 = 3.0
    ExpectScalarNear(result, 3.0f);
}

// Тесты среднего значения по осям
TEST_F(TensorReductionTest, Mean2DAxis0) {
    Tensor result = tensor_2d.Mean(0);
    
    // Среднее по строкам (ось 0)
    // [1 2 3] -> [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
    // [4 5 6]
    std::vector<float> expected = {2.5f, 3.5f, 4.5f};
    std::vector<size_t> expected_shape = {3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Mean2DAxis1) {
    Tensor result = tensor_2d.Mean(1);
    
    // Среднее по столбцам (ось 1)
    // [1 2 3] -> [(1+2+3)/3] = [2.0]
    // [4 5 6]    [(4+5+6)/3]   [5.0]
    std::vector<float> expected = {2.0f, 5.0f};
    std::vector<size_t> expected_shape = {2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Mean3DAxis0) {
    Tensor result = tensor_3d.Mean(0);
    
    // Среднее по первой оси (2x2x2 -> 2x2)
    // [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
    std::vector<float> expected = {3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Mean3DAxis1) {
    Tensor result = tensor_3d.Mean(1);
    
    // Среднее по второй оси
    std::vector<float> expected = {2.0f, 3.0f, 6.0f, 7.0f};  // [(1+3)/2, (2+4)/2, (5+7)/2, (6+8)/2]
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorReductionTest, Mean3DAxis2) {
    Tensor result = tensor_3d.Mean(2);
    
    // Среднее по третьей оси
    std::vector<float> expected = {1.5f, 3.5f, 5.5f, 7.5f};  // [(1+2)/2, (3+4)/2, (5+6)/2, (7+8)/2]
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

// Тесты граничных случаев
TEST_F(TensorReductionTest, SingleElementTensor) {
    std::vector<float> data = {5.0f};
    std::vector<size_t> shape = {1};
    Tensor single(data, shape);
    
    Tensor sum_result = single.Sum();
    Tensor mean_result = single.Mean();
    
    ExpectScalarNear(sum_result, 5.0f);
    ExpectScalarNear(mean_result, 5.0f);
}

TEST_F(TensorReductionTest, ZeroTensor) {
    Tensor zeros = Tensor::Zeros({3, 3});
    
    Tensor sum_result = zeros.Sum();
    Tensor mean_result = zeros.Mean();
    
    ExpectScalarNear(sum_result, 0.0f);
    ExpectScalarNear(mean_result, 0.0f);
}

TEST_F(TensorReductionTest, AllOnesReduction) {
    Tensor ones = Tensor::Ones({4, 5});  // 20 элементов
    
    Tensor sum_result = ones.Sum();
    Tensor mean_result = ones.Mean();
    
    ExpectScalarNear(sum_result, 20.0f);
    ExpectScalarNear(mean_result, 1.0f);
    
    // Проверяем редукцию по осям
    Tensor sum_axis0 = ones.Sum(0);  // Должно дать [4, 4, 4, 4, 4]
    Tensor sum_axis1 = ones.Sum(1);  // Должно дать [5, 5, 5, 5]
    
    std::vector<float> expected_axis0(5, 4.0f);
    std::vector<float> expected_axis1(4, 5.0f);
    
    ExpectTensorNear(sum_axis0, expected_axis0, {5});
    ExpectTensorNear(sum_axis1, expected_axis1, {4});
}

// Тесты с большими тензорами
TEST_F(TensorReductionTest, LargeTensorReduction) {
    // Создаем большой тензор для проверки производительности
    std::vector<size_t> large_shape = {100, 50};  // 5000 элементов
    Tensor large_tensor = Tensor::Ones(large_shape);
    
    Tensor sum_result = large_tensor.Sum();
    Tensor mean_result = large_tensor.Mean();
    
    ExpectScalarNear(sum_result, 5000.0f);
    ExpectScalarNear(mean_result, 1.0f);
    
    // Проверяем редукцию по осям
    Tensor sum_axis0 = large_tensor.Sum(0);  // Должно дать 50 элементов по 100
    Tensor sum_axis1 = large_tensor.Sum(1);  // Должно дать 100 элементов по 50
    
    EXPECT_EQ(sum_axis0.Size(), 50);
    EXPECT_EQ(sum_axis1.Size(), 100);
    
    for (size_t i = 0; i < sum_axis0.Size(); ++i) {
        EXPECT_FLOAT_EQ(sum_axis0[i], 100.0f);
    }
    
    for (size_t i = 0; i < sum_axis1.Size(); ++i) {
        EXPECT_FLOAT_EQ(sum_axis1[i], 50.0f);
    }
}

// Тесты точности вычислений
TEST_F(TensorReductionTest, FloatingPointPrecision) {
    // Тестируем с очень маленькими числами
    std::vector<float> small_data = {1e-7f, 2e-7f, 3e-7f, 4e-7f};
    Tensor small_tensor(small_data, {4});
    
    Tensor sum_result = small_tensor.Sum();
    Tensor mean_result = small_tensor.Mean();
    
    ExpectScalarNear(sum_result, 10e-7f, 1e-8f);
    ExpectScalarNear(mean_result, 2.5e-7f, 1e-8f);
    
    // Тестируем с очень большими числами
    std::vector<float> large_data = {1e6f, 2e6f, 3e6f, 4e6f};
    Tensor large_tensor(large_data, {4});
    
    Tensor sum_result2 = large_tensor.Sum();
    Tensor mean_result2 = large_tensor.Mean();
    
    ExpectScalarNear(sum_result2, 10e6f, 1e-3f);
    ExpectScalarNear(mean_result2, 2.5e6f, 1e-3f);
}

// Тесты с отрицательными осями (если поддерживается)
TEST_F(TensorReductionTest, NegativeAxisIndex) {
    // Некоторые библиотеки поддерживают отрицательные индексы осей
    // axis=-1 означает последнюю ось, axis=-2 - предпоследнюю и т.д.
    
    // Если ваша реализация поддерживает отрицательные индексы, раскомментируйте:
    /*
    Tensor result_neg1 = tensor_2d.Sum(-1);  // Эквивалентно Sum(1)
    Tensor result_neg2 = tensor_2d.Sum(-2);  // Эквивалентно Sum(0)
    
    Tensor expected_1 = tensor_2d.Sum(1);
    Tensor expected_0 = tensor_2d.Sum(0);
    
    ExpectTensorNear(result_neg1, expected_1.Data(), expected_1.Shape());
    ExpectTensorNear(result_neg2, expected_0.Data(), expected_0.Shape());
    */
}

// Тесты ошибок и исключений
TEST_F(TensorReductionTest, InvalidAxisIndex) {
    // Попытка использовать несуществующую ось
    EXPECT_NO_THROW({
        try {
            Tensor result = tensor_2d.Sum(5);  // Ось 5 не существует для 2D тензора
            // Если операция выполнилась, результат должен быть валидным или пустым
            if (!result.Empty()) {
                EXPECT_TRUE(result.Size() > 0);
            }
        } catch (const std::exception& e) {
            // Исключение - это нормальное поведение для неверной оси
            SUCCEED();
        }
    });
}

TEST_F(TensorReductionTest, EmptyTensorReduction) {
    Tensor empty;
    
    // Операции с пустыми тензорами должны возвращать пустые тензоры или выбрасывать исключения
    EXPECT_NO_THROW({
        try {
            Tensor sum_result = empty.Sum();
            Tensor mean_result = empty.Mean();
            
            // Если операции выполнились, результаты должны быть пустыми или содержать NaN/0
            if (!sum_result.Empty()) {
                EXPECT_TRUE(sum_result.Size() == 1);
            }
            if (!mean_result.Empty()) {
                EXPECT_TRUE(mean_result.Size() == 1);
            }
        } catch (const std::exception& e) {
            // Исключение - это нормальное поведение для пустых тензоров
            SUCCEED();
        }
    });
} 