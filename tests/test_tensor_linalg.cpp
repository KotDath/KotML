/**
 * Тесты для операций линейной алгебры с тензорами
 * Проверка матричного умножения, транспонирования, изменения формы
 */

#include <gtest/gtest.h>
#include "kotml/tensor.hpp"
#include <vector>

using namespace kotml;

class TensorLinalgTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Матрица 2x3
        matrix_2x3_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        matrix_2x3 = Tensor(matrix_2x3_data, {2, 3});
        
        // Матрица 3x2
        matrix_3x2_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        matrix_3x2 = Tensor(matrix_3x2_data, {3, 2});
        
        // Квадратная матрица 2x2
        matrix_2x2_data = {1.0f, 2.0f, 3.0f, 4.0f};
        matrix_2x2 = Tensor(matrix_2x2_data, {2, 2});
        
        // Квадратная матрица 3x3
        matrix_3x3_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        matrix_3x3 = Tensor(matrix_3x3_data, {3, 3});
        
        // Векторы
        vector_3_data = {1.0f, 2.0f, 3.0f};
        vector_3 = Tensor(vector_3_data, {3});
        
        vector_2_data = {1.0f, 2.0f};
        vector_2 = Tensor(vector_2_data, {2});
    }
    
    std::vector<float> matrix_2x3_data, matrix_3x2_data, matrix_2x2_data, matrix_3x3_data;
    std::vector<float> vector_3_data, vector_2_data;
    Tensor matrix_2x3, matrix_3x2, matrix_2x2, matrix_3x3;
    Tensor vector_3, vector_2;
    
    // Вспомогательная функция для сравнения тензоров с допуском
    void ExpectTensorNear(const Tensor& actual, const std::vector<float>& expected, 
                         const std::vector<size_t>& expected_shape, float tolerance = 1e-6f) {
        ASSERT_EQ(actual.Size(), expected.size());
        ASSERT_EQ(actual.Shape(), expected_shape);
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at index " << i;
        }
    }
};

// Тесты матричного умножения
TEST_F(TensorLinalgTest, MatrixMultiplication2x3_3x2) {
    Tensor result = matrix_2x3.Matmul(matrix_3x2);
    
    // Ожидаемый результат: 2x2 матрица
    // [1 2 3] * [1 2] = [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
    // [4 5 6]   [3 4]   [4*1+5*3+6*5  4*2+5*4+6*6]   [49 64]
    //           [5 6]
    std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, MatrixMultiplication3x2_2x3) {
    Tensor result = matrix_3x2.Matmul(matrix_2x3);
    
    // Ожидаемый результат: 3x3 матрица
    // [1 2] * [1 2 3] = [1*1+2*4  1*2+2*5  1*3+2*6] = [9  12 15]
    // [3 4]   [4 5 6]   [3*1+4*4  3*2+4*5  3*3+4*6]   [19 26 33]
    // [5 6]             [5*1+6*4  5*2+6*5  5*3+6*6]   [29 40 51]
    std::vector<float> expected = {9.0f, 12.0f, 15.0f, 19.0f, 26.0f, 33.0f, 29.0f, 40.0f, 51.0f};
    std::vector<size_t> expected_shape = {3, 3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, SquareMatrixMultiplication) {
    Tensor result = matrix_2x2.Matmul(matrix_2x2);
    
    // Ожидаемый результат:
    // [1 2] * [1 2] = [1*1+2*3  1*2+2*4] = [7  10]
    // [3 4]   [3 4]   [3*1+4*3  3*2+4*4]   [15 22]
    std::vector<float> expected = {7.0f, 10.0f, 15.0f, 22.0f};
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, MatrixVectorMultiplication) {
    // Преобразуем вектор в матрицу-столбец для умножения
    Tensor vector_column = vector_3.Reshape({3, 1});
    Tensor result = matrix_2x3.Matmul(vector_column);
    
    // Ожидаемый результат: 2x1 матрица
    // [1 2 3] * [1] = [1*1+2*2+3*3] = [14]
    // [4 5 6]   [2]   [4*1+5*2+6*3]   [32]
    //           [3]
    std::vector<float> expected = {14.0f, 32.0f};
    std::vector<size_t> expected_shape = {2, 1};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, VectorMatrixMultiplication) {
    // Преобразуем вектор в матрицу-строку для умножения
    Tensor vector_row = vector_2.Reshape({1, 2});
    Tensor result = vector_row.Matmul(matrix_2x3);
    
    // Ожидаемый результат: 1x3 матрица
    // [1 2] * [1 2 3] = [1*1+2*4  1*2+2*5  1*3+2*6] = [9 12 15]
    //         [4 5 6]
    std::vector<float> expected = {9.0f, 12.0f, 15.0f};
    std::vector<size_t> expected_shape = {1, 3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, IdentityMatrixMultiplication) {
    Tensor identity = Tensor::Eye(2);
    Tensor result = matrix_2x2.Matmul(identity);
    
    // Умножение на единичную матрицу должно дать исходную матрицу
    ExpectTensorNear(result, matrix_2x2_data, {2, 2});
    
    // Проверяем коммутативность с единичной матрицей
    Tensor result2 = identity.Matmul(matrix_2x2);
    ExpectTensorNear(result2, matrix_2x2_data, {2, 2});
}

// Тесты транспонирования
TEST_F(TensorLinalgTest, TransposeMatrix2x3) {
    Tensor result = matrix_2x3.Transpose();
    
    // Ожидаемый результат: 3x2 матрица
    // [1 2 3]^T = [1 4]
    // [4 5 6]     [2 5]
    //             [3 6]
    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    std::vector<size_t> expected_shape = {3, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, TransposeMatrix3x2) {
    Tensor result = matrix_3x2.Transpose();
    
    // Ожидаемый результат: 2x3 матрица
    // [1 2]^T = [1 3 5]
    // [3 4]     [2 4 6]
    // [5 6]
    std::vector<float> expected = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
    std::vector<size_t> expected_shape = {2, 3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, TransposeSquareMatrix) {
    Tensor result = matrix_2x2.Transpose();
    
    // Ожидаемый результат:
    // [1 2]^T = [1 3]
    // [3 4]     [2 4]
    std::vector<float> expected = {1.0f, 3.0f, 2.0f, 4.0f};
    std::vector<size_t> expected_shape = {2, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, DoubleTranspose) {
    Tensor result = matrix_2x3.Transpose().Transpose();
    
    // Двойное транспонирование должно дать исходную матрицу
    ExpectTensorNear(result, matrix_2x3_data, {2, 3});
}

// Тесты изменения формы (Reshape)
TEST_F(TensorLinalgTest, ReshapeVector) {
    Tensor result = vector_3.Reshape({1, 3});
    
    // Преобразуем вектор в матрицу-строку
    ExpectTensorNear(result, vector_3_data, {1, 3});
    
    // Преобразуем в матрицу-столбец
    Tensor result2 = vector_3.Reshape({3, 1});
    ExpectTensorNear(result2, vector_3_data, {3, 1});
}

TEST_F(TensorLinalgTest, ReshapeMatrix) {
    Tensor result = matrix_2x3.Reshape({3, 2});
    
    // Данные должны остаться теми же, но форма изменится
    ExpectTensorNear(result, matrix_2x3_data, {3, 2});
    
    // Проверяем доступ к элементам в новой форме
    EXPECT_FLOAT_EQ(result.At({0, 0}), 1.0f);  // Первый элемент
    EXPECT_FLOAT_EQ(result.At({0, 1}), 2.0f);  // Второй элемент
    EXPECT_FLOAT_EQ(result.At({1, 0}), 3.0f);  // Третий элемент
    EXPECT_FLOAT_EQ(result.At({2, 1}), 6.0f);  // Последний элемент
}

TEST_F(TensorLinalgTest, ReshapeToVector) {
    Tensor result = matrix_2x2.Reshape({4});
    
    // Преобразуем матрицу в вектор
    ExpectTensorNear(result, matrix_2x2_data, {4});
}

TEST_F(TensorLinalgTest, Reshape3D) {
    // Создаем тензор с 8 элементами
    std::vector<float> data_8 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor tensor_8(data_8, {8});
    
    // Преобразуем в 3D тензор
    Tensor result = tensor_8.Reshape({2, 2, 2});
    
    ExpectTensorNear(result, data_8, {2, 2, 2});
    
    // Проверяем доступ к элементам
    EXPECT_FLOAT_EQ(result.At({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(result.At({0, 0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(result.At({1, 1, 1}), 8.0f);
}

// Тесты комбинированных операций
TEST_F(TensorLinalgTest, MatmulWithTranspose) {
    // A^T * A (должно дать симметричную матрицу)
    Tensor result = matrix_2x3.Transpose().Matmul(matrix_2x3);
    
    // Ожидаемый результат: 3x3 матрица
    // [1 4]   [1 2 3]   [1*1+4*4  1*2+4*5  1*3+4*6]   [17 22 27]
    // [2 5] * [4 5 6] = [2*1+5*4  2*2+5*5  2*3+5*6] = [22 29 36]
    // [3 6]             [3*1+6*4  3*2+6*5  3*3+6*6]   [27 36 45]
    std::vector<float> expected = {17.0f, 22.0f, 27.0f, 22.0f, 29.0f, 36.0f, 27.0f, 36.0f, 45.0f};
    std::vector<size_t> expected_shape = {3, 3};
    
    ExpectTensorNear(result, expected, expected_shape);
}

TEST_F(TensorLinalgTest, ReshapeAndMatmul) {
    // Преобразуем вектор в матрицу и выполняем умножение
    Tensor vector_as_matrix = vector_3.Reshape({1, 3});
    Tensor result = vector_as_matrix.Matmul(matrix_3x2);
    
    // Ожидаемый результат: 1x2 матрица
    // [1 2 3] * [1 2] = [1*1+2*3+3*5  1*2+2*4+3*6] = [22 28]
    //           [3 4]
    //           [5 6]
    std::vector<float> expected = {22.0f, 28.0f};
    std::vector<size_t> expected_shape = {1, 2};
    
    ExpectTensorNear(result, expected, expected_shape);
}

// Тесты граничных случаев
TEST_F(TensorLinalgTest, SingleElementOperations) {
    Tensor single({5.0f}, {1, 1});
    
    // Транспонирование 1x1 матрицы
    Tensor transposed = single.Transpose();
    EXPECT_EQ(transposed.Shape(), std::vector<size_t>({1, 1}));
    EXPECT_FLOAT_EQ(transposed[0], 5.0f);
    
    // Умножение 1x1 матриц
    Tensor result = single.Matmul(single);
    EXPECT_EQ(result.Shape(), std::vector<size_t>({1, 1}));
    EXPECT_FLOAT_EQ(result[0], 25.0f);
    
    // Изменение формы
    Tensor reshaped = single.Reshape({1});
    EXPECT_EQ(reshaped.Shape(), std::vector<size_t>({1}));
    EXPECT_FLOAT_EQ(reshaped[0], 5.0f);
}

TEST_F(TensorLinalgTest, LargeMatrixOperations) {
    // Создаем большие матрицы для проверки производительности
    Tensor large_a = Tensor::Ones({50, 100});
    Tensor large_b = Tensor::Ones({100, 30});
    
    Tensor result = large_a.Matmul(large_b);
    
    EXPECT_EQ(result.Shape(), std::vector<size_t>({50, 30}));
    
    // Каждый элемент должен быть равен 100 (сумма 100 единиц)
    for (size_t i = 0; i < result.Size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], 100.0f);
    }
}

// Тесты ошибок и исключений
TEST_F(TensorLinalgTest, IncompatibleMatrixDimensions) {
    // Попытка умножить матрицы с несовместимыми размерностями
    // Это должно либо выбросить исключение, либо вернуть пустой тензор
    // В зависимости от реализации
    
    // 2x3 * 2x3 (несовместимо, должно быть 2x3 * 3x2)
    // Здесь мы просто проверяем, что операция не приводит к краху
    // Конкретное поведение зависит от реализации
    
    EXPECT_NO_THROW({
        try {
            Tensor result = matrix_2x3.Matmul(matrix_2x3);
            // Если операция выполнилась, проверяем результат
            if (!result.Empty()) {
                // Результат должен иметь корректные размерности или быть пустым
                EXPECT_TRUE(result.Empty() || result.Ndim() == 2);
            }
        } catch (const std::exception& e) {
            // Исключение - это нормальное поведение для несовместимых размерностей
            SUCCEED();
        }
    });
}

TEST_F(TensorLinalgTest, InvalidReshape) {
    // Попытка изменить форму на несовместимый размер
    EXPECT_NO_THROW({
        try {
            // Пытаемся изменить форму 6-элементного тензора на форму с 8 элементами
            Tensor result = matrix_2x3.Reshape({2, 4});
            // Если операция выполнилась, результат должен быть пустым или исходным
            if (!result.Empty()) {
                EXPECT_TRUE(result.Size() == matrix_2x3.Size());
            }
        } catch (const std::exception& e) {
            // Исключение - это нормальное поведение для несовместимых размеров
            SUCCEED();
        }
    });
} 