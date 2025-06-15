/**
 * Тесты для базовых операций с тензорами
 * Проверка конструкторов, доступа к данным, размерностей
 */

#include <gtest/gtest.h>
#include "kotml/tensor.hpp"
#include <vector>

using namespace kotml;

class TensorBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Общие данные для тестов
        data1d = {1.0f, 2.0f, 3.0f, 4.0f};
        shape1d = {4};
        
        data2d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        shape2d = {2, 3};
        
        data3d = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        shape3d = {2, 2, 2};
    }
    
    std::vector<float> data1d, data2d, data3d;
    std::vector<size_t> shape1d, shape2d, shape3d;
};

// Тесты конструкторов
TEST_F(TensorBasicTest, DefaultConstructor) {
    Tensor t;
    EXPECT_TRUE(t.Empty());
    EXPECT_EQ(t.Size(), 0);
    EXPECT_EQ(t.Ndim(), 0);
    EXPECT_FALSE(t.RequiresGrad());
}

TEST_F(TensorBasicTest, ShapeConstructor) {
    Tensor t(shape2d);
    EXPECT_FALSE(t.Empty());
    EXPECT_EQ(t.Size(), 6);
    EXPECT_EQ(t.Ndim(), 2);
    EXPECT_EQ(t.Shape(), shape2d);
    EXPECT_FALSE(t.RequiresGrad());
    
    // Проверяем, что данные инициализированы нулями
    for (size_t i = 0; i < t.Size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 0.0f);
    }
}

TEST_F(TensorBasicTest, ShapeConstructorWithGrad) {
    Tensor t(shape1d, true);
    EXPECT_TRUE(t.RequiresGrad());
    EXPECT_EQ(t.Grad().size(), t.Size());
}

TEST_F(TensorBasicTest, DataShapeConstructor) {
    Tensor t(data2d, shape2d);
    EXPECT_EQ(t.Size(), 6);
    EXPECT_EQ(t.Ndim(), 2);
    EXPECT_EQ(t.Shape(), shape2d);
    
    // Проверяем данные
    for (size_t i = 0; i < data2d.size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], data2d[i]);
    }
}

TEST_F(TensorBasicTest, InitializerListConstructor) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_EQ(t.Size(), 4);
    EXPECT_EQ(t.Ndim(), 2);
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[1], 2.0f);
    EXPECT_FLOAT_EQ(t[2], 3.0f);
    EXPECT_FLOAT_EQ(t[3], 4.0f);
}

// Тесты копирования и перемещения
TEST_F(TensorBasicTest, CopyConstructor) {
    Tensor original(data1d, shape1d, true);
    Tensor copy(original);
    
    EXPECT_EQ(copy.Size(), original.Size());
    EXPECT_EQ(copy.Shape(), original.Shape());
    EXPECT_EQ(copy.RequiresGrad(), original.RequiresGrad());
    
    // Проверяем, что данные скопированы
    for (size_t i = 0; i < original.Size(); ++i) {
        EXPECT_FLOAT_EQ(copy[i], original[i]);
    }
    
    // Проверяем независимость копии
    copy[0] = 999.0f;
    EXPECT_NE(copy[0], original[0]);
}

TEST_F(TensorBasicTest, MoveConstructor) {
    Tensor original(data1d, shape1d);
    auto originalSize = original.Size();
    auto originalShape = original.Shape();
    
    Tensor moved(std::move(original));
    
    EXPECT_EQ(moved.Size(), originalSize);
    EXPECT_EQ(moved.Shape(), originalShape);
    EXPECT_FLOAT_EQ(moved[0], data1d[0]);
    
    // Оригинальный тензор должен быть в валидном, но неопределенном состоянии
    // Мы не можем гарантировать конкретное состояние после move
}

TEST_F(TensorBasicTest, AssignmentOperator) {
    Tensor t1(data1d, shape1d);
    Tensor t2;
    
    t2 = t1;
    
    EXPECT_EQ(t2.Size(), t1.Size());
    EXPECT_EQ(t2.Shape(), t1.Shape());
    
    for (size_t i = 0; i < t1.Size(); ++i) {
        EXPECT_FLOAT_EQ(t2[i], t1[i]);
    }
}

// Тесты доступа к элементам
TEST_F(TensorBasicTest, IndexAccess) {
    Tensor t(data1d, shape1d);
    
    // Чтение
    EXPECT_FLOAT_EQ(t[0], 1.0f);
    EXPECT_FLOAT_EQ(t[1], 2.0f);
    EXPECT_FLOAT_EQ(t[2], 3.0f);
    EXPECT_FLOAT_EQ(t[3], 4.0f);
    
    // Запись
    t[0] = 10.0f;
    EXPECT_FLOAT_EQ(t[0], 10.0f);
}

TEST_F(TensorBasicTest, MultiDimensionalAccess) {
    Tensor t(data2d, shape2d);  // 2x3 тензор
    
    // Проверяем доступ по многомерным индексам
    EXPECT_FLOAT_EQ(t.At({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.At({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.At({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.At({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.At({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.At({1, 2}), 6.0f);
    
    // Запись
    t.At({0, 0}) = 100.0f;
    EXPECT_FLOAT_EQ(t.At({0, 0}), 100.0f);
}

// Тесты размерностей и формы
TEST_F(TensorBasicTest, ShapeAndSize) {
    Tensor t1d(data1d, shape1d);
    EXPECT_EQ(t1d.Size(), 4);
    EXPECT_EQ(t1d.Ndim(), 1);
    EXPECT_EQ(t1d.Shape(), shape1d);
    
    Tensor t2d(data2d, shape2d);
    EXPECT_EQ(t2d.Size(), 6);
    EXPECT_EQ(t2d.Ndim(), 2);
    EXPECT_EQ(t2d.Shape(), shape2d);
    
    Tensor t3d(data3d, shape3d);
    EXPECT_EQ(t3d.Size(), 8);
    EXPECT_EQ(t3d.Ndim(), 3);
    EXPECT_EQ(t3d.Shape(), shape3d);
}

// Тесты статических методов создания
TEST_F(TensorBasicTest, ZerosFactory) {
    Tensor zeros = Tensor::Zeros({3, 2});
    EXPECT_EQ(zeros.Size(), 6);
    EXPECT_EQ(zeros.Ndim(), 2);
    
    for (size_t i = 0; i < zeros.Size(); ++i) {
        EXPECT_FLOAT_EQ(zeros[i], 0.0f);
    }
}

TEST_F(TensorBasicTest, OnesFactory) {
    Tensor ones = Tensor::Ones({2, 3});
    EXPECT_EQ(ones.Size(), 6);
    EXPECT_EQ(ones.Ndim(), 2);
    
    for (size_t i = 0; i < ones.Size(); ++i) {
        EXPECT_FLOAT_EQ(ones[i], 1.0f);
    }
}

TEST_F(TensorBasicTest, EyeFactory) {
    Tensor eye = Tensor::Eye(3);
    EXPECT_EQ(eye.Size(), 9);
    EXPECT_EQ(eye.Ndim(), 2);
    EXPECT_EQ(eye.Shape(), std::vector<size_t>({3, 3}));
    
    // Проверяем единичную матрицу
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_FLOAT_EQ(eye.At({i, j}), 1.0f);
            } else {
                EXPECT_FLOAT_EQ(eye.At({i, j}), 0.0f);
            }
        }
    }
}

TEST_F(TensorBasicTest, RandomFactories) {
    // Тестируем, что методы создают тензоры правильного размера
    // Конкретные значения не проверяем, так как они случайные
    
    Tensor randn = Tensor::Randn({2, 3});
    EXPECT_EQ(randn.Size(), 6);
    EXPECT_EQ(randn.Ndim(), 2);
    
    Tensor rand = Tensor::Rand({3, 2});
    EXPECT_EQ(rand.Size(), 6);
    EXPECT_EQ(rand.Ndim(), 2);
    
    // Проверяем, что rand создает значения в диапазоне [0, 1)
    bool allInRange = true;
    for (size_t i = 0; i < rand.Size(); ++i) {
        if (rand[i] < 0.0f || rand[i] >= 1.0f) {
            allInRange = false;
            break;
        }
    }
    EXPECT_TRUE(allInRange);
}

// Тесты утилитарных методов
TEST_F(TensorBasicTest, FillMethod) {
    Tensor t(shape2d);
    t.Fill(5.5f);
    
    for (size_t i = 0; i < t.Size(); ++i) {
        EXPECT_FLOAT_EQ(t[i], 5.5f);
    }
}

TEST_F(TensorBasicTest, GradientMethods) {
    Tensor t(data1d, shape1d, true);
    
    // Проверяем начальное состояние градиентов
    EXPECT_TRUE(t.RequiresGrad());
    EXPECT_EQ(t.Grad().size(), t.Size());
    
    // Устанавливаем градиенты
    for (size_t i = 0; i < t.Size(); ++i) {
        t.Grad()[i] = static_cast<float>(i + 1);
    }
    
    // Проверяем градиенты
    for (size_t i = 0; i < t.Size(); ++i) {
        EXPECT_FLOAT_EQ(t.Grad()[i], static_cast<float>(i + 1));
    }
    
    // Обнуляем градиенты
    t.ZeroGrad();
    for (size_t i = 0; i < t.Size(); ++i) {
        EXPECT_FLOAT_EQ(t.Grad()[i], 0.0f);
    }
    
    // Отключаем градиенты
    t.SetRequiresGrad(false);
    EXPECT_FALSE(t.RequiresGrad());
}

// Тесты вывода и строкового представления
TEST_F(TensorBasicTest, StringRepresentation) {
    std::vector<float> data = {1.0f, 2.0f};
    std::vector<size_t> shape = {2};
    Tensor t(data, shape);
    std::string str = t.ToString();
    EXPECT_FALSE(str.empty());
    // Конкретный формат может варьироваться, просто проверяем, что строка не пустая
} 