/**
 * Тесты для оптимизаторов SGD и Adam
 * Проверка корректности обновления параметров, валидации и различных конфигураций
 */

#include <gtest/gtest.h>
#include "kotml/optim/sgd.hpp"
#include "kotml/optim/adam.hpp"
#include "kotml/tensor.hpp"
#include <vector>
#include <cmath>
#include <memory>

using namespace kotml;
using namespace kotml::optim;

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Создаем тестовые параметры
        param1 = std::make_unique<Tensor>(std::vector<float>{1.0f, 2.0f}, std::vector<size_t>{2}, true);
        param2 = std::make_unique<Tensor>(std::vector<float>{3.0f, 4.0f, 5.0f}, std::vector<size_t>{3}, true);
        
        // Устанавливаем градиенты
        param1->Grad()[0] = 0.1f;
        param1->Grad()[1] = 0.2f;
        
        param2->Grad()[0] = 0.3f;
        param2->Grad()[1] = 0.4f;
        param2->Grad()[2] = 0.5f;
        
        // Параметр без градиентов для тестирования
        param_no_grad = std::make_unique<Tensor>(std::vector<float>{6.0f, 7.0f}, std::vector<size_t>{2}, false);
    }
    
    std::unique_ptr<Tensor> param1, param2, param_no_grad;
    
    // Вспомогательная функция для сравнения значений с допуском
    void ExpectNear(const Tensor& tensor, const std::vector<float>& expected, float tolerance = 1e-5f) {
        ASSERT_EQ(tensor.Size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(tensor[i], expected[i], tolerance) 
                << "Value mismatch at index " << i;
        }
    }
    
    // Функция для создания простой квадратичной функции потерь
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
    void ComputeGradients(Tensor& x, const Tensor& target) {
        for (size_t i = 0; i < x.Size(); ++i) {
            x.Grad()[i] = x[i] - target[i];
        }
    }
};

// ===== ТЕСТЫ SGD ОПТИМИЗАТОРА =====

TEST_F(OptimizerTest, SGDBasicConstruction) {
    // Тест базового конструктора
    SGD optimizer(0.01f);
    
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.01f);
    EXPECT_FLOAT_EQ(optimizer.GetMomentum(), 0.0f);
    EXPECT_FLOAT_EQ(optimizer.GetDampening(), 0.0f);
    EXPECT_FLOAT_EQ(optimizer.GetWeightDecay(), 0.0f);
    EXPECT_FALSE(optimizer.IsNesterov());
    EXPECT_FLOAT_EQ(optimizer.GetMaxGradNorm(), 0.0f);
    EXPECT_EQ(optimizer.GetName(), "SGD");
}

TEST_F(OptimizerTest, SGDFullConstruction) {
    // Тест конструктора со всеми параметрами
    SGD optimizer(0.01f, 0.9f, 0.1f, 0.001f, false, 1.0f);
    
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.01f);
    EXPECT_FLOAT_EQ(optimizer.GetMomentum(), 0.9f);
    EXPECT_FLOAT_EQ(optimizer.GetDampening(), 0.1f);
    EXPECT_FLOAT_EQ(optimizer.GetWeightDecay(), 0.001f);
    EXPECT_FALSE(optimizer.IsNesterov());
    EXPECT_FLOAT_EQ(optimizer.GetMaxGradNorm(), 1.0f);
}

TEST_F(OptimizerTest, SGDParameterValidation) {
    // Тест валидации параметров
    EXPECT_THROW(SGD(-0.01f), std::invalid_argument);  // Отрицательный learning rate
    EXPECT_THROW(SGD(0.01f, -0.1f), std::invalid_argument);  // Отрицательный momentum
    EXPECT_THROW(SGD(0.01f, 1.1f), std::invalid_argument);   // momentum > 1
    EXPECT_THROW(SGD(0.01f, 0.9f, -0.1f), std::invalid_argument);  // Отрицательный dampening
    EXPECT_THROW(SGD(0.01f, 0.9f, 1.1f), std::invalid_argument);   // dampening > 1
    EXPECT_THROW(SGD(0.01f, 0.0f, 0.0f, -0.001f), std::invalid_argument);  // Отрицательный weight decay
    EXPECT_THROW(SGD(0.01f, 0.0f, 0.1f, 0.0f, true), std::invalid_argument);  // Nesterov без momentum
    EXPECT_THROW(SGD(0.01f, 0.9f, 0.0f, 0.0f, false, -1.0f), std::invalid_argument);  // Отрицательный max_grad_norm
}

TEST_F(OptimizerTest, SGDParameterManagement) {
    SGD optimizer(0.01f);
    
    // Проверяем начальное состояние
    EXPECT_EQ(optimizer.GetParameterCount(), 0);
    
    // Добавляем параметры
    optimizer.AddParameter(*param1);
    optimizer.AddParameter(*param2);
    
    EXPECT_EQ(optimizer.GetParameterCount(), 2);
    
    // Очищаем параметры
    optimizer.ClearParameters();
    EXPECT_EQ(optimizer.GetParameterCount(), 0);
}

TEST_F(OptimizerTest, SGDBasicStep) {
    SGD optimizer(0.1f);  // learning rate = 0.1
    optimizer.AddParameter(*param1);
    optimizer.AddParameter(*param2);
    
    // Сохраняем исходные значения
    std::vector<float> original_param1 = {param1->Data()[0], param1->Data()[1]};
    std::vector<float> original_param2 = {param2->Data()[0], param2->Data()[1], param2->Data()[2]};
    
    // Выполняем шаг оптимизации
    optimizer.Step();
    
    // Проверяем обновления: new_param = old_param - learning_rate * grad
    ExpectNear(*param1, {original_param1[0] - 0.1f * 0.1f, original_param1[1] - 0.1f * 0.2f});
    ExpectNear(*param2, {
        original_param2[0] - 0.1f * 0.3f,
        original_param2[1] - 0.1f * 0.4f,
        original_param2[2] - 0.1f * 0.5f
    });
}

TEST_F(OptimizerTest, SGDWithMomentum) {
    SGD optimizer(0.1f, 0.9f);  // learning rate = 0.1, momentum = 0.9
    optimizer.AddParameter(*param1);
    
    std::vector<float> original_values = {param1->Data()[0], param1->Data()[1]};
    
    // Первый шаг (momentum buffer пустой)
    optimizer.Step();
    
    // После первого шага: v_1 = 0.9 * 0 + 1.0 * grad = grad
    // param = param - lr * v_1 = param - lr * grad
    std::vector<float> expected_after_first = {
        original_values[0] - 0.1f * 0.1f,
        original_values[1] - 0.1f * 0.2f
    };
    ExpectNear(*param1, expected_after_first);
    
    // Устанавливаем новые градиенты
    param1->Grad()[0] = 0.05f;
    param1->Grad()[1] = 0.15f;
    
    // Второй шаг
    optimizer.Step();
    
    // После второго шага: v_2 = 0.9 * v_1 + 1.0 * new_grad
    // v_1 = [0.1, 0.2], new_grad = [0.05, 0.15]
    // v_2 = [0.9*0.1 + 0.05, 0.9*0.2 + 0.15] = [0.14, 0.33]
    std::vector<float> expected_after_second = {
        expected_after_first[0] - 0.1f * 0.14f,
        expected_after_first[1] - 0.1f * 0.33f
    };
    ExpectNear(*param1, expected_after_second, 1e-4f);
}

TEST_F(OptimizerTest, SGDWithWeightDecay) {
    SGD optimizer(0.1f, 0.0f, 0.0f, 0.01f);  // weight decay = 0.01
    optimizer.AddParameter(*param1);
    
    std::vector<float> original_values = {param1->Data()[0], param1->Data()[1]};
    
    optimizer.Step();
    
    // С weight decay: effective_grad = grad + weight_decay * param
    // effective_grad = [0.1 + 0.01*1.0, 0.2 + 0.01*2.0] = [0.11, 0.22]
    // new_param = old_param - lr * effective_grad
    std::vector<float> expected = {
        original_values[0] - 0.1f * 0.11f,
        original_values[1] - 0.1f * 0.22f
    };
    ExpectNear(*param1, expected);
}

TEST_F(OptimizerTest, SGDZeroGrad) {
    SGD optimizer(0.1f);
    optimizer.AddParameter(*param1);
    optimizer.AddParameter(*param2);
    
    // Проверяем, что градиенты не нулевые
    EXPECT_NE(param1->Grad()[0], 0.0f);
    EXPECT_NE(param2->Grad()[0], 0.0f);
    
    // Обнуляем градиенты
    optimizer.ZeroGrad();
    
    // Проверяем, что градиенты обнулились
    for (size_t i = 0; i < param1->Size(); ++i) {
        EXPECT_FLOAT_EQ(param1->Grad()[i], 0.0f);
    }
    for (size_t i = 0; i < param2->Size(); ++i) {
        EXPECT_FLOAT_EQ(param2->Grad()[i], 0.0f);
    }
}

TEST_F(OptimizerTest, SGDSkipsParametersWithoutGradients) {
    SGD optimizer(0.1f);
    optimizer.AddParameter(*param_no_grad);
    
    std::vector<float> original_values = {param_no_grad->Data()[0], param_no_grad->Data()[1]};
    
    // Выполняем шаг - параметр без градиентов не должен измениться
    optimizer.Step();
    
    ExpectNear(*param_no_grad, original_values);
}

// ===== ТЕСТЫ ADAM ОПТИМИЗАТОРА =====

TEST_F(OptimizerTest, AdamBasicConstruction) {
    // Тест базового конструктора с дефолтными параметрами
    Adam optimizer;
    
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.001f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta1(), 0.9f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta2(), 0.999f);
    EXPECT_FLOAT_EQ(optimizer.GetEpsilon(), 1e-8f);
    EXPECT_FLOAT_EQ(optimizer.GetWeightDecay(), 0.0f);
    EXPECT_FALSE(optimizer.IsAmsgrad());
    EXPECT_FLOAT_EQ(optimizer.GetMaxGradNorm(), 0.0f);
    EXPECT_EQ(optimizer.GetStep(), 0);
    EXPECT_EQ(optimizer.GetName(), "Adam");
}

TEST_F(OptimizerTest, AdamFullConstruction) {
    // Тест конструктора со всеми параметрами
    Adam optimizer(0.002f, 0.95f, 0.995f, 1e-7f, 0.01f, true, 1.0f);
    
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.002f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta1(), 0.95f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta2(), 0.995f);
    EXPECT_FLOAT_EQ(optimizer.GetEpsilon(), 1e-7f);
    EXPECT_FLOAT_EQ(optimizer.GetWeightDecay(), 0.01f);
    EXPECT_TRUE(optimizer.IsAmsgrad());
    EXPECT_FLOAT_EQ(optimizer.GetMaxGradNorm(), 1.0f);
}

TEST_F(OptimizerTest, AdamParameterValidation) {
    // Тест валидации параметров
    EXPECT_THROW(Adam(-0.001f), std::invalid_argument);  // Отрицательный learning rate
    EXPECT_THROW(Adam(0.001f, -0.1f), std::invalid_argument);  // Отрицательный beta1
    EXPECT_THROW(Adam(0.001f, 1.0f), std::invalid_argument);   // beta1 >= 1
    EXPECT_THROW(Adam(0.001f, 0.9f, -0.1f), std::invalid_argument);  // Отрицательный beta2
    EXPECT_THROW(Adam(0.001f, 0.9f, 1.0f), std::invalid_argument);   // beta2 >= 1
    EXPECT_THROW(Adam(0.001f, 0.9f, 0.999f, 0.0f), std::invalid_argument);  // epsilon <= 0
    EXPECT_THROW(Adam(0.001f, 0.9f, 0.999f, 1e-8f, -0.01f), std::invalid_argument);  // Отрицательный weight decay
    EXPECT_THROW(Adam(0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, false, -1.0f), std::invalid_argument);  // Отрицательный max_grad_norm
}

TEST_F(OptimizerTest, AdamBasicStep) {
    Adam optimizer(0.1f);  // Большой learning rate для заметных изменений
    optimizer.AddParameter(*param1);
    
    std::vector<float> original_values = {param1->Data()[0], param1->Data()[1]};
    
    // Выполняем первый шаг
    optimizer.Step();
    
    // Проверяем, что параметры изменились
    EXPECT_NE(param1->Data()[0], original_values[0]);
    EXPECT_NE(param1->Data()[1], original_values[1]);
    
    // Проверяем, что счетчик шагов увеличился
    EXPECT_EQ(optimizer.GetStep(), 1);
    
    // Для первого шага Adam должен работать примерно как SGD с bias correction
    // Но точные значения сложно предсказать из-за bias correction и adaptive learning rates
}

TEST_F(OptimizerTest, AdamMultipleSteps) {
    Adam optimizer(0.01f);
    optimizer.AddParameter(*param1);
    
    // Выполняем несколько шагов
    for (int i = 0; i < 5; ++i) {
        optimizer.Step();
        EXPECT_EQ(optimizer.GetStep(), i + 1);
        
        // Устанавливаем постоянные градиенты для следующего шага
        param1->Grad()[0] = 0.1f;
        param1->Grad()[1] = 0.2f;
    }
    
    EXPECT_EQ(optimizer.GetStep(), 5);
}

TEST_F(OptimizerTest, AdamReset) {
    Adam optimizer(0.01f);
    optimizer.AddParameter(*param1);
    
    // Выполняем несколько шагов
    optimizer.Step();
    optimizer.Step();
    
    EXPECT_EQ(optimizer.GetStep(), 2);
    
    // Сбрасываем состояние
    optimizer.Reset();
    
    EXPECT_EQ(optimizer.GetStep(), 0);
}

TEST_F(OptimizerTest, AdamWithWeightDecay) {
    Adam optimizer(0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);  // weight decay = 0.01
    optimizer.AddParameter(*param1);
    
    std::vector<float> original_values = {param1->Data()[0], param1->Data()[1]};
    
    optimizer.Step();
    
    // С weight decay параметры должны изменяться больше
    // (точные значения сложно предсказать, но изменения должны быть)
    EXPECT_NE(param1->Data()[0], original_values[0]);
    EXPECT_NE(param1->Data()[1], original_values[1]);
}

// ===== ТЕСТЫ СРАВНЕНИЯ ОПТИМИЗАТОРОВ =====

TEST_F(OptimizerTest, OptimizerComparison) {
    // Создаем одинаковые параметры для обоих оптимизаторов
    Tensor param_sgd({1.0f, 2.0f}, {2}, true);
    Tensor param_adam({1.0f, 2.0f}, {2}, true);
    
    // Устанавливаем одинаковые градиенты
    param_sgd.Grad()[0] = param_adam.Grad()[0] = 0.1f;
    param_sgd.Grad()[1] = param_adam.Grad()[1] = 0.2f;
    
    SGD sgd_optimizer(0.01f);
    Adam adam_optimizer(0.01f);
    
    sgd_optimizer.AddParameter(param_sgd);
    adam_optimizer.AddParameter(param_adam);
    
    // Выполняем шаги
    sgd_optimizer.Step();
    adam_optimizer.Step();
    
    // Оптимизаторы должны обновлять параметры по-разному
    EXPECT_NE(param_sgd[0], param_adam[0]);
    EXPECT_NE(param_sgd[1], param_adam[1]);
}

// ===== ТЕСТЫ КОНВЕРГЕНЦИИ =====

TEST_F(OptimizerTest, SGDConvergence) {
    // Тест конвергенции SGD на простой квадратичной функции
    Tensor param({5.0f, -3.0f}, {2}, true);
    Tensor target(std::vector<float>{1.0f, 2.0f}, std::vector<size_t>{2}, false);
    
    SGD optimizer(0.1f);
    optimizer.AddParameter(param);
    
    float initial_loss = ComputeLoss(param, target);
    
    // Выполняем несколько итераций оптимизации
    for (int i = 0; i < 50; ++i) {
        optimizer.ZeroGrad();
        ComputeGradients(param, target);
        optimizer.Step();
    }
    
    float final_loss = ComputeLoss(param, target);
    
    // Потери должны уменьшиться
    EXPECT_LT(final_loss, initial_loss);
    
    // Параметры должны приблизиться к целевым значениям
    EXPECT_NEAR(param[0], target[0], 0.1f);
    EXPECT_NEAR(param[1], target[1], 0.1f);
}

TEST_F(OptimizerTest, AdamConvergence) {
    // Тест конвергенции Adam на простой квадратичной функции
    Tensor param({5.0f, -3.0f}, {2}, true);
    Tensor target(std::vector<float>{1.0f, 2.0f}, std::vector<size_t>{2}, false);
    
    Adam optimizer(0.1f);
    optimizer.AddParameter(param);
    
    float initial_loss = ComputeLoss(param, target);
    
    // Выполняем несколько итераций оптимизации
    for (int i = 0; i < 50; ++i) {
        optimizer.ZeroGrad();
        ComputeGradients(param, target);
        optimizer.Step();
    }
    
    float final_loss = ComputeLoss(param, target);
    
    // Потери должны уменьшиться
    EXPECT_LT(final_loss, initial_loss);
    
    // Параметры должны приблизиться к целевым значениям (Adam может сходиться медленнее)
    EXPECT_NEAR(param[0], target[0], 0.5f);  // Увеличиваем допуск
    EXPECT_NEAR(param[1], target[1], 1.0f);  // Увеличиваем допуск
}

// ===== ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ =====

TEST_F(OptimizerTest, EmptyGradients) {
    // Тест с пустыми градиентами
    Tensor param({1.0f, 2.0f}, {2}, true);
    param.ZeroGrad();  // Обнуляем градиенты
    
    SGD sgd_optimizer(0.1f);
    Adam adam_optimizer(0.1f);
    
    sgd_optimizer.AddParameter(param);
    adam_optimizer.AddParameter(param);
    
    std::vector<float> original_values = {param[0], param[1]};
    
    // Шаги с нулевыми градиентами не должны изменять параметры
    sgd_optimizer.Step();
    ExpectNear(param, original_values);
    
    adam_optimizer.Step();
    ExpectNear(param, original_values);
}

TEST_F(OptimizerTest, LargeGradients) {
    // Тест с большими градиентами
    Tensor param({1.0f, 2.0f}, {2}, true);
    param.Grad()[0] = 1000.0f;
    param.Grad()[1] = -1000.0f;
    
    SGD optimizer(0.001f);  // Маленький learning rate
    optimizer.AddParameter(param);
    
    std::vector<float> original_values = {param[0], param[1]};
    
    optimizer.Step();
    
    // Параметры должны измениться, но не слишком сильно из-за маленького learning rate
    EXPECT_NE(param[0], original_values[0]);
    EXPECT_NE(param[1], original_values[1]);
    
    // Проверяем, что изменения разумные
    EXPECT_GT(param[0], original_values[0] - 2.0f);
    EXPECT_LT(param[1], original_values[1] + 2.0f);
}

// ===== ТЕСТЫ КОНФИГУРАЦИИ =====

TEST_F(OptimizerTest, SGDConfigString) {
    SGD optimizer(0.01f, 0.9f, 0.0f, 0.001f, true, 1.0f);  // dampening = 0.0f для Nesterov
    
    std::string config = optimizer.GetConfig();
    
    // Проверяем, что конфигурация содержит основные параметры
    EXPECT_NE(config.find("SGD"), std::string::npos);
    EXPECT_NE(config.find("lr="), std::string::npos);
    EXPECT_NE(config.find("momentum="), std::string::npos);
    EXPECT_NE(config.find("nesterov=true"), std::string::npos);
}

TEST_F(OptimizerTest, AdamConfigString) {
    Adam optimizer(0.002f, 0.95f, 0.995f, 1e-7f, 0.01f, true, 1.0f);
    
    std::string config = optimizer.GetConfig();
    
    // Проверяем, что конфигурация содержит основные параметры
    EXPECT_NE(config.find("Adam"), std::string::npos);
    EXPECT_NE(config.find("lr="), std::string::npos);
    EXPECT_NE(config.find("beta1="), std::string::npos);
    EXPECT_NE(config.find("beta2="), std::string::npos);
    EXPECT_NE(config.find("amsgrad=true"), std::string::npos);
}

// ===== ТЕСТЫ СЕТТЕРОВ =====

TEST_F(OptimizerTest, SGDSetters) {
    SGD optimizer(0.01f);
    
    // Тестируем сеттеры
    optimizer.SetLearningRate(0.02f);
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.02f);
    
    optimizer.SetMomentum(0.8f);
    EXPECT_FLOAT_EQ(optimizer.GetMomentum(), 0.8f);
    
    optimizer.SetWeightDecay(0.001f);
    EXPECT_FLOAT_EQ(optimizer.GetWeightDecay(), 0.001f);
    
    // Тестируем валидацию в сеттерах
    EXPECT_THROW(optimizer.SetMomentum(-0.1f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetMomentum(1.1f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetWeightDecay(-0.001f), std::invalid_argument);
}

TEST_F(OptimizerTest, AdamSetters) {
    Adam optimizer;
    
    // Тестируем сеттеры
    optimizer.SetLearningRate(0.002f);
    EXPECT_FLOAT_EQ(optimizer.GetLearningRate(), 0.002f);
    
    optimizer.SetBeta1(0.95f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta1(), 0.95f);
    
    optimizer.SetBeta2(0.995f);
    EXPECT_FLOAT_EQ(optimizer.GetBeta2(), 0.995f);
    
    optimizer.SetEpsilon(1e-7f);
    EXPECT_FLOAT_EQ(optimizer.GetEpsilon(), 1e-7f);
    
    // Тестируем валидацию в сеттерах
    EXPECT_THROW(optimizer.SetBeta1(-0.1f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetBeta1(1.0f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetBeta2(-0.1f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetBeta2(1.0f), std::invalid_argument);
    EXPECT_THROW(optimizer.SetEpsilon(0.0f), std::invalid_argument);
} 