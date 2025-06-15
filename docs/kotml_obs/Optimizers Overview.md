# Optimizers Overview

Обзор системы оптимизаторов в KotML для обучения нейронных сетей.

## Введение

Оптимизаторы в KotML отвечают за обновление параметров модели на основе вычисленных градиентов. Они реализуют различные алгоритмы оптимизации для эффективного обучения нейронных сетей.

## Архитектура системы

### Базовый класс Optimizer

```cpp
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void Step() = 0;                    // Шаг оптимизации
    virtual void ZeroGrad() = 0;                // Обнуление градиентов
    virtual void SetLearningRate(float lr) = 0; // Установка learning rate
    virtual float GetLearningRate() const = 0;  // Получение learning rate
    virtual std::string GetName() const = 0;    // Имя оптимизатора
};
```

### Доступные оптимизаторы

- **SGD** - Стохастический градиентный спуск
- **Adam** - Adaptive Moment Estimation (планируется)

## SGD (Stochastic Gradient Descent)

Базовый алгоритм стохастического градиентного спуска.

### Алгоритм

```
θ = θ - α * ∇θ
```

Где:
- `θ` - параметры модели
- `α` - learning rate (скорость обучения)
- `∇θ` - градиенты параметров

### Конструктор

```cpp
SGD(std::vector<Tensor*> parameters, float learningRate);
```

### Параметры
- `parameters` - указатели на обучаемые параметры
- `learningRate` - скорость обучения (обычно 0.001 - 0.1)

### Использование

```cpp
#include "kotml/optim/sgd.hpp"
using namespace kotml;

// Создание модели
auto network = Sequential()
    .Input(784)
    .Linear(128)
    .ReLU()
    .Linear(10)
    .Build();

// Получение параметров
auto parameters = network.GetParameters();

// Создание оптимизатора
SGD optimizer(parameters, 0.01f);  // learning rate = 0.01

// Цикл обучения
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : train_data) {
        // Обнуление градиентов
        optimizer.ZeroGrad();
        
        // Прямой проход
        Tensor output = network.Forward(batch.input);
        
        // Вычисление функции потерь
        Tensor loss = loss_function(output, batch.target);
        
        // Обратное распространение
        loss.Backward();
        
        // Обновление параметров
        optimizer.Step();
    }
}
```

### Настройка learning rate

```cpp
SGD optimizer(parameters, 0.1f);

// Изменение learning rate в процессе обучения
optimizer.SetLearningRate(0.01f);  // Уменьшение для стабилизации

// Получение текущего значения
float current_lr = optimizer.GetLearningRate();
```

## Adam (планируется)

Адаптивный алгоритм оптимизации с моментом.

### Алгоритм

```
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇θ
v_t = β₂ * v_{t-1} + (1 - β₂) * (∇θ)²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ = θ - α * m̂_t / (√v̂_t + ε)
```

### Планируемый интерфейс

```cpp
// Будущая реализация
Adam optimizer(parameters, 0.001f, 0.9f, 0.999f, 1e-8f);
//                lr      β₁    β₂     ε
```

## Практические примеры

### Базовое обучение

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

// Подготовка данных
Tensor X_train = Tensor::Randn({1000, 784});
Tensor y_train = Tensor::Randn({1000, 10});

// Создание модели
auto model = Sequential()
    .Input(784)
    .Linear(256)
    .ReLU()
    .Dropout(0.2f)
    .Linear(128)
    .ReLU()
    .Output(10)
    .Build();

// Настройка оптимизатора
auto parameters = model.GetParameters();
SGD optimizer(parameters, 0.01f);

// Параметры обучения
int batch_size = 32;
int num_epochs = 100;

// Цикл обучения
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int num_batches = 0;
    
    // Обучение по батчам
    for (int i = 0; i < X_train.Shape()[0]; i += batch_size) {
        // Извлечение батча
        int end_idx = std::min(i + batch_size, (int)X_train.Shape()[0]);
        Tensor X_batch = X_train.Slice(i, end_idx);
        Tensor y_batch = y_train.Slice(i, end_idx);
        
        // Обнуление градиентов
        optimizer.ZeroGrad();
        
        // Прямой проход
        model.SetTraining(true);
        Tensor predictions = model.Forward(X_batch);
        
        // Вычисление потерь
        Tensor loss = MeanSquaredError(predictions, y_batch);
        epoch_loss += loss.Item();
        
        // Обратное распространение
        loss.Backward();
        
        // Обновление параметров
        optimizer.Step();
        
        num_batches++;
    }
    
    // Вывод прогресса
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch 
                  << ", Loss: " << epoch_loss / num_batches << std::endl;
    }
}
```

### Адаптивное изменение learning rate

```cpp
SGD optimizer(parameters, 0.1f);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Уменьшение learning rate каждые 30 эпох
    if (epoch % 30 == 0 && epoch > 0) {
        float current_lr = optimizer.GetLearningRate();
        optimizer.SetLearningRate(current_lr * 0.5f);
        std::cout << "Learning rate reduced to: " 
                  << optimizer.GetLearningRate() << std::endl;
    }
    
    // Обучение...
}
```

### Валидация и ранняя остановка

```cpp
SGD optimizer(parameters, 0.01f);
float best_val_loss = std::numeric_limits<float>::max();
int patience = 10;
int patience_counter = 0;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Обучение
    model.SetTraining(true);
    float train_loss = train_epoch(model, optimizer, train_data);
    
    // Валидация
    model.SetTraining(false);
    float val_loss = validate(model, val_data);
    
    // Ранняя остановка
    if (val_loss < best_val_loss) {
        best_val_loss = val_loss;
        patience_counter = 0;
        // Сохранение лучшей модели
    } else {
        patience_counter++;
        if (patience_counter >= patience) {
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            break;
        }
    }
    
    std::cout << "Epoch " << epoch 
              << ", Train Loss: " << train_loss
              << ", Val Loss: " << val_loss << std::endl;
}
```

## Рекомендации по выбору

### SGD

**Преимущества:**
- ✅ Простой и надежный
- ✅ Хорошо работает для многих задач
- ✅ Низкое потребление памяти
- ✅ Стабильная сходимость

**Недостатки:**
- ❌ Может медленно сходиться
- ❌ Чувствителен к выбору learning rate
- ❌ Может застревать в локальных минимумах

**Рекомендуется для:**
- Простых задач
- Ограниченной памяти
- Стабильного обучения

### Adam (когда будет реализован)

**Преимущества:**
- ✅ Адаптивный learning rate
- ✅ Быстрая сходимость
- ✅ Хорошо работает "из коробки"
- ✅ Подходит для разреженных градиентов

**Недостатки:**
- ❌ Больше потребление памяти
- ❌ Может переобучаться
- ❌ Сложнее в настройке

## Настройка гиперпараметров

### Learning Rate

```cpp
// Слишком большой - нестабильное обучение
SGD optimizer1(parameters, 1.0f);    // Может расходиться

// Оптимальный диапазон
SGD optimizer2(parameters, 0.01f);   // Хорошая отправная точка
SGD optimizer3(parameters, 0.001f);  // Для тонкой настройки

// Слишком маленький - медленное обучение
SGD optimizer4(parameters, 0.00001f); // Очень медленно
```

### Стратегии изменения learning rate

```cpp
// Экспоненциальное затухание
float initial_lr = 0.1f;
float decay_rate = 0.95f;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float current_lr = initial_lr * std::pow(decay_rate, epoch);
    optimizer.SetLearningRate(current_lr);
    // Обучение...
}

// Ступенчатое затухание
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    if (epoch == 50 || epoch == 80) {
        float current_lr = optimizer.GetLearningRate();
        optimizer.SetLearningRate(current_lr * 0.1f);
    }
    // Обучение...
}
```

## Отладка и мониторинг

### Отслеживание градиентов

```cpp
void CheckGradients(const std::vector<Tensor*>& parameters) {
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& grad = parameters[i]->Grad();
        float grad_norm = 0.0f;
        
        for (size_t j = 0; j < grad.size(); ++j) {
            grad_norm += grad[j] * grad[j];
        }
        grad_norm = std::sqrt(grad_norm);
        
        std::cout << "Parameter " << i << " gradient norm: " << grad_norm << std::endl;
        
        if (grad_norm < 1e-8) {
            std::cout << "Warning: Very small gradients!" << std::endl;
        }
        if (grad_norm > 10.0) {
            std::cout << "Warning: Very large gradients!" << std::endl;
        }
    }
}

// Использование в цикле обучения
loss.Backward();
CheckGradients(parameters);
optimizer.Step();
```

### Мониторинг обучения

```cpp
class TrainingMonitor {
private:
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    
public:
    void LogEpoch(float train_loss, float val_loss) {
        train_losses.push_back(train_loss);
        val_losses.push_back(val_loss);
    }
    
    void PrintProgress(int epoch) {
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                      << ", Train: " << train_losses.back()
                      << ", Val: " << val_losses.back() << std::endl;
        }
    }
    
    bool IsConverged(int patience = 10) {
        if (val_losses.size() < patience) return false;
        
        float recent_avg = 0.0f;
        for (int i = val_losses.size() - patience; i < val_losses.size(); ++i) {
            recent_avg += val_losses[i];
        }
        recent_avg /= patience;
        
        return std::abs(val_losses.back() - recent_avg) < 1e-6;
    }
};
```

## Связанные страницы

- [[SGD]] - Подробно о стохастическом градиентном спуске
- [[Adam]] - Адаптивная оптимизация (планируется)
- [[Training Loop]] - Организация цикла обучения
- [[Learning Rate Scheduling]] - Стратегии изменения learning rate

---

**См. также:** [[Hyperparameter Tuning]], [[Convergence Analysis]], [[Optimization Theory]] 