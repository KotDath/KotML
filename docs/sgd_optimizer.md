# SGD Оптимизатор

## Обзор

SGD (Stochastic Gradient Descent) - это фундаментальный алгоритм оптимизации для обучения нейронных сетей. Наша реализация поддерживает стандартный SGD, SGD с моментумом, Nesterov моментум и L2 регуляризацию (weight decay).

## Математическая Формула

Базовое правило обновления SGD:

```
θ = θ - α * ∇θ
```

где:
- `θ` - параметры модели
- `α` - скорость обучения (learning rate)
- `∇θ` - градиенты параметров

### SGD с Моментумом

```
v_t = β * v_{t-1} + (1 - d) * ∇θ_t
θ_t = θ_{t-1} - α * v_t
```

где:
- `v_t` - вектор скорости (momentum buffer)
- `β` - коэффициент моментума (0-1)
- `d` - коэффициент затухания (dampening)

### Nesterov Моментум

```
v_t = β * v_{t-1} + ∇θ_t
θ_t = θ_{t-1} - α * (∇θ_t + β * v_t)
```

### Weight Decay (L2 регуляризация)

```
∇θ_t = ∇θ_t + λ * θ_t
```

где `λ` - коэффициент weight decay.

## Использование

### Базовый SGD

```cpp
#include "kotml/optim/sgd.hpp"

// Создание оптимизатора
kotml::optim::SGD optimizer(0.01f);  // learning rate = 0.01

// Добавление параметров из модели
optimizer.AddParameters(model);

// Цикл обучения
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Прямой проход
    auto output = model.Forward(input);
    auto loss = loss_function(output, target);
    
    // Обратный проход
    optimizer.ZeroGrad();
    loss.Backward();
    
    // Обновление параметров
    optimizer.Step();
}
```

### SGD с Моментумом

```cpp
// SGD с моментумом 0.9
kotml::optim::SGD optimizer(0.01f, 0.9f);
optimizer.AddParameters(model);

// Остальной код обучения аналогичен
```

### SGD с Weight Decay

```cpp
// SGD с моментумом и weight decay
kotml::optim::SGD optimizer(0.01f, 0.9f, 0.0f, 0.0001f);
optimizer.AddParameters(model);
```

### Nesterov SGD

```cpp
// Nesterov SGD
kotml::optim::SGD optimizer(0.01f, 0.9f, 0.0f, 0.0f, true);
optimizer.AddParameters(model);
```

## Конструктор

```cpp
SGD(float learningRate, 
    float momentum = 0.0f,
    float dampening = 0.0f, 
    float weightDecay = 0.0f,
    bool nesterov = false)
```

### Параметры

- **learningRate** - Скорость обучения (должна быть > 0)
- **momentum** - Коэффициент моментума (0-1), по умолчанию 0
- **dampening** - Коэффициент затухания (0-1), по умолчанию 0
- **weightDecay** - Коэффициент L2 регуляризации (≥ 0), по умолчанию 0
- **nesterov** - Включить Nesterov моментум, по умолчанию false

### Ограничения

- Nesterov моментум требует `momentum > 0` и `dampening = 0`
- Все численные параметры должны быть в допустимых диапазонах

## Методы

### Управление Параметрами

```cpp
// Добавить параметры из модуля
void AddParameters(nn::Module& module);

// Добавить отдельный параметр
void AddParameter(Tensor& parameter);

// Очистить все параметры
void ClearParameters();

// Получить количество параметров
size_t GetParameterCount() const;
```

### Оптимизация

```cpp
// Обнулить градиенты
void ZeroGrad();

// Выполнить шаг оптимизации
void Step();
```

### Управление Состоянием

```cpp
// Получить/установить скорость обучения
float GetLearningRate() const;
void SetLearningRate(float learningRate);

// Получить/установить моментум
float GetMomentum() const;
void SetMomentum(float momentum);

// Получить/установить weight decay
float GetWeightDecay() const;
void SetWeightDecay(float weightDecay);

// Очистить буферы моментума
void ClearMomentumBuffers();

// Получить конфигурацию в виде строки
std::string GetConfig() const;
```

## Примеры Использования

### Простое Обучение

```cpp
#include "kotml/kotml.hpp"
#include "kotml/optim/sgd.hpp"

// Создание модели
auto model = kotml::nn::Sequential()
    .Linear(784, 128)
    .ReLU()
    .Linear(128, 10)
    .Build();

// Создание оптимизатора
kotml::optim::SGD optimizer(0.01f);
optimizer.AddParameters(model);

// Обучение
for (int epoch = 0; epoch < 100; ++epoch) {
    for (auto& batch : dataloader) {
        // Прямой проход
        auto output = model.Forward(batch.input);
        auto loss = MSELoss(output, batch.target);
        
        // Обратный проход
        optimizer.ZeroGrad();
        loss.Backward();
        optimizer.Step();
    }
}
```

### Динамическое Изменение Learning Rate

```cpp
kotml::optim::SGD optimizer(0.1f, 0.9f);  // Начальный lr = 0.1
optimizer.AddParameters(model);

for (int epoch = 0; epoch < 100; ++epoch) {
    // Уменьшаем learning rate каждые 30 эпох
    if (epoch % 30 == 0 && epoch > 0) {
        float new_lr = optimizer.GetLearningRate() * 0.1f;
        optimizer.SetLearningRate(new_lr);
        std::cout << "New learning rate: " << new_lr << std::endl;
    }
    
    // Обучение...
}
```

### Различные Конфигурации SGD

```cpp
// Базовый SGD
kotml::optim::SGD sgd_basic(0.01f);

// SGD с моментумом
kotml::optim::SGD sgd_momentum(0.01f, 0.9f);

// SGD с weight decay
kotml::optim::SGD sgd_wd(0.01f, 0.9f, 0.0f, 0.0001f);

// Nesterov SGD
kotml::optim::SGD sgd_nesterov(0.01f, 0.9f, 0.0f, 0.0f, true);

// Вывод конфигурации
std::cout << "Config: " << sgd_nesterov.GetConfig() << std::endl;
// Вывод: SGD(lr=0.010000, momentum=0.900000, nesterov=true)
```

## Рекомендации по Использованию

### Выбор Learning Rate

- **Слишком большой**: Обучение может расходиться
- **Слишком маленький**: Медленная сходимость
- **Рекомендуемые значения**: 0.1, 0.01, 0.001

### Выбор Momentum

- **0.0**: Стандартный SGD
- **0.9**: Популярное значение для большинства задач
- **0.99**: Для задач с шумными градиентами

### Weight Decay

- **Обычные значения**: 1e-4, 1e-5, 1e-6
- **Помогает предотвратить переобучение**
- **Особенно полезен для больших моделей**

## Особенности Реализации

### Эффективность Памяти

- Буферы моментума создаются только при необходимости
- Автоматическое управление памятью через RAII
- Использование unordered_map для быстрого доступа к буферам

### Безопасность Типов

- Строгая валидация параметров в конструкторе
- Проверка consistency для Nesterov momentum
- Информативные сообщения об ошибках

### Интеграция с Архитектурой

- Наследование от базового класса `Optimizer`
- Совместимость со всеми типами модулей (`Module`)
- Поддержка автоматического сбора параметров

## Производительность

### Оптимизации

- Эффективные тензорные операции
- Минимизация копирования данных
- Поддержка batch обработки

### Сравнение с Другими Оптимизаторами

| Оптимизатор | Память | Скорость | Сходимость |
|-------------|---------|----------|------------|
| SGD         | Низкая  | Высокая  | Хорошая    |
| SGD+Momentum| Средняя | Высокая  | Лучше      |
| Adam        | Высокая | Средняя  | Отличная   |

## Устранение Неполадок

### Типичные Проблемы

1. **Взрывающиеся градиенты**
   - Уменьшите learning rate
   - Используйте gradient clipping

2. **Медленная сходимость**
   - Увеличьте learning rate
   - Добавьте momentum
   - Используйте learning rate scheduling

3. **Переобучение**
   - Добавьте weight decay
   - Уменьшите learning rate
   - Используйте dropout

### Отладка

```cpp
// Мониторинг градиентов
auto params = model.Parameters();
float grad_norm = 0.0f;
for (auto* param : params) {
    for (float g : param->Grad()) {
        grad_norm += g * g;
    }
}
grad_norm = std::sqrt(grad_norm);
std::cout << "Gradient norm: " << grad_norm << std::endl;

// Мониторинг параметров
float param_norm = 0.0f;
for (auto* param : params) {
    for (float p : param->Data()) {
        param_norm += p * p;
    }
}
param_norm = std::sqrt(param_norm);
std::cout << "Parameter norm: " << param_norm << std::endl;
```

## Заключение

SGD оптимизатор KotML предоставляет полную и эффективную реализацию классического алгоритма оптимизации с современными улучшениями. Благодаря гибкой архитектуре и строгой валидации параметров, он подходит как для экспериментов, так и для продакшн приложений. 