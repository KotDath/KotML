# Activations

Подробное описание функций активации в KotML и их применения.

## Обзор

Функции активации вводят нелинейность в нейронные сети, позволяя им изучать сложные паттерны. KotML предоставляет несколько основных функций активации.

## Доступные функции

### ReLU (Rectified Linear Unit)

**Формула:** `f(x) = max(0, x)`

**Производная:** 
- `f'(x) = 1` если `x > 0`
- `f'(x) = 0` если `x ≤ 0`

```cpp
// Использование ReLU
Tensor input = Tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
Tensor output = activations::Relu(input);
// Результат: {0.0f, 0.0f, 0.0f, 1.0f, 2.0f}

// В слое
ActivationLayer relu_layer(ActivationType::Relu);
Tensor activated = relu_layer.Forward(input);
```

**Характеристики:**
- ✅ Простая и быстрая вычислительно
- ✅ Решает проблему затухающих градиентов
- ✅ Разреженная активация (многие нейроны = 0)
- ❌ Проблема "мертвых нейронов" (могут навсегда стать 0)

**Применение:**
- Скрытые слои в глубоких сетях
- Сверточные нейронные сети
- Большинство современных архитектур

### Sigmoid

**Формула:** `f(x) = 1 / (1 + e^(-x))`

**Производная:** `f'(x) = f(x) * (1 - f(x))`

```cpp
// Использование Sigmoid
Tensor input = Tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
Tensor output = activations::Sigmoid(input);
// Результат: {0.119f, 0.269f, 0.5f, 0.731f, 0.881f}

// В слое
ActivationLayer sigmoid_layer(ActivationType::Sigmoid);
Tensor activated = sigmoid_layer.Forward(input);
```

**Характеристики:**
- ✅ Выход в диапазоне (0, 1)
- ✅ Гладкая и дифференцируемая
- ✅ Интерпретируется как вероятность
- ❌ Проблема затухающих градиентов
- ❌ Не центрирована относительно нуля

**Применение:**
- Выходной слой для бинарной классификации
- Гейты в LSTM/GRU (когда будут реализованы)
- Задачи, где нужен выход как вероятность

### Tanh (Гиперболический тангенс)

**Формула:** `f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Производная:** `f'(x) = 1 - tanh²(x)`

```cpp
// Использование Tanh
Tensor input = Tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
Tensor output = activations::Tanh(input);
// Результат: {-0.964f, -0.762f, 0.0f, 0.762f, 0.964f}

// В слое
ActivationLayer tanh_layer(ActivationType::Tanh);
Tensor activated = tanh_layer.Forward(input);
```

**Характеристики:**
- ✅ Выход в диапазоне (-1, 1)
- ✅ Центрирована относительно нуля
- ✅ Гладкая и дифференцируемая
- ❌ Проблема затухающих градиентов (меньше чем у sigmoid)

**Применение:**
- Скрытые слои (альтернатива ReLU)
- Рекуррентные нейронные сети
- Когда нужен центрированный выход

## Сравнение функций активации

| Функция | Диапазон | Вычислительная сложность | Градиенты | Лучшее применение |
|---------|----------|-------------------------|-----------|-------------------|
| ReLU | [0, +∞) | Очень низкая | Не затухают | Скрытые слои |
| Sigmoid | (0, 1) | Средняя | Затухают | Бинарная классификация |
| Tanh | (-1, 1) | Средняя | Затухают | Скрытые слои, RNN |

## Использование в коде

### Прямое применение

```cpp
#include "kotml/nn/activations.hpp"
using namespace kotml;

Tensor input = Tensor::Randn({32, 128});

// Прямое применение функций
Tensor relu_out = activations::Relu(input);
Tensor sigmoid_out = activations::Sigmoid(input);
Tensor tanh_out = activations::Tanh(input);

// Применение по типу
Tensor activated = ApplyActivation(input, ActivationType::Relu);
```

### В слоях

```cpp
// Создание слоев активации
ActivationLayer relu(ActivationType::Relu);
ActivationLayer sigmoid(ActivationType::Sigmoid);
ActivationLayer tanh(ActivationType::Tanh);

// Использование в сети
auto network = Sequential()
    .Input(784)
    .Linear(256)
    .ReLU()              // Эквивалентно .Add(ActivationLayer(ActivationType::Relu))
    .Linear(128)
    .Sigmoid()           // Эквивалентно .Add(ActivationLayer(ActivationType::Sigmoid))
    .Linear(10)
    .Build();
```

### В выходных слоях

```cpp
// Выходной слой с активацией
OutputLayer binary_output(128, 1, ActivationType::Sigmoid);
OutputLayer multiclass_output(128, 10, ActivationType::None);  // Логиты
OutputLayer regression_output(64, 1, ActivationType::None);
```

## Производные функций

KotML предоставляет функции для вычисления производных (для ручной реализации обратного распространения):

```cpp
// Производные функций активации
Tensor input = Tensor::Randn({10});

Tensor relu_grad = activations::ReluDerivative(input);
Tensor sigmoid_grad = activations::SigmoidDerivative(input);
Tensor tanh_grad = activations::TanhDerivative(input);
```

## Рекомендации по выбору

### Для скрытых слоев

```cpp
// Рекомендуется: ReLU
auto hidden_network = Sequential()
    .Input(784)
    .Linear(512)
    .ReLU()        // Быстро, эффективно
    .Linear(256)
    .ReLU()        // Хорошо для глубоких сетей
    .Linear(128)
    .ReLU()
    .Build();
```

### Для выходных слоев

```cpp
// Бинарная классификация
OutputLayer binary_classifier(128, 1, ActivationType::Sigmoid);

// Многоклассовая классификация (с softmax в функции потерь)
OutputLayer multiclass_classifier(128, 10, ActivationType::None);

// Регрессия
OutputLayer regressor(64, 1, ActivationType::None);
```

### Специальные случаи

```cpp
// Если ReLU вызывает проблему "мертвых нейронов"
auto alternative_network = Sequential()
    .Input(784)
    .Linear(256)
    .Tanh()        // Альтернатива ReLU
    .Linear(128)
    .Tanh()
    .Output(10)
    .Build();
```

## Производительность

### Сравнение скорости

```cpp
// Тест производительности (псевдокод)
Tensor large_input = Tensor::Randn({1000, 1000});

// ReLU - самая быстрая
auto start = std::chrono::high_resolution_clock::now();
Tensor relu_result = activations::Relu(large_input);
auto relu_time = std::chrono::high_resolution_clock::now() - start;

// Sigmoid - медленнее из-за экспоненты
start = std::chrono::high_resolution_clock::now();
Tensor sigmoid_result = activations::Sigmoid(large_input);
auto sigmoid_time = std::chrono::high_resolution_clock::now() - start;

// Tanh - также медленнее
start = std::chrono::high_resolution_clock::now();
Tensor tanh_result = activations::Tanh(large_input);
auto tanh_time = std::chrono::high_resolution_clock::now() - start;
```

### Оптимизация

> [!tip] Советы по производительности
> - Используйте ReLU для максимальной скорости
> - Избегайте частого переключения между типами активации
> - Рассмотрите векторизацию для больших тензоров

## Проблемы и решения

### Проблема затухающих градиентов

```cpp
// Проблема: глубокая сеть с sigmoid
auto problematic_network = Sequential()
    .Input(784)
    .Linear(512).Sigmoid()  // Градиенты затухают
    .Linear(256).Sigmoid()  // Еще больше затухают
    .Linear(128).Sigmoid()  // Почти исчезают
    .Linear(64).Sigmoid()
    .Output(10)
    .Build();

// Решение: использование ReLU
auto better_network = Sequential()
    .Input(784)
    .Linear(512).ReLU()     // Градиенты не затухают
    .Linear(256).ReLU()
    .Linear(128).ReLU()
    .Linear(64).ReLU()
    .Output(10)
    .Build();
```

### Проблема "мертвых нейронов" ReLU

```cpp
// Если много нейронов становятся неактивными
// Рассмотрите альтернативы:

// 1. Tanh вместо ReLU
.Linear(256).Tanh()

// 2. Уменьшение learning rate
// 3. Лучшая инициализация весов
// 4. Batch normalization (когда будет реализован)
```

## Будущие расширения

> [!info] Планируемые функции активации
> - Leaky ReLU
> - ELU (Exponential Linear Unit)
> - Swish
> - GELU (Gaussian Error Linear Unit)
> - Softmax (для многоклассовой классификации)

## Связанные страницы

- [[Layers]] - Использование в слоях
- [[Neural Networks Overview]] - Роль в нейронных сетях
- [[Loss Functions]] - Взаимодействие с функциями потерь
- [[Backpropagation]] - Роль в обратном распространении

---

**См. также:** [[Gradient Flow]], [[Deep Networks]], [[Activation Patterns]] 