# Функции Потерь KotML

## Обзор

Модуль функций потерь KotML предоставляет полный набор функций потерь для различных задач машинного обучения. Все функции потерь наследуются от базового класса `Loss` и поддерживают автоматическое дифференцирование.

## Архитектура

### Базовый класс `Loss`

```cpp
class Loss {
public:
    virtual Tensor Forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor Backward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual std::string GetName() const = 0;
    virtual std::pair<Tensor, Tensor> ForwardBackward(const Tensor& predictions, const Tensor& targets);
};
```

**Методы:**
- `Forward()` - вычисляет значение функции потерь
- `Backward()` - вычисляет градиенты по предсказаниям
- `GetName()` - возвращает название функции потерь
- `ForwardBackward()` - эффективно вычисляет и потери, и градиенты

## Реализованные Функции Потерь

### 1. MSELoss (Mean Squared Error)

**Назначение:** Регрессионные задачи

**Формула:** 
```
L = (1/n) * Σ(y_pred - y_true)²
```

**Градиент:**
```
∂L/∂y_pred = 2/n * (y_pred - y_true)
```

**Использование:**
```cpp
nn::MSELoss mse_loss;
Tensor loss = mse_loss.Forward(predictions, targets);
Tensor gradients = mse_loss.Backward(predictions, targets);

// Или через convenience функцию
Tensor loss = nn::loss::MSE(predictions, targets);
```

**Особенности:**
- Чувствительна к выбросам (квадратичная функция)
- Дифференцируема везде
- Подходит для непрерывных целевых значений

### 2. MAELoss (Mean Absolute Error)

**Назначение:** Регрессионные задачи, устойчивые к выбросам

**Формула:**
```
L = (1/n) * Σ|y_pred - y_true|
```

**Градиент:**
```
∂L/∂y_pred = sign(y_pred - y_true) / n
```

**Использование:**
```cpp
nn::MAELoss mae_loss;
Tensor loss = mae_loss.Forward(predictions, targets);
Tensor gradients = mae_loss.Backward(predictions, targets);
```

**Особенности:**
- Более устойчива к выбросам чем MSE
- Недифференцируема в точке 0 (используется субградиент)
- Линейная функция потерь

### 3. BCELoss (Binary Cross Entropy)

**Назначение:** Бинарная классификация

**Формула:**
```
L = -(1/n) * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
```

**Градиент:**
```
∂L/∂y_pred = (y_pred - y_true) / [n * y_pred * (1 - y_pred)]
```

**Использование:**
```cpp
nn::BCELoss bce_loss(1e-7f); // epsilon для предотвращения log(0)
Tensor loss = bce_loss.Forward(predictions, targets);
Tensor gradients = bce_loss.Backward(predictions, targets);

// Настройка epsilon
bce_loss.SetEpsilon(1e-8f);
```

**Требования:**
- Предсказания должны быть вероятностями (0-1)
- Целевые значения должны быть бинарными (0 или 1)
- Использует epsilon для численной стабильности

### 4. CrossEntropyLoss (Categorical Cross Entropy)

**Назначение:** Многоклассовая классификация

**Формула:**
```
L = -(1/n) * Σ Σ y_true[i,j] * log(y_pred[i,j])
```

**Градиент:**
```
∂L/∂y_pred = -y_true / (n * y_pred)
```

**Использование:**
```cpp
nn::CrossEntropyLoss ce_loss(1e-7f);
Tensor loss = ce_loss.Forward(predictions, targets);
Tensor gradients = ce_loss.Backward(predictions, targets);
```

**Требования:**
- Предсказания должны быть вероятностями (softmax выход)
- Целевые значения должны быть one-hot encoded
- Поддерживает как одиночные образцы (1D), так и батчи (2D)

### 5. HuberLoss (Smooth L1 Loss)

**Назначение:** Регрессия с устойчивостью к выбросам

**Формула:**
```
L = { 0.5 * (y_pred - y_true)²                    если |y_pred - y_true| ≤ δ
    { δ * |y_pred - y_true| - 0.5 * δ²           иначе
```

**Градиент:**
```
∂L/∂y_pred = { (y_pred - y_true) / n             если |y_pred - y_true| ≤ δ
             { δ * sign(y_pred - y_true) / n      иначе
```

**Использование:**
```cpp
nn::HuberLoss huber_loss(1.0f); // delta = 1.0
Tensor loss = huber_loss.Forward(predictions, targets);
Tensor gradients = huber_loss.Backward(predictions, targets);

// Изменение параметра delta
huber_loss.SetDelta(0.5f);
```

**Особенности:**
- Комбинирует преимущества MSE и MAE
- Квадратичная для малых ошибок, линейная для больших
- Параметр δ (delta) контролирует точку перехода

## Convenience Functions

Для быстрого использования доступны функции в пространстве имен `nn::loss`:

```cpp
// Быстрое вычисление потерь без создания объектов
Tensor mse_loss = nn::loss::MSE(predictions, targets);
Tensor mae_loss = nn::loss::MAE(predictions, targets);
Tensor bce_loss = nn::loss::BCE(predictions, targets, 1e-7f);
Tensor ce_loss = nn::loss::CrossEntropy(predictions, targets, 1e-7f);
Tensor huber_loss = nn::loss::Huber(predictions, targets, 1.0f);
```

## Сравнение Функций Потерь

### Устойчивость к выбросам (по возрастанию):
1. **MSE** - очень чувствительна к выбросам
2. **Huber** - умеренная устойчивость
3. **MAE** - высокая устойчивость

### Дифференцируемость:
- **MSE, BCE, CrossEntropy** - дифференцируемы везде
- **Huber** - дифференцируема везде
- **MAE** - недифференцируема в точке 0

### Скорость сходимости:
- **MSE** - быстрая сходимость вблизи минимума
- **MAE** - медленная сходимость
- **Huber** - компромисс между MSE и MAE

## Примеры Использования

### Регрессия
```cpp
// Создание данных
Tensor predictions = /* ваши предсказания */;
Tensor targets = /* истинные значения */;

// MSE для обычной регрессии
nn::MSELoss mse;
auto [loss, gradients] = mse.ForwardBackward(predictions, targets);

// Huber для устойчивой регрессии
nn::HuberLoss huber(1.0f);
auto huber_loss = huber.Forward(predictions, targets);
```

### Бинарная классификация
```cpp
// Предсказания должны быть вероятностями
Tensor predictions = /* sigmoid выход [0,1] */;
Tensor targets = /* бинарные метки {0,1} */;

nn::BCELoss bce;
auto loss = bce.Forward(predictions, targets);
auto gradients = bce.Backward(predictions, targets);
```

### Многоклассовая классификация
```cpp
// Батч из 32 образцов, 10 классов
Tensor predictions = /* softmax выход [32, 10] */;
Tensor targets = /* one-hot метки [32, 10] */;

nn::CrossEntropyLoss ce;
auto loss = ce.Forward(predictions, targets);
auto gradients = ce.Backward(predictions, targets);
```

## Обработка Ошибок

Все функции потерь выполняют валидацию входных данных:

```cpp
try {
    nn::MSELoss mse;
    mse.Forward(predictions, wrong_shape_targets);
} catch (const std::invalid_argument& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // "Predictions and targets must have the same shape"
}
```

**Типичные ошибки:**
- Несовпадение размерностей предсказаний и целей
- Пустые тензоры
- Неверные значения для BCE (не бинарные)
- Отрицательный параметр delta для Huber Loss

## Интеграция с Оптимизаторами

Функции потерь легко интегрируются с оптимизаторами:

```cpp
// Создание модели и оптимизатора
auto model = /* ваша модель */;
nn::MSELoss loss_fn;
optim::SGD optimizer(model.GetParameters(), 0.01f);

// Цикл обучения
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto predictions = model.Forward(inputs);
    auto [loss, gradients] = loss_fn.ForwardBackward(predictions, targets);
    
    // Обратное распространение через модель
    model.Backward(gradients);
    
    // Обновление параметров
    optimizer.Step();
    optimizer.ZeroGrad();
}
```

## Рекомендации по Выбору

### Для регрессии:
- **MSE** - стандартный выбор для большинства задач
- **MAE** - при наличии выбросов в данных
- **Huber** - компромисс между MSE и MAE

### Для классификации:
- **BCE** - бинарная классификация
- **CrossEntropy** - многоклассовая классификация

### Настройка параметров:
- **BCE/CrossEntropy epsilon** - обычно 1e-7 или 1e-8
- **Huber delta** - зависит от масштаба данных, обычно 1.0

## Производительность

Все функции потерь оптимизированы для:
- Минимального использования памяти
- Эффективного вычисления градиентов
- Поддержки батчевой обработки
- RAII управления ресурсами

## Расширение

Для добавления новой функции потерь:

1. Наследуйтесь от класса `Loss`
2. Реализуйте методы `Forward()`, `Backward()`, `GetName()`
3. Добавьте валидацию входных данных
4. Опционально добавьте convenience функцию

```cpp
class CustomLoss : public Loss {
public:
    Tensor Forward(const Tensor& predictions, const Tensor& targets) override {
        // Ваша реализация
    }
    
    Tensor Backward(const Tensor& predictions, const Tensor& targets) override {
        // Ваша реализация градиентов
    }
    
    std::string GetName() const override { return "CustomLoss"; }
};
``` 