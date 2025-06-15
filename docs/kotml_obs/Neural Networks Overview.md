# Neural Networks Overview

KotML предоставляет модульную систему для построения нейронных сетей с различными типами слоев и архитектур.

## Архитектура системы

### Базовый класс Module
Все компоненты нейронных сетей наследуются от базового класса `Module`, который определяет общий интерфейс.

### Типы слоев
- **InputLayer** - Входной слой с валидацией
- **LinearLayer** - Полносвязный слой
- **ActivationLayer** - Слой функций активации
- **DropoutLayer** - Слой регуляризации
- **OutputLayer** - Выходной слой (Linear + Activation)

### Строители сетей
- **FFN** - Простые прямонаправленные сети
- **Sequential** - Гибкий строитель для сложных архитектур

## Основные классы

### Module (базовый класс)

```cpp
class Module {
public:
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual size_t CountParameters() const = 0;
    virtual void SetTraining(bool training) = 0;
    virtual bool IsTraining() const = 0;
    virtual void ZeroGrad() = 0;
    virtual std::string Summary() const = 0;
};
```

### InputLayer

```cpp
class InputLayer : public Module {
public:
    InputLayer(size_t inputSize);
    Tensor Forward(const Tensor& input) override;
    // Валидация размерности входа
};
```

### LinearLayer

```cpp
class LinearLayer : public Module {
public:
    LinearLayer(size_t inputSize, size_t outputSize, bool bias = true);
    Tensor Forward(const Tensor& input) override;
    
private:
    Tensor m_weights;  // Веса [inputSize, outputSize]
    Tensor m_bias;     // Смещения [outputSize]
};
```

### ActivationLayer

```cpp
class ActivationLayer : public Module {
public:
    ActivationLayer(ActivationType type);
    Tensor Forward(const Tensor& input) override;
    
private:
    ActivationType m_activationType;
};
```

## Построение сетей

### Простые сети (FFN)

```cpp
// Создание простой прямонаправленной сети
FFN network(784);  // Входной размер

network.AddLayer(128, ActivationType::Relu)
       .AddLayer(64, ActivationType::Relu)
       .AddLayer(10, ActivationType::None);  // Выходной слой

auto built_network = network.Build();
```

### Гибкие архитектуры (Sequential)

```cpp
// Создание сложной архитектуры
auto network = Sequential()
    .Input(784)                           // Входной слой
    .Linear(256)                          // Полносвязный слой
    .ReLU()                              // Активация ReLU
    .Dropout(0.5f)                       // Dropout 50%
    .Linear(128)                         // Еще один полносвязный слой
    .ReLU()                              // Активация ReLU
    .Output(10, ActivationType::None)     // Выходной слой
    .Build();
```

## Функции активации

### Доступные функции

```cpp
enum class ActivationType {
    None,     // Без активации
    Relu,     // ReLU: max(0, x)
    Sigmoid,  // Сигмоида: 1/(1+e^(-x))
    Tanh      // Гиперболический тангенс
};
```

### Использование

```cpp
// В слое активации
ActivationLayer relu_layer(ActivationType::Relu);

// В выходном слое
OutputLayer output(128, 10, ActivationType::Sigmoid);

// Прямое применение
Tensor input = Tensor::Randn({32, 128});
Tensor activated = ApplyActivation(input, ActivationType::Relu);
```

## Прямой проход

### Базовое использование

```cpp
// Создание сети
auto network = Sequential()
    .Input(784)
    .Linear(128)
    .ReLU()
    .Linear(10)
    .Build();

// Входные данные
Tensor input = Tensor::Randn({32, 784});  // Batch size = 32

// Прямой проход
Tensor output = network.Forward(input);
```

### Режимы работы

```cpp
// Режим обучения (включен dropout)
network.SetTraining(true);
Tensor train_output = network.Forward(input);

// Режим вывода (dropout отключен)
network.SetTraining(false);
Tensor eval_output = network.Forward(input);
```

## Управление параметрами

### Подсчет параметров

```cpp
// Общее количество параметров
size_t total_params = network.CountParameters();

// Информация о сети
std::string summary = network.Summary();
network.PrintArchitecture();
```

### Работа с градиентами

```cpp
// Обнуление градиентов
network.ZeroGrad();

// Получение параметров (для оптимизаторов)
auto parameters = network.GetParameters();
```

## Dropout и регуляризация

### DropoutLayer

```cpp
class DropoutLayer : public Module {
public:
    DropoutLayer(float dropoutRate);
    Tensor Forward(const Tensor& input) override;
    
private:
    float m_dropoutRate;  // Вероятность отключения нейрона
    bool m_training;      // Режим обучения/вывода
};
```

### Использование

```cpp
// Создание слоя dropout
DropoutLayer dropout(0.3f);  // 30% нейронов отключается

// В Sequential
auto network = Sequential()
    .Linear(256)
    .ReLU()
    .Dropout(0.5f)  // 50% dropout
    .Linear(128)
    .Build();
```

## Примеры архитектур

### Классификатор MNIST

```cpp
auto mnist_classifier = Sequential()
    .Input(784)           // 28x28 изображения
    .Linear(512)
    .ReLU()
    .Dropout(0.2f)
    .Linear(256)
    .ReLU()
    .Dropout(0.2f)
    .Linear(128)
    .ReLU()
    .Output(10, ActivationType::None)  // 10 классов
    .Build();
```

### Простой автоэнкодер

```cpp
auto autoencoder = Sequential()
    .Input(784)
    .Linear(256)          // Энкодер
    .ReLU()
    .Linear(64)           // Узкое место
    .ReLU()
    .Linear(256)          // Декодер
    .ReLU()
    .Output(784, ActivationType::Sigmoid)
    .Build();
```

### Регрессионная сеть

```cpp
auto regressor = Sequential()
    .Input(10)            // 10 признаков
    .Linear(64)
    .ReLU()
    .Linear(32)
    .ReLU()
    .Output(1, ActivationType::None)  // Одно выходное значение
    .Build();
```

## Лучшие практики

### Инициализация весов

> [!tip] Рекомендации
> - Используйте Xavier/Glorot инициализацию для сигмоиды и tanh
> - Используйте He инициализацию для ReLU
> - Избегайте инициализации нулями для весов

### Архитектура сети

> [!tip] Советы по дизайну
> - Начинайте с простых архитектур
> - Добавляйте dropout для предотвращения переобучения
> - Используйте ReLU для скрытых слоев
> - Подбирайте размеры слоев экспериментально

### Режимы обучения

```cpp
// Цикл обучения
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    network.SetTraining(true);
    
    // Обучение на батчах
    for (auto& batch : train_data) {
        network.ZeroGrad();
        auto output = network.Forward(batch.input);
        auto loss = loss_function(output, batch.target);
        loss.Backward();
        optimizer.Step();
    }
    
    // Валидация
    network.SetTraining(false);
    evaluate(network, validation_data);
}
```

## Ограничения

> [!warning] Текущие ограничения
> - Поддерживаются только полносвязные слои
> - Ограниченный набор функций активации
> - Нет сверточных или рекуррентных слоев
> - Простая реализация dropout

## Связанные страницы

- [[Layers]] - Подробно о каждом типе слоя
- [[Activations]] - Функции активации
- [[Sequential Builder]] - Построение сложных архитектур
- [[Training Loop]] - Цикл обучения
- [[Loss Functions]] - Функции потерь

---

**См. также:** [[FFN]], [[Module System]], [[Architecture Patterns]] 