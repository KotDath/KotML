# Layers

Подробное описание всех типов слоев в KotML и их использования.

## Базовый класс Module

Все слои наследуются от абстрактного класса `Module`:

```cpp
class Module {
public:
    virtual ~Module() = default;
    virtual Tensor Forward(const Tensor& input) = 0;
    virtual size_t CountParameters() const = 0;
    virtual void SetTraining(bool training) = 0;
    virtual bool IsTraining() const = 0;
    virtual void ZeroGrad() = 0;
    virtual std::string Summary() const = 0;
};
```

## InputLayer

Входной слой для валидации размерности входных данных.

### Конструктор

```cpp
InputLayer(size_t inputSize);
```

### Параметры
- `inputSize` - ожидаемый размер входного вектора

### Использование

```cpp
// Создание входного слоя
InputLayer input_layer(784);  // Ожидает вектор размера 784

// Валидация входа
Tensor input = Tensor::Randn({32, 784});  // Batch size = 32
Tensor output = input_layer.Forward(input);  // Проверяет размерность
```

### Особенности
- Не имеет обучаемых параметров
- Выполняет только валидацию размерности
- Возвращает входной тензор без изменений

## LinearLayer

Полносвязный (линейный) слой - основа большинства нейронных сетей.

### Конструктор

```cpp
LinearLayer(size_t inputSize, size_t outputSize, bool bias = true);
```

### Параметры
- `inputSize` - размер входного вектора
- `outputSize` - размер выходного вектора  
- `bias` - использовать ли смещения (по умолчанию true)

### Математическая операция

```
output = input * weights + bias
```

Где:
- `weights` имеет форму `[inputSize, outputSize]`
- `bias` имеет форму `[outputSize]`

### Использование

```cpp
// Создание линейного слоя
LinearLayer linear(784, 128);  // 784 -> 128

// Прямой проход
Tensor input = Tensor::Randn({32, 784});   // Batch size = 32
Tensor output = linear.Forward(input);     // Форма: {32, 128}

// Подсчет параметров
size_t params = linear.CountParameters();  // 784 * 128 + 128 = 100,480
```

### Инициализация весов

```cpp
// Веса инициализируются случайно при создании
// Можно получить доступ к параметрам для кастомной инициализации
auto parameters = linear.GetParameters();
```

## ActivationLayer

Слой функций активации для введения нелинейности.

### Конструктор

```cpp
ActivationLayer(ActivationType type);
```

### Типы активации

```cpp
enum class ActivationType {
    None,     // Без активации (линейная)
    Relu,     // ReLU: max(0, x)
    Sigmoid,  // Сигмоида: 1/(1+e^(-x))
    Tanh      // Гиперболический тангенс: tanh(x)
};
```

### Использование

```cpp
// Создание слоев активации
ActivationLayer relu(ActivationType::Relu);
ActivationLayer sigmoid(ActivationType::Sigmoid);
ActivationLayer tanh(ActivationType::Tanh);

// Применение активации
Tensor input = Tensor::Randn({32, 128});
Tensor relu_output = relu.Forward(input);
Tensor sigmoid_output = sigmoid.Forward(input);
```

### Характеристики функций

| Функция | Диапазон | Производная | Применение |
|---------|----------|-------------|------------|
| ReLU | [0, +∞) | 0 или 1 | Скрытые слои |
| Sigmoid | (0, 1) | Гладкая | Бинарная классификация |
| Tanh | (-1, 1) | Гладкая | Скрытые слои |

## DropoutLayer

Слой регуляризации для предотвращения переобучения.

### Конструктор

```cpp
DropoutLayer(float dropoutRate);
```

### Параметры
- `dropoutRate` - вероятность отключения нейрона (0.0 - 1.0)

### Принцип работы

**Режим обучения:**
- Случайно обнуляет `dropoutRate * 100%` элементов
- Масштабирует оставшиеся элементы на `1/(1-dropoutRate)`

**Режим вывода:**
- Пропускает входной тензор без изменений

### Использование

```cpp
// Создание dropout слоя
DropoutLayer dropout(0.5f);  // 50% нейронов отключается

// Режим обучения
dropout.SetTraining(true);
Tensor train_output = dropout.Forward(input);  // Применяется dropout

// Режим вывода
dropout.SetTraining(false);
Tensor eval_output = dropout.Forward(input);   // Dropout отключен
```

### Рекомендации по использованию

> [!tip] Лучшие практики
> - Используйте 0.2-0.5 для скрытых слоев
> - Не применяйте к выходному слою
> - Всегда переключайте режимы обучения/вывода

## OutputLayer

Композитный выходной слой, объединяющий Linear и Activation.

### Конструктор

```cpp
OutputLayer(size_t inputSize, size_t outputSize, 
           ActivationType activation = ActivationType::None);
```

### Параметры
- `inputSize` - размер входного вектора
- `outputSize` - размер выходного вектора
- `activation` - функция активации для выхода

### Использование

```cpp
// Выходной слой для классификации
OutputLayer classifier(128, 10, ActivationType::None);  // Логиты

// Выходной слой для регрессии
OutputLayer regressor(64, 1, ActivationType::None);

// Выходной слой с сигмоидой
OutputLayer binary_classifier(32, 1, ActivationType::Sigmoid);
```

### Эквивалентность

```cpp
// OutputLayer эквивалентен:
LinearLayer linear(inputSize, outputSize);
ActivationLayer activation(activationType);

// output = activation.Forward(linear.Forward(input));
```

## Комбинирование слоев

### Ручное комбинирование

```cpp
// Создание отдельных слоев
InputLayer input(784);
LinearLayer hidden1(784, 256);
ActivationLayer relu1(ActivationType::Relu);
DropoutLayer dropout1(0.3f);
LinearLayer hidden2(256, 128);
ActivationLayer relu2(ActivationType::Relu);
OutputLayer output(128, 10);

// Прямой проход
Tensor x = input.Forward(data);
x = hidden1.Forward(x);
x = relu1.Forward(x);
x = dropout1.Forward(x);
x = hidden2.Forward(x);
x = relu2.Forward(x);
x = output.Forward(x);
```

### Использование Sequential

```cpp
// Более удобный способ
auto network = Sequential()
    .Input(784)
    .Linear(256)
    .ReLU()
    .Dropout(0.3f)
    .Linear(128)
    .ReLU()
    .Output(10)
    .Build();

Tensor output = network.Forward(data);
```

## Управление параметрами

### Подсчет параметров

```cpp
LinearLayer layer(100, 50);
size_t params = layer.CountParameters();  // 100 * 50 + 50 = 5,050

ActivationLayer activation(ActivationType::Relu);
size_t act_params = activation.CountParameters();  // 0 (нет параметров)
```

### Обнуление градиентов

```cpp
// Для отдельного слоя
layer.ZeroGrad();

// Для всей сети
network.ZeroGrad();
```

### Режимы обучения

```cpp
// Установка режима для слоя
dropout_layer.SetTraining(true);   // Режим обучения
dropout_layer.SetTraining(false);  // Режим вывода

// Проверка режима
bool is_training = dropout_layer.IsTraining();
```

## Информация о слоях

### Сводка слоя

```cpp
LinearLayer layer(784, 128);
std::string summary = layer.Summary();
// Выводит информацию о типе слоя, размерах, параметрах
```

### Архитектура сети

```cpp
auto network = Sequential()
    .Input(784)
    .Linear(256)
    .ReLU()
    .Linear(10)
    .Build();

network.PrintArchitecture();
// Выводит полную архитектуру сети
```

## Производительность

### Оптимизация памяти

> [!tip] Советы по производительности
> - Используйте семантику перемещения для больших слоев
> - Избегайте ненужных копий тензоров
> - Переиспользуйте буферы где возможно

### Размеры батчей

```cpp
// Эффективная обработка батчей
Tensor batch = Tensor::Randn({64, 784});  // Batch size = 64
Tensor output = layer.Forward(batch);     // Обрабатывает весь батч
```

## Расширение системы

### Создание кастомного слоя

```cpp
class CustomLayer : public Module {
private:
    Tensor m_weights;
    bool m_training;

public:
    CustomLayer(size_t inputSize, size_t outputSize) 
        : m_weights(Tensor::Randn({inputSize, outputSize}, true))
        , m_training(true) {}

    Tensor Forward(const Tensor& input) override {
        // Кастомная логика прямого прохода
        return input.Matmul(m_weights);
    }

    size_t CountParameters() const override {
        return m_weights.Size();
    }

    void SetTraining(bool training) override {
        m_training = training;
    }

    bool IsTraining() const override {
        return m_training;
    }

    void ZeroGrad() override {
        m_weights.ZeroGrad();
    }

    std::string Summary() const override {
        return "CustomLayer";
    }
};
```

## Связанные страницы

- [[Neural Networks Overview]] - Обзор системы нейронных сетей
- [[Activations]] - Подробно о функциях активации
- [[Sequential Builder]] - Построение архитектур
- [[Module System]] - Система модулей

---

**См. также:** [[FFN]], [[Training Loop]], [[Parameter Management]] 