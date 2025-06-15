# Sequential Class - Builder Pattern для создания нейронных сетей

## Описание

Класс `Sequential` предоставляет удобный способ создания нейронных сетей с помощью паттерна "строитель" (Builder Pattern). Это позволяет создавать сети в декларативном стиле, последовательно добавляя слои.

## Основной синтаксис

```cpp
auto network = Sequential()
    .Linear(inputSize, outputSize)
    .ReLU()
    .Linear(hiddenSize, outputSize)
    .Build();
```

## Доступные методы

### Слои
- `.Linear(inputSize, outputSize, useBias=true)` - Полносвязный слой
- `.Input(inputSize)` - Входной слой (валидация размерности)
- `.Output(inputSize, outputSize, activation)` - Выходной слой с активацией

### Активации
- `.ReLU()` - ReLU активация
- `.Sigmoid()` - Sigmoid активация  
- `.Tanh()` - Tanh активация
- `.Activation(ActivationType)` - Произвольная активация

### Регуляризация
- `.Dropout(rate=0.5f)` - Dropout слой

### Управление
- `.Build()` - Завершение построения сети (обязательно!)
- `.Add(std::unique_ptr<Module>)` - Добавление произвольного модуля

## Примеры использования

### 1. Простая классификационная сеть
```cpp
auto classifier = Sequential()
    .Linear(784, 128)    // Входной слой
    .ReLU()              // Активация
    .Dropout(0.2f)       // Регуляризация
    .Linear(128, 10)     // Выходной слой
    .Sigmoid()           // Выходная активация
    .Build();
```

### 2. Регрессионная сеть
```cpp
auto regressor = Sequential()
    .Linear(10, 64)
    .Tanh()
    .Linear(64, 32)
    .ReLU()
    .Linear(32, 1)       // Один выход для регрессии
    .Build();
```

### 3. Глубокая сеть
```cpp
auto deepNet = Sequential()
    .Input(100)          // Валидация входа
    .Linear(100, 256)
    .ReLU()
    .Dropout(0.3f)
    .Linear(256, 128)
    .Tanh()
    .Linear(128, 64)
    .ReLU()
    .Linear(64, 10)
    .Build();
```

## Использование построенной сети

```cpp
// Создание сети
auto net = Sequential().Linear(10, 5).ReLU().Linear(5, 2).Build();

// Прямой проход
Tensor input({10}, 1.0f);
Tensor output = net.Forward(input);

// Режимы обучения
net.SetTraining(true);   // Включить Dropout
net.SetTraining(false);  // Отключить Dropout

// Работа с параметрами
auto params = net.Parameters();
net.ZeroGrad();

// Информация о сети
net.Summary();                    // Полная информация
size_t numParams = net.CountParameters();
size_t numLayers = net.GetNumLayers();
```

## Особенности реализации

1. **Move-only семантика**: Sequential использует move семантику для эффективности
2. **Безопасность**: Нельзя изменять сеть после вызова `Build()`
3. **Валидация**: Проверка корректности на этапе построения
4. **Совместимость**: Полная совместимость с существующими модулями

## Сравнение с FFN

```cpp
// Sequential стиль
auto seqNet = Sequential()
    .Linear(4, 8)
    .ReLU()
    .Linear(8, 3)
    .Build();

// FFN стиль
FFN ffnNet({4, 8, 3}, ActivationType::Relu);
```

Оба подхода создают идентичные сети, но Sequential предоставляет больше гибкости в выборе активаций и добавлении дополнительных слоев. 