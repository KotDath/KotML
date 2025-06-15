# Руководство по Обучению Моделей в KotML

## Обзор

Библиотека KotML теперь поддерживает полноценное обучение нейронных сетей с помощью методов `Compile()` и `Train()` для классов `FFN` и `Sequential`. Это обеспечивает удобный интерфейс в стиле Keras/PyTorch для создания и обучения моделей.

## Основные Компоненты

### 1. Компиляция Модели

Перед обучением модель должна быть скомпилирована с оптимизатором и функцией потерь:

```cpp
// Создание модели
FFN model({5, 32, 16, 1}, ActivationType::Relu);

// Компиляция
auto optimizer = std::make_unique<SGD>(0.01f, 0.9f, 1e-4f, false);
auto lossFunction = std::make_unique<MSELoss>();
model.Compile(std::move(optimizer), std::move(lossFunction));
```

### 2. Обучение Модели

После компиляции модель можно обучать:

```cpp
// Подготовка данных
auto [trainInputs, trainTargets] = PrepareTrainingData();
auto [valInputs, valTargets] = PrepareValidationData();

// Обучение
auto history = model.Train(
    trainInputs, trainTargets,  // Обучающие данные
    50,                         // Количество эпох
    32,                         // Размер батча
    &valInputs, &valTargets,    // Валидационные данные (опционально)
    true                        // Вывод прогресса
);
```

## Поддерживаемые Оптимизаторы

### SGD (Stochastic Gradient Descent)
```cpp
auto optimizer = std::make_unique<SGD>(
    0.01f,    // learning rate
    0.9f,     // momentum (опционально)
    1e-4f,    // weight decay (опционально)
    false     // Nesterov momentum (опционально)
);
```

### Adam (планируется)
```cpp
auto optimizer = std::make_unique<Adam>(0.001f, 0.9f, 0.999f, 1e-8f);
```

## Поддерживаемые Функции Потерь

### Для Регрессии
- **MSELoss**: Среднеквадратичная ошибка
- **MAELoss**: Средняя абсолютная ошибка
- **HuberLoss**: Робустная функция потерь

```cpp
auto mse = std::make_unique<MSELoss>();
auto mae = std::make_unique<MAELoss>();
auto huber = std::make_unique<HuberLoss>(1.0f); // delta parameter
```

### Для Классификации
- **BCELoss**: Бинарная кросс-энтропия
- **CrossEntropyLoss**: Многоклассовая кросс-энтропия

```cpp
auto bce = std::make_unique<BCELoss>(1e-7f); // epsilon parameter
auto ce = std::make_unique<CrossEntropyLoss>(1e-7f);
```

## Примеры Использования

### FFN для Регрессии

```cpp
#include "kotml/kotml.hpp"

// Создание модели
FFN model({10, 64, 32, 1}, ActivationType::Relu, ActivationType::None, 0.1f);

// Компиляция
auto optimizer = std::make_unique<SGD>(0.01f, 0.9f);
auto loss = std::make_unique<MSELoss>();
model.Compile(std::move(optimizer), std::move(loss));

// Обучение
auto history = model.Train(trainInputs, trainTargets, 100, 32);

// Оценка
float testLoss = model.Evaluate(testInputs, testTargets);

// Предсказания
auto predictions = model.Predict(newInputs);
```

### Sequential для Классификации

```cpp
// Создание модели с Builder pattern
auto model = Sequential()
    .Input(4)
    .Linear(4, 16)
    .ReLU()
    .Dropout(0.2f)
    .Linear(16, 8)
    .ReLU()
    .Linear(8, 3)
    .Sigmoid()
    .Build();

// Компиляция
auto optimizer = std::make_unique<SGD>(0.1f);
auto loss = std::make_unique<BCELoss>();
model.Compile(std::move(optimizer), std::move(loss));

// Обучение с валидацией
auto history = model.Train(
    trainInputs, trainTargets,
    50, 16,
    &valInputs, &valTargets,
    true
);
```

## Параметры Обучения

### Метод Train()

```cpp
std::vector<float> Train(
    const std::vector<Tensor>& trainInputs,      // Входные данные для обучения
    const std::vector<Tensor>& trainTargets,     // Целевые значения для обучения
    int epochs = 100,                            // Количество эпох
    int batchSize = 32,                          // Размер батча (0 = полный батч)
    const std::vector<Tensor>* validationInputs = nullptr,  // Валидационные входы
    const std::vector<Tensor>* validationTargets = nullptr, // Валидационные цели
    bool verbose = true                          // Вывод прогресса обучения
);
```

**Возвращает**: Вектор значений потерь по эпохам (training history)

### Размеры Батчей
- `batchSize > 0`: Мини-батч обучение
- `batchSize = 0`: Полный батч (все данные сразу)

## Форматы Данных

### Входные Тензоры
Все входные тензоры должны быть 2D с формой `[batch_size, features]`:

```cpp
// Для одного образца
Tensor input(inputData, {1, numFeatures});

// Для батча
Tensor batchInput(batchData, {batchSize, numFeatures});
```

### Целевые Тензоры
- **Регрессия**: `[batch_size, num_outputs]`
- **Классификация**: `[batch_size, num_classes]` (one-hot encoding)

## Мониторинг Обучения

### Прогресс Обучения
```cpp
// Обучение с выводом прогресса каждые 10 эпох
auto history = model.Train(trainInputs, trainTargets, 100, 32, nullptr, nullptr, true);

// Тихое обучение
auto history = model.Train(trainInputs, trainTargets, 100, 32, nullptr, nullptr, false);
```

### История Обучения
```cpp
auto history = model.Train(trainInputs, trainTargets, 50);

// Анализ сходимости
std::cout << "Initial loss: " << history.front() << std::endl;
std::cout << "Final loss: " << history.back() << std::endl;

// Построение графика потерь (пользовательская реализация)
PlotLossHistory(history);
```

## Оценка и Предсказания

### Оценка Модели
```cpp
// Оценка на тестовых данных
float testLoss = model.Evaluate(testInputs, testTargets);
std::cout << "Test loss: " << testLoss << std::endl;
```

### Получение Предсказаний
```cpp
// Предсказания для новых данных
auto predictions = model.Predict(newInputs);

// Обработка результатов
for (size_t i = 0; i < predictions.size(); ++i) {
    std::cout << "Prediction " << i << ": " << predictions[i][0] << std::endl;
}
```

## Обработка Ошибок

### Типичные Ошибки
1. **Обучение без компиляции**:
   ```cpp
   // ОШИБКА: модель не скомпилирована
   model.Train(inputs, targets); // Бросит исключение
   ```

2. **Несовпадение размеров данных**:
   ```cpp
   // ОШИБКА: разное количество входов и целей
   model.Train(inputs, targets); // inputs.size() != targets.size()
   ```

3. **Пустые данные**:
   ```cpp
   // ОШИБКА: пустые векторы данных
   model.Train({}, {}); // Бросит исключение
   ```

### Проверка Состояния
```cpp
// Проверка компиляции
if (model.IsCompiled()) {
    // Модель готова к обучению
    model.Train(inputs, targets);
} else {
    // Необходима компиляция
    model.Compile(optimizer, loss);
}
```

## Сравнение Производительности

### FFN vs Sequential
```cpp
// Эквивалентные архитектуры
FFN ffnModel({10, 20, 1});
auto seqModel = Sequential().Linear(10, 20).ReLU().Linear(20, 1).Build();

// Сравнение времени обучения
auto start = std::chrono::high_resolution_clock::now();
ffnModel.Train(inputs, targets, 10);
auto ffnTime = std::chrono::high_resolution_clock::now() - start;

start = std::chrono::high_resolution_clock::now();
seqModel.Train(inputs, targets, 10);
auto seqTime = std::chrono::high_resolution_clock::now() - start;
```

## Лучшие Практики

### 1. Подготовка Данных
- Нормализуйте входные данные
- Используйте правильные размерности тензоров
- Разделяйте данные на обучающие/валидационные/тестовые

### 2. Выбор Гиперпараметров
- Начинайте с малых learning rate (0.01-0.001)
- Используйте momentum для ускорения сходимости
- Экспериментируйте с размерами батчей

### 3. Мониторинг Обучения
- Используйте валидационные данные для контроля переобучения
- Сохраняйте историю обучения для анализа
- Останавливайте обучение при стагнации потерь

### 4. Архитектура Модели
- FFN для простых задач с фиксированной архитектурой
- Sequential для гибкого построения сложных моделей
- Используйте Dropout для регуляризации

## Интеграция с Существующим Кодом

### Загрузка Данных из CSV
```cpp
#include "kotml/utils/data_loader.hpp"

// Загрузка данных
auto dataset = LoadCSV("data.csv");
auto [inputs, targets] = PrepareDataForTraining(dataset);

// Обучение
model.Compile(optimizer, loss);
model.Train(inputs, targets);
```

### Сохранение и Загрузка Моделей
```cpp
// Сохранение параметров (планируется)
model.SaveWeights("model_weights.bin");

// Загрузка параметров (планируется)
model.LoadWeights("model_weights.bin");
```

## Заключение

Система обучения KotML предоставляет мощный и удобный интерфейс для создания и обучения нейронных сетей. Комбинация методов `Compile()` и `Train()` обеспечивает:

- **Простоту использования**: Интуитивный API в стиле современных фреймворков
- **Гибкость**: Поддержка различных архитектур, оптимизаторов и функций потерь
- **Производительность**: Эффективная реализация с поддержкой батч-обработки
- **Надежность**: Комплексная обработка ошибок и валидация данных

Для более подробных примеров см. `examples/training_example.cpp`. 