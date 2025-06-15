# Basic Examples

Базовые примеры использования библиотеки KotML для различных задач машинного обучения.

## Быстрый старт

### Простейший пример

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

int main() {
    // Создание простого тензора
    Tensor x = Tensor({1.0f, 2.0f, 3.0f}, {3});
    
    // Арифметические операции
    Tensor y = x * 2.0f + 1.0f;
    
    // Вывод результата
    std::cout << "x: " << x << std::endl;
    std::cout << "y = x * 2 + 1: " << y << std::endl;
    
    return 0;
}
```

### Создание простой нейронной сети

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

int main() {
    // Создание сети
    auto network = Sequential()
        .Input(784)      // Входной слой (28x28 изображения)
        .Linear(128)     // Скрытый слой
        .ReLU()          // Функция активации
        .Linear(10)      // Выходной слой (10 классов)
        .Build();
    
    // Тестовые данные
    Tensor input = Tensor::Randn({32, 784});  // Батч из 32 примеров
    
    // Прямой проход
    Tensor output = network.Forward(input);
    
    std::cout << "Input shape: " << input.Shape()[0] << "x" << input.Shape()[1] << std::endl;
    std::cout << "Output shape: " << output.Shape()[0] << "x" << output.Shape()[1] << std::endl;
    std::cout << "Network parameters: " << network.CountParameters() << std::endl;
    
    return 0;
}
```

## Работа с тензорами

### Создание и манипуляция тензорами

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

void TensorBasics() {
    std::cout << "=== Tensor Basics ===" << std::endl;
    
    // Различные способы создания тензоров
    Tensor zeros = Tensor::Zeros({2, 3});
    Tensor ones = Tensor::Ones({2, 3});
    Tensor randn = Tensor::Randn({2, 3});
    Tensor eye = Tensor::Eye(3);
    
    std::cout << "Zeros:\n" << zeros << std::endl;
    std::cout << "Ones:\n" << ones << std::endl;
    std::cout << "Random:\n" << randn << std::endl;
    std::cout << "Identity:\n" << eye << std::endl;
    
    // Арифметические операции
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    
    std::cout << "A:\n" << a << std::endl;
    std::cout << "B:\n" << b << std::endl;
    std::cout << "A + B:\n" << (a + b) << std::endl;
    std::cout << "A * B (element-wise):\n" << (a * b) << std::endl;
    std::cout << "A @ B (matrix multiply):\n" << a.Matmul(b) << std::endl;
}
```

### Автоматическое дифференцирование

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

void AutogradExample() {
    std::cout << "=== Autograd Example ===" << std::endl;
    
    // Создание тензоров с градиентами
    Tensor x = Tensor({2.0f, 3.0f}, {2}, true);  // requiresGrad = true
    Tensor y = Tensor({1.0f, 4.0f}, {2}, true);
    
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    
    // Вычисления
    Tensor z = x * y;           // z = x * y
    Tensor w = z + x;           // w = z + x = x*y + x
    Tensor loss = w.Sum();      // loss = sum(x*y + x)
    
    std::cout << "z = x * y: " << z << std::endl;
    std::cout << "w = z + x: " << w << std::endl;
    std::cout << "loss = sum(w): " << loss << std::endl;
    
    // Обратное распространение
    loss.Backward();
    
    // Градиенты
    std::cout << "∂loss/∂x: " << x.Grad()[0] << ", " << x.Grad()[1] << std::endl;
    std::cout << "∂loss/∂y: " << y.Grad()[0] << ", " << y.Grad()[1] << std::endl;
    
    // Ожидаемые градиенты:
    // ∂loss/∂x = y + 1 = {2.0, 5.0}
    // ∂loss/∂y = x = {2.0, 3.0}
}
```

## Построение нейронных сетей

### Использование Sequential

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void SequentialExample() {
    std::cout << "=== Sequential Network Example ===" << std::endl;
    
    // Создание сети с различными слоями
    auto network = Sequential()
        .Input(784)                    // Входной слой
        .Linear(256)                   // Первый скрытый слой
        .ReLU()                        // Активация ReLU
        .Dropout(0.2f)                 // Dropout для регуляризации
        .Linear(128)                   // Второй скрытый слой
        .ReLU()                        // Активация ReLU
        .Dropout(0.2f)                 // Dropout
        .Output(10, ActivationType::None) // Выходной слой без активации
        .Build();
    
    // Информация о сети
    std::cout << "Network Summary:" << std::endl;
    network.PrintArchitecture();
    std::cout << "Total parameters: " << network.CountParameters() << std::endl;
    
    // Тестирование прямого прохода
    Tensor input = Tensor::Randn({16, 784});  // Батч размером 16
    
    // Режим обучения
    network.SetTraining(true);
    Tensor train_output = network.Forward(input);
    std::cout << "Training output shape: " << train_output.Shape()[0] 
              << "x" << train_output.Shape()[1] << std::endl;
    
    // Режим вывода
    network.SetTraining(false);
    Tensor eval_output = network.Forward(input);
    std::cout << "Evaluation output shape: " << eval_output.Shape()[0] 
              << "x" << eval_output.Shape()[1] << std::endl;
}
```

### Использование FFN

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void FFNExample() {
    std::cout << "=== FFN Example ===" << std::endl;
    
    // Создание простой прямонаправленной сети
    FFN network(784);  // Входной размер
    
    network.AddLayer(512, ActivationType::Relu)
           .AddLayer(256, ActivationType::Relu)
           .AddLayer(128, ActivationType::Relu)
           .AddLayer(10, ActivationType::None);  // Выходной слой
    
    auto built_network = network.Build();
    
    // Информация о сети
    std::cout << "FFN Architecture:" << std::endl;
    built_network.PrintArchitecture();
    
    // Тестирование
    Tensor input = Tensor::Randn({8, 784});
    Tensor output = built_network.Forward(input);
    
    std::cout << "Input: " << input.Shape()[0] << "x" << input.Shape()[1] << std::endl;
    std::cout << "Output: " << output.Shape()[0] << "x" << output.Shape()[1] << std::endl;
}
```

## Обучение модели

### Простой цикл обучения

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void SimpleTrainingLoop() {
    std::cout << "=== Simple Training Loop ===" << std::endl;
    
    // Генерация синтетических данных
    int num_samples = 1000;
    int input_size = 20;
    int output_size = 1;
    
    Tensor X = Tensor::Randn({num_samples, input_size});
    Tensor y = Tensor::Randn({num_samples, output_size});
    
    // Создание модели
    auto model = Sequential()
        .Input(input_size)
        .Linear(64)
        .ReLU()
        .Linear(32)
        .ReLU()
        .Output(output_size)
        .Build();
    
    // Настройка оптимизатора
    auto parameters = model.GetParameters();
    SGD optimizer(parameters, 0.01f);
    
    // Параметры обучения
    int num_epochs = 50;
    int batch_size = 32;
    
    std::cout << "Starting training..." << std::endl;
    std::cout << "Dataset size: " << num_samples << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Epochs: " << num_epochs << std::endl;
    
    // Цикл обучения
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        // Обучение по батчам
        for (int i = 0; i < num_samples; i += batch_size) {
            int end_idx = std::min(i + batch_size, num_samples);
            
            // Извлечение батча (упрощенно)
            Tensor X_batch = X.Slice(i, end_idx);
            Tensor y_batch = y.Slice(i, end_idx);
            
            // Обнуление градиентов
            optimizer.ZeroGrad();
            
            // Прямой проход
            model.SetTraining(true);
            Tensor predictions = model.Forward(X_batch);
            
            // Вычисление функции потерь (MSE)
            Tensor diff = predictions - y_batch;
            Tensor loss = (diff * diff).Mean();
            
            epoch_loss += loss.Item();
            num_batches++;
            
            // Обратное распространение
            loss.Backward();
            
            // Обновление параметров
            optimizer.Step();
        }
        
        // Вывод прогресса
        if (epoch % 10 == 0 || epoch == num_epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                      << ", Loss: " << epoch_loss / num_batches << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}
```

### Обучение с валидацией

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void TrainingWithValidation() {
    std::cout << "=== Training with Validation ===" << std::endl;
    
    // Подготовка данных
    int train_size = 800;
    int val_size = 200;
    int input_size = 10;
    int output_size = 3;
    
    Tensor X_train = Tensor::Randn({train_size, input_size});
    Tensor y_train = Tensor::Randn({train_size, output_size});
    Tensor X_val = Tensor::Randn({val_size, input_size});
    Tensor y_val = Tensor::Randn({val_size, output_size});
    
    // Создание модели
    auto model = Sequential()
        .Input(input_size)
        .Linear(128)
        .ReLU()
        .Dropout(0.3f)
        .Linear(64)
        .ReLU()
        .Dropout(0.3f)
        .Output(output_size)
        .Build();
    
    // Оптимизатор
    auto parameters = model.GetParameters();
    SGD optimizer(parameters, 0.01f);
    
    // Параметры обучения
    int num_epochs = 100;
    int batch_size = 32;
    float best_val_loss = std::numeric_limits<float>::max();
    int patience = 10;
    int patience_counter = 0;
    
    std::cout << "Model parameters: " << model.CountParameters() << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // === ОБУЧЕНИЕ ===
        model.SetTraining(true);
        float train_loss = 0.0f;
        int train_batches = 0;
        
        for (int i = 0; i < train_size; i += batch_size) {
            int end_idx = std::min(i + batch_size, train_size);
            
            Tensor X_batch = X_train.Slice(i, end_idx);
            Tensor y_batch = y_train.Slice(i, end_idx);
            
            optimizer.ZeroGrad();
            Tensor predictions = model.Forward(X_batch);
            Tensor loss = ((predictions - y_batch) * (predictions - y_batch)).Mean();
            
            train_loss += loss.Item();
            train_batches++;
            
            loss.Backward();
            optimizer.Step();
        }
        train_loss /= train_batches;
        
        // === ВАЛИДАЦИЯ ===
        model.SetTraining(false);
        float val_loss = 0.0f;
        int val_batches = 0;
        
        for (int i = 0; i < val_size; i += batch_size) {
            int end_idx = std::min(i + batch_size, val_size);
            
            Tensor X_batch = X_val.Slice(i, end_idx);
            Tensor y_batch = y_val.Slice(i, end_idx);
            
            Tensor predictions = model.Forward(X_batch);
            Tensor loss = ((predictions - y_batch) * (predictions - y_batch)).Mean();
            
            val_loss += loss.Item();
            val_batches++;
        }
        val_loss /= val_batches;
        
        // === МОНИТОРИНГ ===
        if (epoch % 10 == 0 || epoch == num_epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                      << " - Train Loss: " << train_loss
                      << ", Val Loss: " << val_loss << std::endl;
        }
        
        // === РАННЯЯ ОСТАНОВКА ===
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }
    
    std::cout << "Best validation loss: " << best_val_loss << std::endl;
}
```

## Специализированные примеры

### Линейная регрессия

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void LinearRegressionExample() {
    std::cout << "=== Linear Regression Example ===" << std::endl;
    
    // Генерация данных для линейной регрессии
    // y = 2*x1 + 3*x2 + 1 + noise
    int num_samples = 500;
    Tensor X = Tensor::Randn({num_samples, 2});
    
    // Создание целевых значений
    std::vector<float> y_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float x1 = X.At({i, 0});
        float x2 = X.At({i, 1});
        float noise = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
        y_data[i] = 2.0f * x1 + 3.0f * x2 + 1.0f + noise;
    }
    Tensor y = Tensor(y_data, {num_samples, 1});
    
    // Простая линейная модель
    auto model = Sequential()
        .Input(2)
        .Output(1)  // Без активации для регрессии
        .Build();
    
    auto parameters = model.GetParameters();
    SGD optimizer(parameters, 0.01f);
    
    std::cout << "True coefficients: w1=2.0, w2=3.0, b=1.0" << std::endl;
    std::cout << "Training linear regression..." << std::endl;
    
    // Обучение
    for (int epoch = 0; epoch < 1000; ++epoch) {
        optimizer.ZeroGrad();
        
        Tensor predictions = model.Forward(X);
        Tensor loss = ((predictions - y) * (predictions - y)).Mean();
        
        loss.Backward();
        optimizer.Step();
        
        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.Item() << std::endl;
        }
    }
    
    // Финальная оценка
    model.SetTraining(false);
    Tensor final_predictions = model.Forward(X);
    Tensor final_loss = ((final_predictions - y) * (final_predictions - y)).Mean();
    
    std::cout << "Final loss: " << final_loss.Item() << std::endl;
    std::cout << "Model trained successfully!" << std::endl;
}
```

### Бинарная классификация

```cpp
#include "kotml/kotml.hpp"
using namespace kotml;

void BinaryClassificationExample() {
    std::cout << "=== Binary Classification Example ===" << std::endl;
    
    // Генерация данных для бинарной классификации
    int num_samples = 1000;
    int num_features = 5;
    
    Tensor X = Tensor::Randn({num_samples, num_features});
    
    // Создание меток (простое правило)
    std::vector<float> y_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < num_features; ++j) {
            sum += X.At({i, j});
        }
        y_data[i] = (sum > 0.0f) ? 1.0f : 0.0f;
    }
    Tensor y = Tensor(y_data, {num_samples, 1});
    
    // Модель для бинарной классификации
    auto model = Sequential()
        .Input(num_features)
        .Linear(32)
        .ReLU()
        .Linear(16)
        .ReLU()
        .Output(1, ActivationType::Sigmoid)  // Сигмоида для вероятности
        .Build();
    
    auto parameters = model.GetParameters();
    SGD optimizer(parameters, 0.1f);
    
    std::cout << "Training binary classifier..." << std::endl;
    
    // Обучение
    int batch_size = 64;
    for (int epoch = 0; epoch < 100; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;
        
        for (int i = 0; i < num_samples; i += batch_size) {
            int end_idx = std::min(i + batch_size, num_samples);
            
            Tensor X_batch = X.Slice(i, end_idx);
            Tensor y_batch = y.Slice(i, end_idx);
            
            optimizer.ZeroGrad();
            
            Tensor predictions = model.Forward(X_batch);
            
            // Binary cross-entropy loss (упрощенная версия)
            Tensor loss = ((y_batch * predictions.Log()) + 
                          ((1.0f - y_batch) * (1.0f - predictions).Log())).Mean() * (-1.0f);
            
            epoch_loss += loss.Item();
            num_batches++;
            
            loss.Backward();
            optimizer.Step();
        }
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss / num_batches << std::endl;
        }
    }
    
    // Оценка точности
    model.SetTraining(false);
    Tensor predictions = model.Forward(X);
    
    int correct = 0;
    for (int i = 0; i < num_samples; ++i) {
        float pred = predictions.At({i, 0}) > 0.5f ? 1.0f : 0.0f;
        float true_label = y.At({i, 0});
        if (pred == true_label) correct++;
    }
    
    float accuracy = float(correct) / num_samples;
    std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}
```

## Полный пример программы

```cpp
#include "kotml/kotml.hpp"
#include <iostream>

using namespace kotml;

int main() {
    std::cout << "KotML Basic Examples" << std::endl;
    std::cout << "===================" << std::endl;
    
    try {
        // Запуск всех примеров
        TensorBasics();
        std::cout << std::endl;
        
        AutogradExample();
        std::cout << std::endl;
        
        SequentialExample();
        std::cout << std::endl;
        
        FFNExample();
        std::cout << std::endl;
        
        SimpleTrainingLoop();
        std::cout << std::endl;
        
        TrainingWithValidation();
        std::cout << std::endl;
        
        LinearRegressionExample();
        std::cout << std::endl;
        
        BinaryClassificationExample();
        std::cout << std::endl;
        
        std::cout << "All examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Компиляция и запуск

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(KotMLExamples)

set(CMAKE_CXX_STANDARD 17)

# Найти библиотеку KotML
find_package(kotml REQUIRED)

# Создать исполняемый файл
add_executable(basic_examples basic_examples.cpp)

# Связать с библиотекой
target_link_libraries(basic_examples kotml)
```

### Команды сборки

```bash
mkdir build
cd build
cmake ..
make
./basic_examples
```

## Связанные страницы

- [[Getting Started]] - Начало работы с KotML
- [[Neural Networks Overview]] - Подробно о нейронных сетях
- [[Training Loop]] - Организация обучения
- [[Advanced Examples]] - Продвинутые примеры

---

**См. также:** [[Installation]], [[API Reference]], [[Troubleshooting]] 