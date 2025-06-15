# CSV Datasets в KotML

## Обзор

Модуль CSV datasets предоставляет мощные возможности для чтения и обработки данных из CSV файлов. Это позволяет легко работать с реальными наборами данных в различных форматах.

## Основные возможности

- **Автоматическое определение колонок**: Автоматически использует все колонки кроме последней как входные данные, а последнюю как целевую
- **Ручное указание колонок**: Полный контроль над тем, какие колонки использовать для входных данных и целей
- **Поддержка заголовков**: Работа с файлами с заголовками и без них
- **Различные разделители**: Поддержка запятой, точки с запятой и других разделителей
- **Обработка ошибок**: Подробные сообщения об ошибках при некорректных данных
- **Интеграция с DataLoader**: Полная совместимость с системой батчевой загрузки

## Класс CSVDataset

### Конструкторы

#### Автоматическое определение колонок
```cpp
CSVDataset(const std::string& filename,
           bool hasHeader = true,
           char delimiter = ',',
           size_t skipRows = 0)
```

#### Ручное указание колонок
```cpp
CSVDataset(const std::string& filename, 
           const std::vector<size_t>& inputColumns,
           const std::vector<size_t>& targetColumns,
           bool hasHeader = true,
           char delimiter = ',',
           size_t skipRows = 0)
```

### Параметры

- `filename`: Путь к CSV файлу
- `inputColumns`: Индексы колонок для входных данных (начиная с 0)
- `targetColumns`: Индексы колонок для целевых данных (начиная с 0)
- `hasHeader`: Содержит ли первая строка заголовки колонок
- `delimiter`: Символ-разделитель (по умолчанию запятая)
- `skipRows`: Количество строк для пропуска после заголовка

### Основные методы

```cpp
// Получить образец по индексу
std::pair<Tensor, Tensor> GetItem(size_t index) const;

// Получить размер датасета
size_t Size() const;

// Получить имена колонок входных данных
std::vector<std::string> GetInputColumnNames() const;

// Получить имена колонок целевых данных
std::vector<std::string> GetTargetColumnNames() const;

// Вывести информацию о датасете
void PrintInfo() const;
```

## Утилиты для DataLoader

### Создание DataLoader из CSV

#### Автоматическое определение колонок
```cpp
auto loader = data::utils::CreateCSVLoader(
    "data.csv",           // файл
    32,                   // размер батча
    true,                 // перемешивание
    true,                 // есть заголовок
    ',',                  // разделитель
    0,                    // пропустить строк
    42                    // seed
);
```

#### Ручное указание колонок
```cpp
std::vector<size_t> inputCols = {0, 1, 2};  // колонки 0, 1, 2 как входные
std::vector<size_t> targetCols = {3};       // колонка 3 как целевая

auto loader = data::utils::CreateCSVLoader(
    "data.csv", inputCols, targetCols, 32, true
);
```

### Разделение на обучение/валидацию

```cpp
auto [trainLoader, valLoader] = data::utils::CreateCSVTrainValLoaders(
    "data.csv",
    0.8f,                 // 80% для обучения
    32,                   // размер батча
    true                  // перемешивание
);
```

## Примеры использования

### Простой регрессионный датасет

```cpp
// Файл: housing.csv
// price,area,rooms,age
// 250000,120,3,5
// 300000,150,4,3
// ...

// Автоматическое определение: area,rooms,age -> price
data::CSVDataset dataset("housing.csv");
dataset.PrintInfo();

// Получить первый образец
auto [input, target] = dataset.GetItem(0);
// input: [120, 3, 5]
// target: [250000]
```

### Классификационный датасет

```cpp
// Файл: iris.csv
// sepal_length,sepal_width,petal_length,petal_width,species
// 5.1,3.5,1.4,0.2,0
// 4.9,3.0,1.4,0.2,0
// ...

std::vector<size_t> features = {0, 1, 2, 3};  // все признаки
std::vector<size_t> labels = {4};             // класс

data::CSVDataset dataset("iris.csv", features, labels);

// Создать DataLoader
auto loader = data::utils::CreateCSVLoader("iris.csv", features, labels, 16);

// Итерация по батчам
for (auto [inputs, targets] : *loader) {
    // inputs: [16, 4] - 16 образцов, 4 признака
    // targets: [16, 1] - 16 меток классов
}
```

### Файл без заголовков

```cpp
// Файл: data.txt
// 1.5,2.3,4.1
// 2.1,3.4,5.2
// ...

data::CSVDataset dataset("data.txt", false);  // hasHeader = false
```

### Различные разделители

```cpp
// Файл с точкой с запятой: data.csv
// x1;x2;y
// 1.0;2.0;3.0
// ...

data::CSVDataset dataset("data.csv", true, ';');  // delimiter = ';'
```

### Пропуск строк

```cpp
// Файл с метаданными в начале:
// # Dataset created on 2024-01-01
// # Total samples: 1000
// x,y
// 1.0,2.0
// ...

data::CSVDataset dataset("data.csv", true, ',', 2);  // skipRows = 2
```

## Обработка ошибок

CSVDataset предоставляет подробные сообщения об ошибках:

```cpp
try {
    data::CSVDataset dataset("nonexistent.csv");
} catch (const std::runtime_error& e) {
    // "Cannot open CSV file: nonexistent.csv"
}

try {
    std::vector<size_t> invalidCols = {10, 11};  // колонки не существуют
    data::CSVDataset dataset("data.csv", invalidCols, {0});
} catch (const std::runtime_error& e) {
    // "Input column index 10 out of range at line 2 (available columns: 0-3)"
}
```

## Производительность

- **Загрузка в память**: Все данные загружаются в память при создании датасета
- **Быстрый доступ**: O(1) доступ к образцам после загрузки
- **Эффективная итерация**: Оптимизированная батчевая обработка через DataLoader

### Рекомендации по производительности

1. **Для больших файлов**: Рассмотрите возможность предварительной обработки данных
2. **Типы данных**: Все данные конвертируются в float для совместимости с тензорами
3. **Память**: Убедитесь, что у вас достаточно RAM для загрузки всего датасета

## Интеграция с обучением

```cpp
// Полный пример обучения с CSV данными
auto [trainLoader, valLoader] = data::utils::CreateCSVTrainValLoaders(
    "training_data.csv", 0.8f, 32, true
);

// Создание модели
auto model = Sequential()
    .Linear(trainLoader->GetDataset().GetInputShape()[0], 64)
    .ReLU()
    .Linear(64, 32)
    .ReLU()
    .Linear(32, trainLoader->GetDataset().GetTargetShape()[0])
    .Build();

// Оптимизатор и функция потерь
SGD optimizer(model.GetParameters(), 0.01f);
MSELoss lossFunction;

// Цикл обучения
for (int epoch = 0; epoch < 100; ++epoch) {
    float totalLoss = 0.0f;
    
    for (auto [inputs, targets] : *trainLoader) {
        // Прямой проход
        auto predictions = model.Forward(inputs);
        auto loss = lossFunction.Forward(predictions, targets);
        
        // Обратный проход
        optimizer.ZeroGrad();
        lossFunction.Backward();
        optimizer.Step();
        
        totalLoss += loss[0];
    }
    
    std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / trainLoader->NumBatches() << std::endl;
}
```

## Поддерживаемые форматы

### Разделители
- Запятая (`,`) - по умолчанию
- Точка с запятой (`;`)
- Табуляция (`\t`)
- Любой другой символ

### Кодировки
- UTF-8 (рекомендуется)
- ASCII

### Типы данных
- Числовые значения (конвертируются в float)
- Пустые ячейки (заменяются на 0.0)

## Ограничения

1. **Только числовые данные**: Все данные должны быть конвертируемы в float
2. **Загрузка в память**: Весь датасет загружается в память
3. **Фиксированная структура**: Все строки должны иметь одинаковое количество колонок

## Лучшие практики

1. **Проверка данных**: Всегда проверяйте корректность CSV файла перед использованием
2. **Обработка ошибок**: Используйте try-catch блоки для обработки ошибок загрузки
3. **Валидация**: Проверяйте размеры входных и целевых данных
4. **Документирование**: Документируйте структуру ваших CSV файлов
5. **Тестирование**: Тестируйте с небольшими файлами перед использованием больших датасетов 