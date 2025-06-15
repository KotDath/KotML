# KotML - Библиотека машинного обучения на C++

KotML - это современная библиотека машинного обучения на C++17, предоставляющая мощные инструменты для работы с тензорами, нейронными сетями и автоматическим дифференцированием.

## Особенности

- 🚀 **Высокая производительность**: Оптимизированные операции с тензорами
- 🔄 **Автоматическое дифференцирование**: Полная поддержка обратного распространения
- 🧠 **Нейронные сети**: Готовые слои и функции активации
- 📊 **Анализ данных**: Инструменты для обработки и анализа данных
- 🎯 **Простой API**: Интуитивно понятный интерфейс
- 🔧 **Модульность**: Гибкая архитектура для расширения

## Структура проекта

```
KotML/
├── include/kotml/          # Заголовочные файлы
│   ├── tensor.hpp         # Основной класс Tensor
│   ├── ops/               # Операции над тензорами
│   ├── nn/                # Нейронные сети
│   ├── optim/             # Оптимизаторы
│   └── utils/             # Утилиты
├── src/                   # Исходные файлы
├── examples/              # Примеры использования
├── tests/                 # Тесты
└── CMakeLists.txt        # Конфигурация сборки
```

## Быстрый старт

### Требования

- C++17 или новее
- CMake 3.12+
- Компилятор с поддержкой C++17 (GCC 7+, Clang 5+, MSVC 2017+)

### Сборка

```bash
git clone https://github.com/kotdath/KotML.git
cd KotML
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Запуск примера

```bash
./examples/basic_usage
```

## API Reference

### Класс Tensor

Основной класс для работы с многомерными массивами данных.

#### Конструкторы

```cpp
// Создание пустого тензора
Tensor tensor;

// Создание тензора заданной формы
Tensor tensor({3, 4}, true);  // 3x4, requires_grad=true

// Создание из данных
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

// Создание из initializer_list
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
```

#### Статические методы создания

```cpp
// Тензор нулей
Tensor zeros = Tensor::zeros({3, 4});

// Тензор единиц
Tensor ones = Tensor::ones({3, 4});

// Единичная матрица
Tensor eye = Tensor::eye(3);

// Случайные значения (нормальное распределение)
Tensor randn = Tensor::randn({3, 4});

// Случайные значения (равномерное распределение)
Tensor rand = Tensor::rand({3, 4});
```

#### Арифметические операции

```cpp
Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
Tensor b({2.0f, 3.0f, 4.0f, 5.0f}, {2, 2});

// Поэлементные операции
Tensor c = a + b;  // Сложение
Tensor d = a - b;  // Вычитание
Tensor e = a * b;  // Умножение
Tensor f = a / b;  // Деление

// Операции со скалярами
Tensor g = a + 2.0f;
Tensor h = a * 3.0f;
```

#### Линейная алгебра

```cpp
// Матричное умножение
Tensor result = a.matmul(b);

// Транспонирование
Tensor transposed = a.transpose();

// Изменение формы
Tensor reshaped = a.reshape({4, 1});
```

#### Операции сокращения

```cpp
// Сумма всех элементов
Tensor sum_all = a.sum();

// Сумма по оси
Tensor sum_axis0 = a.sum(0);
Tensor sum_axis1 = a.sum(1);

// Среднее значение
Tensor mean_all = a.mean();
Tensor mean_axis0 = a.mean(0);
```

#### Автоматическое дифференцирование

```cpp
// Создание тензоров с градиентами
Tensor x({2.0f, 3.0f}, {2, 1}, true);  // requires_grad=true
Tensor y({1.0f, 4.0f}, {2, 1}, true);

// Вычисления
Tensor z = x * y + x;
Tensor loss = z.sum();

// Обратное распространение
loss.backward();

// Получение градиентов
const auto& grad_x = x.grad();
const auto& grad_y = y.grad();
```

## Примеры использования

### Базовые операции с тензорами

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

int main() {
    // Создание тензоров
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b = Tensor::ones({2, 2}) * 2.0f;
    
    // Арифметические операции
    Tensor c = a + b;
    Tensor d = a.matmul(b);
    
    // Вывод результатов
    std::cout << "a + b = " << c << std::endl;
    std::cout << "a @ b = " << d << std::endl;
    
    return 0;
}
```

### Автоматическое дифференцирование

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

int main() {
    // Создание переменных с градиентами
    Tensor x({2.0f}, {1}, true);
    Tensor y({3.0f}, {1}, true);
    
    // Функция: f(x, y) = x² + 2xy + y²
    Tensor z = x * x + 2.0f * x * y + y * y;
    
    // Обратное распространение
    z.backward();
    
    // Градиенты: df/dx = 2x + 2y, df/dy = 2x + 2y
    std::cout << "df/dx = " << x.grad()[0] << std::endl;  // 10
    std::cout << "df/dy = " << y.grad()[0] << std::endl;  // 10
    
    return 0;
}
```

### Линейная регрессия

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

int main() {
    // Данные
    Tensor X({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1});
    Tensor y({2.0f, 4.0f, 6.0f, 8.0f, 10.0f}, {5, 1});
    
    // Параметры модели
    Tensor w({0.1f}, {1, 1}, true);
    Tensor b({0.0f}, {1}, true);
    
    float learning_rate = 0.01f;
    
    // Обучение
    for (int epoch = 0; epoch < 1000; ++epoch) {
        // Прямой проход
        Tensor y_pred = X.matmul(w) + b;
        
        // Функция потерь (MSE)
        Tensor diff = y_pred - y;
        Tensor loss = (diff * diff).mean();
        
        // Обратное распространение
        w.zero_grad();
        b.zero_grad();
        loss.backward();
        
        // Обновление параметров
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= learning_rate * w.grad()[i];
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] -= learning_rate * b.grad()[i];
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss[0] << std::endl;
        }
    }
    
    std::cout << "Final w: " << w[0] << ", b: " << b[0] << std::endl;
    
    return 0;
}
```

## Планы развития

- [ ] Поддержка GPU (CUDA)
- [ ] Дополнительные слои нейронных сетей
- [ ] Сверточные нейронные сети
- [ ] Рекуррентные нейронные сети
- [ ] Оптимизация производительности
- [ ] Python bindings
- [ ] Поддержка различных типов данных (double, int)

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## Лицензия

Этот проект распространяется под лицензией MIT. См. файл [LICENSE](LICENSE) для подробностей.

## Контакты

- GitHub: [https://github.com/yourusername/KotML](https://github.com/yourusername/KotML)
- Email: your.email@example.com

---

**KotML** - мощная и гибкая библиотека для машинного обучения на C++! 