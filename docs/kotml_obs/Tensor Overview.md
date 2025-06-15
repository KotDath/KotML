# Tensor Overview

Класс `Tensor` является основой библиотеки KotML и представляет многомерные массивы с поддержкой автоматического дифференцирования.

## Основные возможности

- ✅ Многомерные массивы произвольной размерности
- ✅ Автоматическое дифференцирование
- ✅ Арифметические операции
- ✅ Линейная алгебра
- ✅ Операции сокращения
- ✅ Статические методы создания

## Структура класса

```cpp
class Tensor {
private:
    std::vector<float> m_data;           // Данные тензора
    std::vector<size_t> m_shape;        // Форма тензора
    std::vector<size_t> m_strides;      // Шаги для индексации
    std::vector<float> m_grad;          // Градиенты
    bool m_requiresGrad;                // Требуется ли вычисление градиентов
    // ... другие поля для автодифференцирования
};
```

## Создание тензоров

### Конструкторы

```cpp
// Пустой тензор
Tensor tensor;

// Тензор заданной формы
Tensor tensor({3, 4}, true);  // 3x4 тензор с градиентами

// Тензор из данных
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
Tensor tensor(data, {2, 2});

// Тензор из списка инициализации
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
```

### Статические методы

```cpp
// Тензор из нулей
auto zeros = Tensor::Zeros({3, 3});

// Тензор из единиц
auto ones = Tensor::Ones({2, 4});

// Единичная матрица
auto eye = Tensor::Eye(5);

// Случайные значения (нормальное распределение)
auto randn = Tensor::Randn({10, 10});

// Случайные значения (равномерное распределение)
auto rand = Tensor::Rand({5, 5});
```

## Основные операции

### Доступ к данным

```cpp
// Размер и форма
size_t size = tensor.Size();
auto shape = tensor.Shape();
size_t ndim = tensor.Ndim();

// Доступ к элементам
float value = tensor[0];                    // Линейный индекс
float value = tensor.At({1, 2});           // Многомерный индекс

// Доступ к данным
const auto& data = tensor.Data();
```

### Изменение формы

```cpp
// Изменение формы
auto reshaped = tensor.Reshape({6, 2});

// Транспонирование (только для 2D)
auto transposed = tensor.Transpose();
```

## Связанные страницы

- [[Tensor Operations]] - Подробно об операциях с тензорами
- [[Automatic Differentiation]] - Автоматическое дифференцирование
- [[Basic Examples]] - Примеры использования тензоров

## Примеры кода

### Базовое использование

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

// Создание тензора
Tensor a = Tensor::Randn({3, 3}, true);
Tensor b = Tensor::Ones({3, 3}, true);

// Арифметические операции
Tensor c = a + b;
Tensor d = a * 2.0f;

// Вывод информации
a.Print();
std::cout << "Shape: ";
for (auto dim : a.Shape()) {
    std::cout << dim << " ";
}
```

### Автоматическое дифференцирование

```cpp
// Создание тензоров с градиентами
Tensor x = Tensor::Randn({2, 2}, true);
Tensor y = x * x + x;

// Обратное распространение
y.Backward();

// Получение градиентов
auto grad = x.Grad();
```

> [!info] Важно
> Тензоры используют семантику копирования по умолчанию. Для эффективности используйте семантику перемещения где это возможно.

> [!warning] Ограничения
> - Поддерживаются только тензоры типа `float`
> - Транспонирование работает только для 2D тензоров
> - Некоторые операции требуют совпадения размерностей

---

**См. также:** [[API Reference]], [[Tensor Operations]], [[Memory Management]] 