# Basic Operations

Документация по базовым операциям в KotML, включая арифметические операции, операции сравнения и элементарные функции.

## Обзор

Базовые операции в KotML реализованы как методы класса `Tensor` и свободные функции в пространстве имен `kotml::ops`. Все операции поддерживают автоматическое дифференцирование.

## Арифметические операции

### Операции между тензорами

```cpp
#include "kotml/tensor.hpp"
using namespace kotml;

Tensor a = Tensor({1.0f, 2.0f, 3.0f}, {3});
Tensor b = Tensor({4.0f, 5.0f, 6.0f}, {3});

// Сложение
Tensor sum = a + b;        // {5.0f, 7.0f, 9.0f}

// Вычитание
Tensor diff = a - b;       // {-3.0f, -3.0f, -3.0f}

// Поэлементное умножение
Tensor prod = a * b;       // {4.0f, 10.0f, 18.0f}

// Поэлементное деление
Tensor quot = a / b;       // {0.25f, 0.4f, 0.5f}
```

### Операции со скалярами

```cpp
Tensor tensor = Tensor({1.0f, 2.0f, 3.0f}, {3});
float scalar = 2.0f;

// Сложение со скаляром
Tensor result1 = tensor + scalar;    // {3.0f, 4.0f, 5.0f}
Tensor result2 = scalar + tensor;    // {3.0f, 4.0f, 5.0f}

// Вычитание скаляра
Tensor result3 = tensor - scalar;    // {-1.0f, 0.0f, 1.0f}
Tensor result4 = scalar - tensor;    // {1.0f, 0.0f, -1.0f}

// Умножение на скаляр
Tensor result5 = tensor * scalar;    // {2.0f, 4.0f, 6.0f}
Tensor result6 = scalar * tensor;    // {2.0f, 4.0f, 6.0f}

// Деление на скаляр
Tensor result7 = tensor / scalar;    // {0.5f, 1.0f, 1.5f}
Tensor result8 = scalar / tensor;    // {2.0f, 1.0f, 0.667f}
```

### Присваивающие операции

```cpp
Tensor a = Tensor({1.0f, 2.0f, 3.0f}, {3});
Tensor b = Tensor({1.0f, 1.0f, 1.0f}, {3});

// Присваивающие операции изменяют левый операнд
a += b;  // a становится {2.0f, 3.0f, 4.0f}
a -= b;  // a становится {1.0f, 2.0f, 3.0f}
a *= b;  // a становится {1.0f, 2.0f, 3.0f}
a /= b;  // a становится {1.0f, 2.0f, 3.0f}
```

## Требования к размерностям

### Совместимость форм

```cpp
// Операции требуют одинаковых форм
Tensor a({2, 3});  // Форма: {2, 3}
Tensor b({2, 3});  // Форма: {2, 3}
Tensor c = a + b;  // ✅ Работает

Tensor d({3, 2});  // Форма: {3, 2}
// Tensor e = a + d;  // ❌ Ошибка: формы не совпадают
```

### Обработка ошибок

```cpp
try {
    Tensor a({2, 3});
    Tensor b({3, 2});
    Tensor c = a + b;  // Вызовет исключение
} catch (const std::invalid_argument& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // "Tensor shapes don't match for addition"
}
```

## Автоматическое дифференцирование

### Градиенты арифметических операций

```cpp
// Создание тензоров с градиентами
Tensor x = Tensor({2.0f, 3.0f}, {2}, true);  // requiresGrad = true
Tensor y = Tensor({1.0f, 4.0f}, {2}, true);

// Операции с автодифференцированием
Tensor z = x + y;      // z = x + y
Tensor w = x * y;      // w = x * y
Tensor loss = w.Sum(); // Скалярная функция потерь

// Обратное распространение
loss.Backward();

// Градиенты:
// ∂loss/∂x = y = {1.0f, 4.0f}
// ∂loss/∂y = x = {2.0f, 3.0f}
const auto& x_grad = x.Grad();
const auto& y_grad = y.Grad();
```

### Правила дифференцирования

| Операция | Формула | Градиент по x | Градиент по y |
|----------|---------|---------------|---------------|
| x + y | z = x + y | ∂z/∂x = 1 | ∂z/∂y = 1 |
| x - y | z = x - y | ∂z/∂x = 1 | ∂z/∂y = -1 |
| x * y | z = x * y | ∂z/∂x = y | ∂z/∂y = x |
| x / y | z = x / y | ∂z/∂x = 1/y | ∂z/∂y = -x/y² |

## Утилитарные операции

### Заполнение тензоров

```cpp
Tensor tensor({3, 3});

// Заполнение константой
tensor.Fill(5.0f);  // Все элементы = 5.0f

// Случайные значения (нормальное распределение)
tensor.RandomNormal(0.0f, 1.0f);  // μ=0, σ=1

// Случайные значения (равномерное распределение)
tensor.RandomUniform(-1.0f, 1.0f);  // от -1 до 1
```

### Статические методы создания

```cpp
// Тензор из нулей
Tensor zeros = Tensor::Zeros({3, 4});

// Тензор из единиц
Tensor ones = Tensor::Ones({2, 5});

// Единичная матрица
Tensor eye = Tensor::Eye(4);

// Случайные тензоры
Tensor randn = Tensor::Randn({10, 10});  // Нормальное распределение
Tensor rand = Tensor::Rand({5, 5});      // Равномерное распределение
```

## Операции доступа к данным

### Индексация

```cpp
Tensor tensor = Tensor::Randn({3, 4});

// Линейная индексация
float value1 = tensor[0];           // Первый элемент
float value2 = tensor[5];           // Шестой элемент

// Многомерная индексация
float value3 = tensor.At({1, 2});   // Элемент [1,2]
float value4 = tensor.At({2, 3});   // Элемент [2,3]

// Изменение элементов
tensor[0] = 1.5f;
tensor.At({1, 2}) = 2.5f;
```

### Информация о тензоре

```cpp
Tensor tensor = Tensor::Randn({2, 3, 4});

// Размеры
size_t total_size = tensor.Size();      // 24
size_t dimensions = tensor.Ndim();      // 3
auto shape = tensor.Shape();            // {2, 3, 4}
auto strides = tensor.Strides();        // {12, 4, 1}

// Проверки
bool is_empty = tensor.Empty();         // false
```

## Операции вывода и отладки

### Вывод в консоль

```cpp
Tensor tensor = Tensor::Randn({2, 3});

// Прямой вывод
tensor.Print();

// Строковое представление
std::string str = tensor.ToString();
std::cout << str << std::endl;

// Операторный вывод
std::cout << tensor << std::endl;
```

### Форматирование вывода

```cpp
// Пример вывода:
// Tensor(shape=[2, 3], data=[0.123, -0.456, 0.789, 1.234, -2.345, 0.567])

// Для больших тензоров показываются только первые элементы:
Tensor large = Tensor::Randn({100, 100});
large.Print();
// Tensor(shape=[100, 100], data=[0.123, -0.456, 0.789, 1.234, -2.345, 0.567, 0.890, -1.123, 2.456, -0.789...])
```

## Производительность

### Оптимизация операций

> [!tip] Советы по производительности
> - Используйте присваивающие операции (`+=`, `-=`) для экономии памяти
> - Избегайте ненужных копий в цепочках операций
> - Предпочитайте операции на месте где возможно

### Пример оптимизации

```cpp
// Неэффективно - много временных объектов
Tensor result = a + b + c + d;

// Более эффективно - переиспользование
Tensor result = a;
result += b;
result += c;
result += d;

// Или с семантикой перемещения
Tensor result = std::move(a);
result += b;
result += c;
result += d;
```

## Обработка особых случаев

### Деление на ноль

```cpp
try {
    Tensor a = Tensor({1.0f, 2.0f}, {2});
    Tensor b = Tensor({0.0f, 1.0f}, {2});
    Tensor c = a / b;  // Вызовет исключение из-за деления на 0
} catch (const std::runtime_error& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // "Division by zero"
}
```

### Переполнение и потеря точности

```cpp
// Большие значения
Tensor large = Tensor::Ones({1000}) * 1e30f;
Tensor result = large + large;  // Может вызвать переполнение

// Малые значения
Tensor small = Tensor::Ones({1000}) * 1e-30f;
Tensor result2 = small + 1.0f;  // Может потерять точность
```

## Расширение операций

### Добавление кастомных операций

```cpp
// Пример кастомной операции (в будущих версиях)
namespace kotml {
namespace ops {

// Кастомная операция
Tensor CustomOperation(const Tensor& input) {
    // Реализация кастомной логики
    Tensor result = input * 2.0f + 1.0f;
    
    // Настройка автодифференцирования
    if (input.RequiresGrad()) {
        // Установка градиентной функции
        // result.SetGradFn(...);
    }
    
    return result;
}

} // namespace ops
} // namespace kotml
```

## Связанные страницы

- [[Tensor Overview]] - Основы работы с тензорами
- [[Linear Algebra]] - Операции линейной алгебры
- [[Reduction Operations]] - Операции сокращения
- [[Automatic Differentiation]] - Автоматическое дифференцирование

---

**См. также:** [[Performance Tips]], [[Error Handling]], [[Memory Management]] 