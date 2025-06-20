# Tensor Operations

Подробное описание всех операций, доступных для класса `Tensor`.

## Арифметические операции

### Операции между тензорами

```cpp
Tensor a({2.0f, 4.0f}, {2});
Tensor b({1.0f, 2.0f}, {2});

// Сложение
Tensor sum = a + b;        // {3.0f, 6.0f}

// Вычитание
Tensor diff = a - b;       // {1.0f, 2.0f}

// Умножение (поэлементное)
Tensor prod = a * b;       // {2.0f, 8.0f}

// Деление
Tensor quot = a / b;       // {2.0f, 2.0f}
```

### Операции со скалярами

```cpp
Tensor tensor({1.0f, 2.0f, 3.0f}, {3});

// Сложение со скаляром
Tensor result1 = tensor + 5.0f;    // {6.0f, 7.0f, 8.0f}
Tensor result2 = 5.0f + tensor;    // {6.0f, 7.0f, 8.0f}

// Вычитание скаляра
Tensor result3 = tensor - 1.0f;    // {0.0f, 1.0f, 2.0f}
Tensor result4 = 10.0f - tensor;   // {9.0f, 8.0f, 7.0f}

// Умножение на скаляр
Tensor result5 = tensor * 2.0f;    // {2.0f, 4.0f, 6.0f}
Tensor result6 = 3.0f * tensor;    // {3.0f, 6.0f, 9.0f}

// Деление на скаляр
Tensor result7 = tensor / 2.0f;    // {0.5f, 1.0f, 1.5f}
Tensor result8 = 6.0f / tensor;    // {6.0f, 3.0f, 2.0f}
```

### Присваивающие операции

```cpp
Tensor a({1.0f, 2.0f}, {2});
Tensor b({3.0f, 4.0f}, {2});

a += b;  // a становится {4.0f, 6.0f}
a -= b;  // a становится {1.0f, 2.0f}
a *= b;  // a становится {3.0f, 8.0f}
a /= b;  // a становится {1.0f, 2.0f}
```

## Линейная алгебра

### Матричное умножение

```cpp
// Создание матриц
Tensor A({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});  // 2x2 матрица
Tensor B({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});  // 2x2 матрица

// Матричное умножение
Tensor C = A.Matmul(B);
```

### Транспонирование

```cpp
Tensor matrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
Tensor transposed = matrix.Transpose();  // Форма становится {3, 2}
```

### Изменение формы

```cpp
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

// Изменение формы (общий размер должен совпадать)
Tensor reshaped1 = tensor.Reshape({3, 2});  // 3x2
Tensor reshaped2 = tensor.Reshape({6});     // Вектор
Tensor reshaped3 = tensor.Reshape({1, 6});  // 1x6
```

## Операции сокращения

### Суммирование

```cpp
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

// Сумма всех элементов
Tensor total_sum = tensor.Sum();  // Скаляр: 21.0f

// Сумма по оси
Tensor sum_axis0 = tensor.Sum(0);  // Сумма по строкам: {5.0f, 7.0f, 9.0f}
Tensor sum_axis1 = tensor.Sum(1);  // Сумма по столбцам: {6.0f, 15.0f}
```

### Среднее значение

```cpp
Tensor tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

// Среднее всех элементов
Tensor total_mean = tensor.Mean();  // Скаляр: 2.5f

// Среднее по оси
Tensor mean_axis0 = tensor.Mean(0);  // {2.5f, 3.5f}
Tensor mean_axis1 = tensor.Mean(1);  // {1.5f, 3.5f}
```

## Утилиты

### Заполнение значениями

```cpp
Tensor tensor({3, 3});

// Заполнение константой
tensor.Fill(5.0f);  // Все элементы становятся 5.0f

// Случайные значения (нормальное распределение)
tensor.RandomNormal(0.0f, 1.0f);  // Среднее=0, стандартное отклонение=1

// Случайные значения (равномерное распределение)
tensor.RandomUniform(-1.0f, 1.0f);  // От -1 до 1
```

### Вывод и отладка

```cpp
Tensor tensor = Tensor::Randn({2, 3});

// Вывод в консоль
tensor.Print();

// Получение строкового представления
std::string str = tensor.ToString();

// Вывод в поток
std::cout << tensor << std::endl;
```

## Требования к размерностям

### Арифметические операции
- Тензоры должны иметь одинаковую форму
- Исключение: операции со скалярами

### Матричное умножение
- Первый тензор: форма `{m, n}`
- Второй тензор: форма `{n, p}`
- Результат: форма `{m, p}`

### Операции сокращения
- Ось должна быть в пределах размерности тензора
- Результирующая форма уменьшается на одну размерность

## Обработка ошибок

```cpp
try {
    Tensor a({2, 3});
    Tensor b({3, 2});
    
    // Это вызовет исключение - формы не совпадают
    Tensor c = a + b;
} catch (const std::invalid_argument& e) {
    std::cout << "Error: " << e.what() << std::endl;
}
```

## Производительность

> [!tip] Оптимизация
> - Используйте семантику перемещения для больших тензоров
> - Избегайте ненужных копий в цепочках операций
> - Предпочитайте присваивающие операции (`+=`, `-=`, etc.) где возможно

## Связанные страницы

- [[Tensor Overview]] - Обзор класса Tensor
- [[Automatic Differentiation]] - Автоматическое дифференцирование
- [[Linear Algebra]] - Подробнее о линейной алгебре
- [[Basic Examples]] - Примеры использования

---

**См. также:** [[API Reference]], [[Performance Tips]], [[Error Handling]] 