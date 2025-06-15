# Automatic Differentiation

Автоматическое дифференцирование (автодифференцирование) в KotML позволяет автоматически вычислять градиенты для обучения нейронных сетей.

## Основные концепции

### Вычислительный граф
KotML строит вычислительный граф операций для отслеживания зависимостей между тензорами и автоматического вычисления градиентов.

### Режим обратного распространения
Используется режим обратного распространения (reverse mode), который эффективен для функций с множеством входов и небольшим количеством выходов.

## Включение градиентов

### При создании тензора

```cpp
// Тензор с градиентами
Tensor x = Tensor::Randn({3, 3}, true);  // requiresGrad = true

// Тензор без градиентов
Tensor y = Tensor::Randn({3, 3}, false); // requiresGrad = false (по умолчанию)
```

### Управление градиентами

```cpp
Tensor tensor = Tensor::Randn({2, 2});

// Проверка требования градиентов
bool requires_grad = tensor.RequiresGrad();

// Включение/выключение градиентов
tensor.SetRequiresGrad(true);
tensor.SetRequiresGrad(false);
```

## Вычисление градиентов

### Базовый пример

```cpp
// Создание тензора с градиентами
Tensor x = Tensor::Randn({2, 2}, true);

// Вычисления (строится граф)
Tensor y = x * x;           // y = x²
Tensor z = y + x;           // z = x² + x
Tensor loss = z.Sum();      // Скалярная функция потерь

// Обратное распространение
loss.Backward();

// Получение градиентов
const auto& grad = x.Grad();  // ∂loss/∂x = 2x + 1
```

### Сложный пример

```cpp
Tensor a = Tensor::Randn({3, 3}, true);
Tensor b = Tensor::Randn({3, 3}, true);

// Сложные вычисления
Tensor c = a.Matmul(b);
Tensor d = c * 2.0f + a;
Tensor e = d.Sum();

// Вычисление градиентов
e.Backward();

// Градиенты доступны в a.Grad() и b.Grad()
```

## Управление градиентами

### Обнуление градиентов

```cpp
Tensor x = Tensor::Randn({2, 2}, true);

// После вычислений градиенты накапливаются
// Необходимо обнулять перед новой итерацией
x.ZeroGrad();
```

### Доступ к градиентам

```cpp
Tensor x = Tensor::Randn({2, 2}, true);
// ... вычисления и backward ...

// Константный доступ
const auto& grad_const = x.Grad();

// Изменяемый доступ
auto& grad_mut = x.Grad();
grad_mut[0] = 0.5f;  // Ручное изменение градиента
```

## Правила дифференцирования

### Арифметические операции

```cpp
Tensor x = Tensor::Randn({2}, true);
Tensor y = Tensor::Randn({2}, true);

// Сложение: ∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1
Tensor sum = x + y;

// Умножение: ∂(x*y)/∂x = y, ∂(x*y)/∂y = x
Tensor prod = x * y;

// Скалярное умножение: ∂(c*x)/∂x = c
Tensor scaled = x * 2.0f;
```

### Матричные операции

```cpp
Tensor A = Tensor::Randn({3, 4}, true);
Tensor B = Tensor::Randn({4, 2}, true);

// Матричное умножение
Tensor C = A.Matmul(B);  // C = A @ B
// ∂C/∂A = B^T, ∂C/∂B = A^T
```

### Операции сокращения

```cpp
Tensor x = Tensor::Randn({3, 3}, true);

// Суммирование: градиент распространяется на все элементы
Tensor sum_all = x.Sum();        // ∂sum/∂x[i,j] = 1

// Суммирование по оси
Tensor sum_axis = x.Sum(0);      // Градиент по соответствующим элементам

// Среднее значение
Tensor mean_all = x.Mean();      // ∂mean/∂x[i,j] = 1/size
```

## Практические примеры

### Линейная регрессия

```cpp
// Данные
Tensor X = Tensor::Randn({100, 2}, false);  // Входные данные
Tensor y = Tensor::Randn({100, 1}, false);  // Целевые значения

// Параметры модели
Tensor W = Tensor::Randn({2, 1}, true);     // Веса
Tensor b = Tensor::Randn({1}, true);        // Смещение

// Прямой проход
Tensor predictions = X.Matmul(W) + b;
Tensor diff = predictions - y;
Tensor loss = (diff * diff).Mean();         // MSE Loss

// Обратный проход
loss.Backward();

// Градиенты доступны в W.Grad() и b.Grad()
```

### Простая нейронная сеть

```cpp
// Параметры
Tensor W1 = Tensor::Randn({784, 128}, true);
Tensor b1 = Tensor::Randn({128}, true);
Tensor W2 = Tensor::Randn({128, 10}, true);
Tensor b2 = Tensor::Randn({10}, true);

// Входные данные
Tensor x = Tensor::Randn({32, 784}, false);  // Batch size = 32

// Прямой проход
Tensor h1 = x.Matmul(W1) + b1;
// Здесь должна быть функция активации (ReLU)
Tensor output = h1.Matmul(W2) + b2;

// Функция потерь
Tensor loss = output.Sum();  // Упрощенная функция потерь

// Обратный проход
loss.Backward();

// Градиенты доступны во всех параметрах с requiresGrad=true
```

## Ограничения и особенности

### Текущие ограничения

> [!warning] Ограничения
> - Поддерживается только режим обратного распространения
> - Градиенты накапливаются - требуется ручное обнуление
> - Нет автоматической оптимизации вычислительного графа
> - Ограниченная поддержка функций активации в автодифференцировании

### Рекомендации по использованию

> [!tip] Лучшие практики
> - Всегда обнуляйте градиенты перед новой итерацией обучения
> - Используйте `requiresGrad=true` только для обучаемых параметров
> - Вызывайте `Backward()` только для скалярных тензоров
> - Избегайте циклических зависимостей в вычислительном графе

## Внутренняя реализация

### Структура градиентного графа

```cpp
class Tensor {
private:
    std::vector<float> m_grad;                          // Градиенты
    bool m_requiresGrad;                                // Флаг вычисления градиентов
    std::vector<std::shared_ptr<Tensor>> m_gradParents; // Родительские тензоры
    GradFunction m_gradFn;                              // Функция вычисления градиентов
};
```

### Функции градиентов

```cpp
using GradFunction = std::function<std::vector<Tensor>(const Tensor&)>;

// Пример функции градиента для сложения
auto add_grad_fn = [](const Tensor& grad_output) -> std::vector<Tensor> {
    return {grad_output, grad_output};  // Градиент для обоих операндов
};
```

## Связанные страницы

- [[Tensor Overview]] - Основы работы с тензорами
- [[Neural Networks Overview]] - Использование в нейронных сетях
- [[Optimizers Overview]] - Оптимизаторы для обновления параметров
- [[Basic Examples]] - Примеры использования

---

**См. также:** [[Backpropagation]], [[Gradient Descent]], [[Training Loop]] 