# Примеры использования библиотеки KotML

# Базовое использование тензоров
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage kotml)

# Простое автоматическое дифференцирование
add_executable(simple_autograd simple_autograd.cpp)
target_link_libraries(simple_autograd kotml)

# Демонстрация FFN (Feed-Forward Network)
add_executable(ffn_demo ffn_demo.cpp)
target_link_libraries(ffn_demo kotml)

# Пример классификации с FFN
add_executable(ffn_classification ffn_classification.cpp)
target_link_libraries(ffn_classification kotml)

# Демонстрация InputLayer и OutputLayer
add_executable(layers_demo layers_demo.cpp)
target_link_libraries(layers_demo kotml)

# Демонстрация Sequential класса
add_executable(sequential_demo sequential_demo.cpp)
target_link_libraries(sequential_demo kotml)

# Простой пример Sequential класса
add_executable(sequential_simple sequential_simple.cpp)
target_link_libraries(sequential_simple kotml)

# Сравнение FFN и Sequential
add_executable(ffn_vs_sequential ffn_vs_sequential.cpp)
target_link_libraries(ffn_vs_sequential kotml)

# Демонстрация SGD оптимизатора
add_executable(sgd_example sgd_example.cpp)
target_link_libraries(sgd_example kotml)

# Сравнение оптимизаторов SGD и Adam
add_executable(optimizer_comparison optimizer_comparison.cpp)
target_link_libraries(optimizer_comparison kotml)

# Демонстрация функций потерь
add_executable(loss_example loss_example.cpp)
target_link_libraries(loss_example kotml)

# Пример загрузчиков данных
add_executable(dataloader_example dataloader_example.cpp)
target_link_libraries(dataloader_example kotml)

# Пример работы с CSV файлами
add_executable(csv_example csv_example.cpp)
target_link_libraries(csv_example kotml)

# Пример обучения с методами Compile и Train
add_executable(training_example training_example.cpp)
target_link_libraries(training_example kotml)

# Add linear regression example
add_executable(linear_regression_example linear_regression_example.cpp)
target_link_libraries(linear_regression_example kotml)

# Simple linear test with zero initialization
add_executable(simple_linear_test simple_linear_test.cpp)
target_link_libraries(simple_linear_test kotml)

# Optimization algorithms correctness test
add_executable(optimization_test optimization_test.cpp)
target_link_libraries(optimization_test kotml)

# Progress bar demonstration
add_executable(progress_bar_demo progress_bar_demo.cpp)
target_link_libraries(progress_bar_demo kotml)

# Установка примеров (опционально)
install(TARGETS basic_usage simple_autograd ffn_demo ffn_classification layers_demo sequential_demo sequential_simple ffn_vs_sequential sgd_example optimizer_comparison loss_example dataloader_example csv_example training_example linear_regression_example simple_linear_test optimization_test progress_bar_demo
    RUNTIME DESTINATION bin/examples
) 