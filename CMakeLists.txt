cmake_minimum_required(VERSION 3.12)
project(KotML VERSION 1.0.0 LANGUAGES CXX)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опции компиляции
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Включение директорий
include_directories(include)

# Исходные файлы
set(SOURCES
    src/tensor.cpp
    src/ops/basic.cpp
    src/ops/linalg.cpp
    src/ops/reduction.cpp
    src/nn/module.cpp
    src/nn/layers.cpp
    src/nn/input_layer.cpp
    src/nn/linear_layer.cpp
    src/nn/activation_layer.cpp
    src/nn/dropout_layer.cpp
    src/nn/output_layer.cpp
    src/nn/ffn.cpp
    src/nn/sequential.cpp
    src/nn/activations.cpp
    src/nn/loss.cpp
    src/optim/sgd.cpp
    src/optim/adam.cpp
    src/utils/helpers.cpp
    src/utils/progress_bar.cpp
)

# Создание библиотеки
add_library(kotml STATIC ${SOURCES})

# Заголовочные файлы для установки
target_include_directories(kotml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Поиск зависимостей и связывание
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(kotml PUBLIC OpenMP::OpenMP_CXX)
endif()

# Опция для включения тестов
option(BUILD_TESTS "Build tests" ON)

# Добавляем тесты, если они включены
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Опции для примеров
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Debug executables
add_executable(debug_chain debug_chain.cpp)
target_link_libraries(debug_chain kotml) 