# Тесты для KotML с использованием Google Test

# Поиск Google Test
find_package(GTest REQUIRED)

# Включаем директории заголовков
include_directories(${CMAKE_SOURCE_DIR}/include)

# Тесты для тензорных операций
add_executable(tensor_tests
    test_simple.cpp
    test_tensor_basic.cpp
    test_tensor_arithmetic.cpp
    test_tensor_linalg.cpp
    test_tensor_reduction.cpp
    test_tensor_autograd.cpp
)

target_link_libraries(tensor_tests
    kotml
    GTest::gtest
    GTest::gtest_main
    pthread
)

# Добавляем тесты в CTest
include(GoogleTest)
gtest_discover_tests(tensor_tests)

# Тесты для нейронных сетей
add_executable(nn_tests
    test_ffn.cpp
    test_sequential.cpp
    test_ffn_vs_sequential.cpp
)

target_link_libraries(nn_tests
    kotml
    GTest::gtest
    GTest::gtest_main
    pthread
)

gtest_discover_tests(nn_tests)

# Тесты для оптимизаторов
add_executable(optim_tests
    test_optimizers.cpp
)

target_link_libraries(optim_tests
    kotml
    GTest::gtest
    GTest::gtest_main
    pthread
)

gtest_discover_tests(optim_tests) 