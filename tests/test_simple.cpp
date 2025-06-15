/**
 * Простой тест для проверки работы Google Test
 */

#include <gtest/gtest.h>

// Простой тест для проверки работы системы тестирования
TEST(SimpleTest, BasicAssertion) {
    EXPECT_EQ(2 + 2, 4);
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
}

TEST(SimpleTest, StringComparison) {
    std::string hello = "Hello";
    std::string world = "World";
    
    EXPECT_EQ(hello, "Hello");
    EXPECT_NE(hello, world);
    EXPECT_LT(hello.length(), 10);
}

TEST(SimpleTest, FloatingPoint) {
    float a = 0.1f + 0.2f;
    float b = 0.3f;
    
    EXPECT_NEAR(a, b, 1e-6f);
    EXPECT_FLOAT_EQ(1.0f, 1.0f);
    EXPECT_DOUBLE_EQ(1.0, 1.0);
} 