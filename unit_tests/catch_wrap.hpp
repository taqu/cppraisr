#ifndef INC_CATCH_WRAP_HPP
#define INC_CATCH_WRAP_HPP
#include "catch_amalgamated.hpp"
#include <iostream>

namespace util
{
    template<class T>
    inline T absolute(T val)
    {
        return 0<=val? val : -val;
    }

    inline bool isEqual(float x1, float x2)
    {
        return (util::absolute<float>(x1 - x2) <= 1.0e-6f);
    }

    inline bool isEqual(float x1, float x2, float epsilon)
    {
        return (util::absolute<float>(x1 - x2) <= epsilon);
    }


    inline bool isEqual(double x1, double x2)
    {
        return (util::absolute<double>(x1 - x2) <= 1.0e-15);
    }

    inline bool isEqual(double x1, double x2, double epsilon)
    {
        return (util::absolute<double>(x1 - x2) <= epsilon);
    }
}

#define EXPECT_TRUE(expression) CHECK((expression))
#define EXPECT_FALSE(expression) CHECK_FALSE((expression))
#define EXPECT_EQ(expected, actual) CHECK((expected) == (actual))
#define EXPECT_NE(expected, actual) CHECK((expected) != (actual))
#define EXPECT_FLOAT_EQ(expected, actual) CHECK(util::isEqual((expected), (actual)))
#define EXPECT_FLOAT_NEAR(expected, actual, epsilon) CHECK(util::isEqual((expected), (actual), (epsilon)))
#define EXPECT_STR_EQ(expected, actual) CHECK(0 == ::strcmp((expected), (actual)))
#define EXPECT_GT(expected, actual) CHECK((expected)<(actual))
#define EXPECT_GE(expected, actual) CHECK((expected)<=(actual))
#define EXPECT_LT(expected, actual) CHECK((expected)<(actual))
#define EXPECT_LE(expected, actual) CHECK((expected)<(actual))

#define LOG_INFO(message) std::cout << message << std::endl
#define LOG_INFO_PUT(message) std::cout << message

#endif //INC_CATCH_WRAP_HPP
