#include "catch_amalgamated.hpp"
#include "catch_wrap.hpp"
#include "util.h"

TEST_CASE("Transpose")
{
    using namespace cppraisr;
    int32_t size = 3;
    double m[9] = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
    };
    double r[9];
    transpose(3, r, m);
    EXPECT_FLOAT_EQ(m[0], r[0]);
    EXPECT_FLOAT_EQ(m[1], r[3]);
    EXPECT_FLOAT_EQ(m[2], r[6]);
    EXPECT_FLOAT_EQ(m[4], r[4]);
    EXPECT_FLOAT_EQ(m[5], r[7]);
    EXPECT_FLOAT_EQ(m[8], r[8]);
}

TEST_CASE("Mul00")
{
    using namespace cppraisr;
    double m0[4] = {1, 2, 3, 4};
    double m1[4] = {1, 2, 3, 4};
    double r[4];
    mul(2, r, m0, m1);
    EXPECT_FLOAT_EQ(7, r[0]);
    EXPECT_FLOAT_EQ(10, r[1]);
    EXPECT_FLOAT_EQ(15, r[2]);
    EXPECT_FLOAT_EQ(22, r[3]);
}

TEST_CASE("Mul01")
{
    using namespace cppraisr;
    double m0[4] = {1, 2, 3, 4};
    mul(2, m0, 2);
    EXPECT_FLOAT_EQ(2, m0[0]);
    EXPECT_FLOAT_EQ(4, m0[1]);
    EXPECT_FLOAT_EQ(6, m0[2]);
    EXPECT_FLOAT_EQ(8, m0[3]);
}

TEST_CASE("Square")
{
    using namespace cppraisr;
    double m[4] = {1, 2, 3, 4};
    double r[4];
    square(2, r, m);
    EXPECT_FLOAT_EQ(1, r[0]);
    EXPECT_FLOAT_EQ(4, r[1]);
    EXPECT_FLOAT_EQ(9, r[2]);
    EXPECT_FLOAT_EQ(16, r[3]);
}

TEST_CASE("Solv2x2")
{
    using namespace cppraisr;
    double m[4] = {8, 1, 4, 5};
    double evalues[2];
    double evectors[4];
    solv2x2(evalues, evectors, m);
    EXPECT_FLOAT_EQ(9, evalues[0]);
    EXPECT_FLOAT_EQ(4, evalues[1]);

    EXPECT_FLOAT_EQ(1, evectors[0]);
    EXPECT_FLOAT_EQ(1, evectors[1]);
    EXPECT_FLOAT_EQ(1, evectors[2]);
    EXPECT_FLOAT_EQ(-4, evectors[3]);
}

