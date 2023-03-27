#include "catch_amalgamated.hpp"
#include "catch_wrap.hpp"
#include "util.h"

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

