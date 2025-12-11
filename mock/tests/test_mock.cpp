#include "gtest/gtest.h"
#include "mock/mock.h"

namespace mock {

TEST(TestMock, TestMock) {
    Mock example("example");
    EXPECT_EQ(example.Get(), "example");
}

}  // namespace mock
