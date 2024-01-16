#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include "generation.h"

using namespace testing;
using namespace ::testing;
using namespace generation;

TEST(GenerationTest, queryServerlessLLM)
{
    VirtualTextGenerator generator;

    std::string query = "Explain to me in one sentence what is the meaning of life.";

    generator.queryVirtualLLM(query);
    
    //std::cout << res << std::endl;
    EXPECT_TRUE(generator.streamed_res.size() > 0);
    while (generator.streamed_res.size()) {
        std::cout << generator.streamed_res.front() << std::endl;
        generator.streamed_res.pop();
    }
    // EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}


