#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <vector>
#include "common.h"
#include "torch/torch.h"
#include "onnxruntime_cxx_api.h"

using namespace testing;
using namespace ::testing;

TEST(TorchTests, ConcatTorchTensorTest) {
    torch::Tensor l_tensor = torch::full({1, 2, 5}, 1.0);
    torch::Tensor r_tensor = torch::full({1, 2, 5}, 2.0);

    std::vector<int64_t> shape = {1, 2, 5};

    torch::Tensor res_tensor = torch::cat({l_tensor, r_tensor}, 0);

    torch::Tensor res_sum = res_tensor.sum(1).sum(1);
    EXPECT_EQ(res_tensor.sizes().size(), 3);
    EXPECT_EQ(res_tensor.sizes()[0], 2);
    EXPECT_EQ(res_tensor.sizes()[1], 2);
    EXPECT_EQ(res_tensor.sizes()[2], 5);

    float i = res_sum[0].item<float>();
    EXPECT_EQ(res_sum[0].item<float>(), 10.0);
    EXPECT_EQ(res_sum[1].item<float>(), 20.0);
    //EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}

TEST(TorchTests, rowMajorConcatValueTensorTest) {
    torch::Tensor l_tensor = torch::full({1, 2, 5}, 1.0);
    torch::Tensor r_tensor = torch::full({1, 2, 5}, 2.0);

    std::vector<int64_t> shape = {1, 2, 5};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value l_value = Ort::Value::CreateTensor<float>(
        mem_info, 
        static_cast<float*>(l_tensor.data_ptr()), 
        10, 
        shape.data(), 
        3);
    Ort::Value r_value = Ort::Value::CreateTensor<float>(
        mem_info, 
        static_cast<float*>(r_tensor.data_ptr()), 
        10, 
        shape.data(), 
        3);

    
    Ort::Value res = rowMajorValue3DConcatBatch<float>(l_value, 10, r_value, 10);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor res_tensor = torch::from_blob(
        const_cast<float*>(res.GetTensorData<float>()),
        {2, 2, 5},
        options);

    torch::Tensor res_sum = res_tensor.sum(1).sum(1);
    EXPECT_EQ(res_tensor.sizes().size(), 3);
    EXPECT_EQ(res_tensor.sizes()[0], 2);
    EXPECT_EQ(res_tensor.sizes()[1], 2);
    EXPECT_EQ(res_tensor.sizes()[2], 5);

    float i = res_sum[0].item<float>();
    EXPECT_EQ(res_sum[0].item<float>(), 10.0);
    EXPECT_EQ(res_sum[1].item<float>(), 20.0);
    //EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}


TEST(TorchTests, ConcatValueTensorTest) {
    torch::Tensor l_tensor = torch::full({1, 2, 5}, 1.0);
    torch::Tensor r_tensor = torch::full({1, 2, 5}, 2.0);

    std::vector<int64_t> shape = {1, 2, 5};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value l_value = Ort::Value::CreateTensor<float>(
        mem_info, 
        static_cast<float*>(l_tensor.data_ptr()), 
        10, 
        shape.data(), 
        3);
    Ort::Value r_value = Ort::Value::CreateTensor<float>(
        mem_info, 
        static_cast<float*>(r_tensor.data_ptr()), 
        10, 
        shape.data(), 
        3);

    Ort::Value res = pyArrayToTorchAndConcat(l_value, r_value);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor res_tensor = torch::from_blob(
        const_cast<float*>(res.GetTensorData<float>()),
        {2, 2, 5},
        options);

    torch::Tensor res_sum = res_tensor.sum(1).sum(1);
    EXPECT_EQ(res_tensor.sizes().size(), 3);
    EXPECT_EQ(res_tensor.sizes()[0], 2);
    EXPECT_EQ(res_tensor.sizes()[1], 2);
    EXPECT_EQ(res_tensor.sizes()[2], 5);

    float i = res_sum[0].item<float>();
    EXPECT_EQ(res_sum[0].item<float>(), 10.0);
    EXPECT_EQ(res_sum[1].item<float>(), 20.0);
    //EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}



