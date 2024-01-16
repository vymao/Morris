#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <vector>
#include "audio_tensor.h"
#include "torch/torch.h"
#include "onnxruntime_cxx_api.h"

using namespace testing;
using namespace ::testing;
using namespace audio;

TEST(AudioTensorTest, torchTensorStore)
{
    torch::Tensor tensor = torch::full({1, 2, 5}, 1.0);

    AudioTensor audio_tensor = AudioTensor();
    audio_tensor.setValueTensor(std::move(tensor));
    std::shared_ptr<torch::Tensor> stored_tensor = audio_tensor.getTorchTensor();

    torch::Tensor res_sum = stored_tensor->sum(1).sum(1);
    EXPECT_EQ(stored_tensor->sizes().size(), 3) << "Not size 3.";
    EXPECT_EQ(stored_tensor->sizes()[0], 1) << "Dim 1 incorrect";
    EXPECT_EQ(stored_tensor->sizes()[1], 2) << "Dim 2 incorrect";
    EXPECT_EQ(stored_tensor->sizes()[2], 5) << "Dim 3 incorrect";

    EXPECT_EQ(res_sum[0].item<float>(), 10.0) << "Sum incorrect";
    // EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}

TEST(AudioTensorTest, audioTensorCat)
{
    torch::Tensor tensor_1 = torch::full({1, 2, 5}, 1.0);
    torch::Tensor tensor_2 = torch::full({1, 2, 5}, 2.0);

    AudioTensor audio_tensor_1 = AudioTensor();
    AudioTensor audio_tensor_2 = AudioTensor();
    audio_tensor_1.setValueTensor(std::move(tensor_1));
    audio_tensor_2.setValueTensor(std::move(tensor_2));

    AudioTensor audio_tensor_cat = batchConcatAudioTensors(audio_tensor_1, audio_tensor_2);
    std::shared_ptr<torch::Tensor> torch_tensor_cat = audio_tensor_cat.getTorchTensor();

    torch::Tensor res_sum = torch_tensor_cat->sum(1).sum(1);
    EXPECT_EQ(res_sum.sizes().size(), 1);
    EXPECT_EQ(res_sum.sizes()[0], 2);
    EXPECT_EQ(res_sum[0].item<float>(), 10.0);
    EXPECT_EQ(res_sum[1].item<float>(), 20.0);
    EXPECT_TRUE((res_sum[1].item().toSymFloat() > res_sum[0].item().toSymFloat()));
}

