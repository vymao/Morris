#include "common.h"
#include <fstream>
#include "torch/torch.h"

std::string print_shape(const std::vector<std::int64_t> &v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int ortValueToTorchAndArgmax(Ort::Value& value_tensor) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor torch_tensor = torch::from_blob(
        const_cast<float*>(value_tensor.GetTensorData<float>()),
        {1, 35},
        options);
    return torch_tensor.argmax(-1).item().toInt();
}

json parseJSON(std::string file) {
    std::ifstream f(file);
    json data = json::parse(f);
    return data;
};