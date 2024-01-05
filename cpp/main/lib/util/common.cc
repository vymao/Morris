#include "common.h"
#include <tuple>
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

Ort::Value pyArrayToTorchAndConcat(py::array_t<float>& left_mat, std::tuple<int> left_size, py::array_t<float>& right_mat, std::tuple<int> right_size) {
    py::buffer_info ldata_buf = left_mat.request();
    py::buffer_info rdata_buf = right_mat.request();
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor left_tensor = torch::from_blob(
        static_cast<float*>(ldata_buf.ptr),
        {ldata_buf.shape[0], ldata_buf.shape[1], ldata_buf.shape[2]},
        options);
    torch::Tensor right_tensor = torch::from_blob(
        static_cast<float*>(rdata_buf.ptr),
        {rdata_buf.shape[0], rdata_buf.shape[1], rdata_buf.shape[2]},
        options);
    
    torch::Tensor cat_tensor = torch::cat({left_tensor, right_tensor}, 0);
    std::vector<ssize_t> shape(ldata_buf.shape);
    shape[0] = 2;

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem_info, 
        static_cast<float*>(cat_tensor.data_ptr()), 
        ldata_buf.size + rdata_buf.size, 
        (int64_t *)shape.data(), 
        ldata_buf.ndim);
    return tensor;
}

json parseJSON(std::string file) {
    std::ifstream f(file);
    json data = json::parse(f);
    return data;
};