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

template <typename T>
Ort::Value rowMajorValue3DConcatBatch(Ort::Value& l_value, int64_t l_nelem, Ort::Value& r_value, int64_t r_nelem) {
    const std::vector<int64_t> ldata_shape = l_value.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> rdata_shape = r_value.GetTensorTypeAndShapeInfo().GetShape();

    T* l_value_ptr = const_cast<T*>(l_value.GetTensorData<T>());
    T* r_value_ptr = const_cast<T*>(r_value.GetTensorData<T>());

    assert(ldata_shape.size() == 3 && rdata_shape.size() == 3);
    assert(std::reduce(ldata_shape.begin(), ldata_shape.end()) > 0);
    assert(std::reduce(rdata_shape.begin(), rdata_shape.end()) > 0);

    int64_t total_batch = ldata_shape[0] + rdata_shape[0];
    T* concat_array = new T[l_nelem + r_nelem];

    T* ptr = concat_array;
    for (int row = 0; row < ldata_shape[1]; row++) {
        for (int col = 0; col < ldata_shape[2]; col++) {
            for (int batch = 0; batch < ldata_shape[0]; batch++) {
                *ptr = *l_value_ptr;
                ptr++;
                l_value_ptr++;
            }

            for (int batch = 0; batch < rdata_shape[0]; batch++) {
                *ptr = *r_value_ptr;
                ptr++;
                r_value_ptr++;
            }
        }
    }

    std::vector<int64_t> shape = {total_batch, ldata_shape[1], ldata_shape[2]};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value concat_tensor = Ort::Value::CreateTensor<T>(
        mem_info, 
        concat_array, 
        (size_t) shape[0] * shape[1] * shape[2], 
        (int64_t *)shape.data(), 
        3);

    return concat_tensor;

}

json parseJSON(std::string file) {
    std::ifstream f(file);
    json data = json::parse(f);
    return data;
};