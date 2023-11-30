// myextension.cpp

#include <pybind11/embed.h> 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "feature_extractor.h"

py::buffer_info getBufferInfoForMatrix(Matrix& m) {
    return py::buffer_info(
                m.data(),                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                1,                                      /* Number of dimensions */
                { m.size() },                 /* Buffer dimensions */
                { sizeof(float) * m.size()}             /* Strides (in bytes) for each index */
                
        );
}

py::array_t<float> getFloatArrayforMatrix(Matrix& m) {
    py::str dummyDataOwner;
    py::buffer_info buffer = getBufferInfoForMatrix(m);
    return py::array_t<float>(
        buffer, dummyDataOwner
    );
}

std::vector<Ort::Value> rawAudioToValueVector(std::vector<float>& float_vector, py::object& extractor) {
    py::array_t<float> data = py::array_t<float>(py::cast(float_vector));

    py::object rdict = extractor.attr("__call__")(data, 16000, "np");
    py::array_t<float> rdata = rdict.attr("get")("input_values");
    py::buffer_info rdata_buf = rdata.request();
    const auto rdata_size = rdata_buf.shape;
    //float *carray = rdata.mutable_data();

    std::vector<Ort::Value>  input_tensors;
    input_tensors.emplace_back(buffer_to_tensor<float>(rdata_buf));
    return input_tensors;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //std::cout << "Tensor numbers: " << data.data() << " " << data.size() << " " << shape.data() << " " << shape.size() << std::endl;
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}

template <typename T>
Ort::Value buffer_to_tensor(py::buffer_info &data)
{
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    //std::cout << "Tensor numbers: " << data.data() << " " << data.size() << " " << shape.data() << " " << shape.size() << std::endl;
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, (T*) data.ptr, data.size, (int64_t*) data.shape.data(), data.ndim);
    return tensor;
}

