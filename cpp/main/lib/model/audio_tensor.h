#ifndef AUDIO_TENSOR_BASE
#define AUDIO_TENSOR_BASE
#include <string>
#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <onnxruntime_cxx_api.h>
#include "torch/torch.h"

namespace py = pybind11;

namespace audio {
    class AudioTensor
    {
    public:
        AudioTensor(bool layered = false);

        bool isLayered();
        bool isTorchTensorSet();
        bool isPyBufferInfoSet();
        bool isValueTensorSet();

        void setValueTensor(torch::Tensor&& tensor);
        void setValueTensor(py::buffer_info&& buffer);
        void setValueTensor(Ort::Value&& tensor);

        std::shared_ptr<torch::Tensor> getTorchTensor();
        Ort::Value* getValueTensor();

        int64_t* getShape();

        void prepareValueTensorInference(std::vector<Ort::Value>&& inference_setting_tensors);

    private:
        std::shared_ptr<torch::Tensor> torch_tensor;
        std::shared_ptr<py::buffer_info> py_buffer_info;
        std::shared_ptr<std::vector<Ort::Value>> value_tensors;
        bool is_layered;
        std::vector<int64_t> shape;
        bool has_torch_tensor;
        bool has_buffer_info;
        bool has_value_tensor;

        void clearAndSetTorchTensor();
    };

    template <typename T>
    Ort::Value buffer_to_tensor(py::buffer_info &data);

    AudioTensor batchConcatAudioTensors(AudioTensor& l_tensor, AudioTensor& r_tensor);

}

#endif