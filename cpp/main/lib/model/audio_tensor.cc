#include "audio_tensor.h"

namespace audio {
    AudioTensor::AudioTensor(bool layered) : is_layered(layered) {
        value_tensors = std::make_shared<std::vector<Ort::Value>>();
    }

    bool AudioTensor::isLayered() {
        return is_layered;
    }

    int64_t* AudioTensor::getShape() {
        return shape.data();
    }

    void AudioTensor::setValueTensor(torch::Tensor&& tensor) {
        torch_tensor = std::make_shared<torch::Tensor>(std::move(tensor));
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        auto sizes = torch_tensor->sizes();
        shape.assign(sizes.begin(), sizes.end());

        value_tensors->emplace_back(Ort::Value::CreateTensor<float>(
            mem_info, 
            static_cast<float*>(torch_tensor->data_ptr()), 
            (size_t) torch::numel(*torch_tensor), 
            shape.data(), 
            sizes.size()));

        has_torch_tensor = true;
        has_value_tensor = true;
    }

    void AudioTensor::setValueTensor(py::buffer_info&& buffer) {
        py_buffer_info = std::make_shared<py::buffer_info>(std::move(buffer));
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        value_tensors->emplace_back(Ort::Value::CreateTensor<float>(
            mem_info, 
            (float* ) py_buffer_info->ptr, 
            py_buffer_info->size, 
            (int64_t *)py_buffer_info->shape.data(), 
            py_buffer_info->ndim));

        for (auto& i : py_buffer_info->shape) {
            shape.emplace_back((int64_t) i);
        }

        has_buffer_info = true;
        has_value_tensor = true;
    }

    void AudioTensor::clearAndSetTorchTensor() {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor torch_tensor = torch::from_blob(
            const_cast<float*>((*value_tensors)[0].GetTensorData<float>()),
            {shape[0], shape[1], shape[2]},
            options);
        value_tensors->clear();
        setValueTensor(std::move(torch_tensor));
    }


    std::shared_ptr<torch::Tensor> AudioTensor::getTorchTensor() {
        if (!has_value_tensor) {
            std::cerr << "Value tensor not set by torch tensor. Torch tensor not found." << std::endl;
            exit(1);
        }
        else if (has_torch_tensor) {
            return torch_tensor;
        }

        clearAndSetTorchTensor();
        return torch_tensor;
    }
    
    Ort::Value* AudioTensor::getValueTensor() {
        return value_tensors->data();
    }

    bool AudioTensor::isTorchTensorSet() {
        return has_torch_tensor;
    }

    bool AudioTensor::isPyBufferInfoSet() {
        return has_buffer_info;
    }

    bool AudioTensor::isValueTensorSet() {
        return has_value_tensor;
    }

    void AudioTensor::prepareValueTensorInference(std::vector<Ort::Value>&& inference_setting_tensors) {
        value_tensors->insert(
            value_tensors->end(), 
            std::make_move_iterator(inference_setting_tensors.begin()),
            std::make_move_iterator(inference_setting_tensors.end()));
    }

    template <typename T>
    Ort::Value buffer_to_tensor(py::buffer_info &data)
    {
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        float* tensor_data = new float[data.size];
        
        std::memcpy(tensor_data, (float* ) data.ptr, sizeof(float) * data.size);
        // std::cout << "Tensor numbers: " << data.data() << " " << data.size() << " " << shape.data() << " " << shape.size() << std::endl;
        Ort::Value tensor = Ort::Value::CreateTensor<T>(mem_info, tensor_data, data.size, (int64_t *)data.shape.data(), data.ndim);
        return tensor;
    }

    AudioTensor batchConcatAudioTensors(AudioTensor& l_tensor, AudioTensor& r_tensor) {
        torch::Tensor cat_tensor = torch::cat({*(l_tensor.getTorchTensor()), *(r_tensor.getTorchTensor())}, 0);
        AudioTensor concat_audio_tensor = AudioTensor(true);
        concat_audio_tensor.setValueTensor(std::move(cat_tensor));
        return concat_audio_tensor;
    }


}