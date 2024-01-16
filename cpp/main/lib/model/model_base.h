#ifndef AUDIO_BASE
#define AUDIO_BASE

#include <string>
#include <queue>
#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <onnxruntime_cxx_api.h>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "audio_tensor.h"

namespace py = pybind11;

class AudioModelBase
{
public:
    AudioModelBase(std::string model,
                   std::shared_ptr<Ort::Session> new_session,
                   std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
                   py::object &extractor);
    virtual ~AudioModelBase() = 0;
    virtual std::vector<Ort::Value> runModelSync(std::vector<Ort::Value> &input_tensors);
    virtual void runModelAsync() = 0;
    virtual std::shared_ptr<audio::AudioTensor> prepareInputs(std::vector<float> &input_values) = 0;
    virtual void prepareInputsAndPush(std::shared_ptr<std::vector<float>> input_values) = 0;
    virtual bool isReadyForRun() = 0;

    virtual std::string getTotalOutput() {};
    std::shared_ptr<Ort::Session> getSession();

    bool has_secondary_data_queue;
    std::thread::id caller_tid;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

protected:
    std::string model_name;
    std::shared_ptr<Ort::Session> session;
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> allocator;
    py::object feature_extractor;
    std::vector<const char *> input_names_arrays;
    std::vector<const char *> output_names_arrays;
    std::queue<std::shared_ptr<audio::AudioTensor>> data_queue;
    std::queue<std::shared_ptr<std::vector<Ort::Value>>> out_value_queue;



    template <typename Function>
    std::vector<std::string> getInputOrOutputNames(size_t size, Function name_allocator_func);
    std::vector<const char *> getInputOrOutputNameArray(std::vector<std::string> &name_vector);
    virtual std::shared_ptr<audio::AudioTensor> audioToValueVector(std::vector<float> &float_vector, py::object &extractor) = 0;
    
};

#endif
