#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "main/lib/util/common.h"
#include "wake_classifier.h"
#include "main/lib/util/feature_extractor.h"

using namespace std::placeholders;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;


std::atomic_bool WakeClassifier::atomic_wait = false;
std::atomic_bool WakeClassifier::wakeup = false;

WakeClassifier::WakeClassifier(
    std::string model,
    std::shared_ptr<Ort::Session> new_session,
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
    py::object &extractor) : AudioModelBase(model, new_session, global_allocator, extractor)
{}

int WakeClassifier::getNumOutputNames() {
    return (int) output_names_arrays.size();
}

std::shared_ptr<AudioTensor> WakeClassifier::audioToValueVector(std::vector<float> &float_vector, py::object &extractor)
{
    py::array_t<float> data = py::array_t<float>(py::cast(float_vector));

    py::object rdict = extractor.attr("__call__")(data, 16000, "np");
    py::array_t<float> rdata = rdict.attr("get")("input_values");
    py::buffer_info rdata_buf = rdata.request();
    const auto rdata_size = rdata_buf.shape;
    // float *carray = rdata.mutable_data();

    std::shared_ptr<AudioTensor> audio_tensor = std::make_shared<AudioTensor>();
    audio_tensor->setValueTensor(std::move(rdata_buf));
    return audio_tensor;
}

std::shared_ptr<AudioTensor> WakeClassifier::prepareInputs(std::vector<float> &input_values) {
    return audioToValueVector(input_values, feature_extractor);
}

void WakeClassifier::prepareInputsAndPush(std::shared_ptr<std::vector<float>> input_values) {
    data_queue.push(audioToValueVector(*input_values, feature_extractor));
}

bool WakeClassifier::isReadyForRun() {
    return !data_queue.empty() && !atomic_wait.load();
}

void WakeClassifier::runModelAsync()
{
    std::shared_ptr<std::vector<Ort::Value>> ort_outputs = std::make_shared<std::vector<Ort::Value>>();
    ort_outputs->emplace_back(Ort::Value{nullptr});
    out_value_queue.push(ort_outputs);

    std::shared_ptr<AudioTensor> input_tensors = data_queue.front();
    atomic_wait.store(true);

    std::cout << "Running classifier async..." << std::endl;

    session->RunAsync(
        Ort::RunOptions{nullptr},
        input_names_arrays.data(),
        input_tensors->getValueTensor(),
        input_names_arrays.size(),
        output_names_arrays.data(),
        ort_outputs->data(),
        output_names_arrays.size(),
        mainRunCallback,
        this);
}

void WakeClassifier::mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr)
{
    std::cout << "classifier callback" << std::endl;
    WakeClassifier* curr_obj = reinterpret_cast<WakeClassifier*>(user_data);
    Ort::Status status(status_ptr);
    if (!status.IsOK())
    {
        std::cout << "ERROR running model inference: " << status.GetErrorMessage() << std::endl;
        exit(-1);
    }
    Ort::Value output_value(outputs[0]);
    int test = ortValueToTorchAndArgmax(output_value);
    std::cout << "Res: " << test << std::endl;
    if (test == 27) {
        std::cout << "Marvin" << std::endl;
        wakeup.store(true);
    }
    curr_obj->data_queue.pop();
    atomic_wait.store(false);
    output_value.release();
}