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

WakeClassifier::WakeClassifier(
    std::string model,
    std::shared_ptr<Ort::Session> new_session,
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
    py::object &extractor) : model_name(model), session(new_session), allocator(global_allocator), feature_extractor(extractor)
{
    auto input_func = std::bind(&Ort::Session::GetInputNameAllocated, &(*session), _1, *allocator);
    auto output_func = std::bind(&Ort::Session::GetOutputNameAllocated, &(*session), _1, *allocator);
    input_names = getInputOrOutputNames(session->GetInputCount(), input_func);
    input_names_arrays = getInputOrOutputNameArray(input_names);
    output_names = getInputOrOutputNames(session->GetOutputCount(), output_func);
    output_names_arrays = getInputOrOutputNameArray(output_names);

    caller_tid = std::this_thread::get_id();
}

int WakeClassifier::getNumOutputNames() {
    return (int) output_names_arrays.size();
}

template <typename Function>
std::vector<std::string> WakeClassifier::getInputOrOutputNames(size_t size, Function name_allocator_func)
{
    std::vector<std::string> names_list;
    for (std::size_t i = 0; i < size; i++)
    {
        names_list.emplace_back(name_allocator_func(i).get());
    }
    return names_list;
}

std::vector<const char *> WakeClassifier::getInputOrOutputNameArray(std::vector<std::string> &name_vector)
{

    std::vector<const char *> names_char(name_vector.size(), nullptr);
    std::transform(std::begin(name_vector), std::end(name_vector), std::begin(names_char),
                   [&](const std::string &str)
                   { return str.c_str(); });
    return names_char;
}

std::shared_ptr<std::vector<Ort::Value>> WakeClassifier::audioToValueVector(std::vector<float> &float_vector, py::object &extractor)
{
    py::array_t<float> data = py::array_t<float>(py::cast(float_vector));

    py::object rdict = extractor.attr("__call__")(data, 16000, "np");
    py::array_t<float> rdata = rdict.attr("get")("input_values");
    py::buffer_info rdata_buf = rdata.request();
    const auto rdata_size = rdata_buf.shape;
    // float *carray = rdata.mutable_data();

    std::shared_ptr<std::vector<Ort::Value>> input_tensors = std::make_shared<std::vector<Ort::Value>>();
    input_tensors->emplace_back(buffer_to_tensor<float>(rdata_buf));
    return input_tensors;
}

std::shared_ptr<std::vector<Ort::Value>> WakeClassifier::prepareInputs(std::vector<float> &input_values) {
    return audioToValueVector(input_values, feature_extractor);
}

std::vector<Ort::Value> WakeClassifier::runModelSync(std::vector<Ort::Value> &input_tensors)
{
    std::vector<Ort::Value> output_tensors;
    try
    {
        auto t1 = high_resolution_clock::now();
        std::string str(input_names[0]);
        std::cout << "Name: " << str << std::endl;
        output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_arrays.data(), input_tensors.data(),
                                      input_names_arrays.size(), output_names_arrays.data(), output_names_arrays.size());
        auto t2 = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(t2 - t1);
        std::cout << ms_int.count() << "ms\n";
        // std::cout << "Done!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
    }
    catch (const Ort::Exception &exception)
    {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }

    return output_tensors;
}

void WakeClassifier::runModelAsync(std::vector<Ort::Value> &output_values, std::queue<std::shared_ptr<std::vector<Ort::Value>>> &data_queue)
{
    std::shared_ptr<std::vector<Ort::Value>> input_tensors = data_queue.front();
    atomic_wait.store(true);

    std::cout << "Running async..." << std::endl;

    session->RunAsync(
        Ort::RunOptions{nullptr},
        input_names_arrays.data(),
        input_tensors->data(),
        input_names_arrays.size(),
        output_names_arrays.data(),
        output_values.data(),
        output_names_arrays.size(),
        mainRunCallback,
        &data_queue);
    std::cout << "Done!" << std::endl;
}

void WakeClassifier::runModelAsync(std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_values)
{
    std::cout << "Running model..." << std::endl;
    std::chrono::duration<double, std::milli> dur{100};
    atomic_wait.store(true);

    auto t1 = high_resolution_clock::now();

    std::cout << "Running async..." << std::endl;


    session->RunAsync(
        Ort::RunOptions{nullptr},
        input_names_arrays.data(),
        input_tensors.data(),
        input_names_arrays.size(),
        output_names_arrays.data(),
        output_values.data(),
        output_names_arrays.size(),
        mainRunCallback,
        &caller_tid);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(dur);
    }

    std::cout << ms_int.count() << "ms\n";
    // std::cout << "Done!" << std::endl;
}

void WakeClassifier::mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr)
{
    std::queue<std::shared_ptr<std::vector<Ort::Value>>>* d_queue = reinterpret_cast<std::queue<std::shared_ptr<std::vector<Ort::Value>>>*>(user_data);
    Ort::Status status(status_ptr);
    if (!status.IsOK())
    {
        std::cout << "ERROR running model inference: " << status.GetErrorMessage() << std::endl;
        exit(-1);
    }
    Ort::Value output_value(outputs[0]);
    d_queue->pop();
    int test = ortValueToTorchAndArgmax(output_value);
    std::cout << "Res: " << test << std::endl;
    if (test == 27) {
        std::cout << "Marvin" << std::endl;
    }
    output_value.release();
    atomic_wait.store(false);
}