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
#include "model_base.h"
#include "main/lib/util/feature_extractor.h"

using namespace std::placeholders;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

AudioModelBase::AudioModelBase(
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

AudioModelBase::~AudioModelBase() {};

template <typename Function>
std::vector<std::string> AudioModelBase::getInputOrOutputNames(size_t size, Function name_allocator_func)
{
    std::vector<std::string> names_list;
    for (std::size_t i = 0; i < size; i++)
    {
        names_list.emplace_back(name_allocator_func(i).get());
    }
    return names_list;
}

std::vector<const char *> AudioModelBase::getInputOrOutputNameArray(std::vector<std::string> &name_vector)
{

    std::vector<const char *> names_char(name_vector.size(), nullptr);
    std::transform(std::begin(name_vector), std::end(name_vector), std::begin(names_char),
                   [&](const std::string &str)
                   { return str.c_str(); });
    return names_char;
}

std::shared_ptr<Ort::Session> AudioModelBase::getSession() {
    return session;
}

std::vector<Ort::Value> AudioModelBase::runModelSync(std::vector<Ort::Value> &input_tensors)
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