#include <pybind11/embed.h> 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <io>

#include "wake_classifier.h"
#include "main/lib/util/feature_extractor.h"

using namespace std::placeholders;

WakeClassifier::WakeClassifier (
    std::string model,
    std::queue<std::vector<float>>& data_queue, 
    std::shared_ptr<Ort::Session> new_session
) {
    model_name = model;
    raw_audio_queue = data_queue;

    session = new_session;

    auto input_func = std::bind(&Ort::Session::GetInputNameAllocated, &(*new_session), _1, allocator);
    auto output_func = std::bind(&Ort::Session::GetOutputNameAllocated, &(*new_session), _1, allocator);
    input_names = getInputOrOutputNames(session->GetInputCount(), input_func);
    output_names = getInputOrOutputNames(session->GetOutputCount(), output_func);

}

template <typename Function>
std::vector<const char *> WakeClassifier::getInputOrOutputNames(size_t size, Function name_allocator_func) {
    std::vector<std::string> names_list;
    for (std::size_t i = 0; i < size; i++) {
        names_list.emplace_back(name_allocator_func(i, allocator).get());
    }

    std::vector<const char *> names_char(names_list.size(), nullptr);
    std::transform(std::begin(names_list), std::end(names_list), std::begin(names_char),
                   [&](const std::string &str)
                   { return str.c_str(); });
    
    return names_char;
}

std::vector<Ort::Value> WakeClassifier::audioToValueVector(std::vector<float>& float_vector, py::object& extractor) {
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

Ort::Value WakeClassifier::runModel() {
    try
    {
       // auto t1 = high_resolution_clock::now();
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                          input_names.size(), output_names.data(), output_names.size());
       // auto t2 = high_resolution_clock::now();
        //auto ms_int = duration_cast<milliseconds>(t2 - t1);
        //std::cout << ms_int.count() << "ms\n";
        //std::cout << "Done!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
    }
    catch (const Ort::Exception &exception)
    {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }
}