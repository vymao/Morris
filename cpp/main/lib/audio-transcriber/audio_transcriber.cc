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
#include <regex>

#include "main/lib/util/common.h"
#include "audio_transcriber.h"
#include "main/lib/util/feature_extractor.h"

using namespace std::placeholders;
using namespace pybind11::literals; 
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

std::atomic_bool AudioTranscriber::atomic_wait = false;

AudioTranscriber::AudioTranscriber(
    std::string model,
    std::shared_ptr<Ort::Session> new_session,
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
    py::object &extractor,
    const std::vector<std::variant<int32_t, float>>& model_args = {}) : AudioModelBase(model, new_session, global_allocator, extractor), add_args(model_args)
{
    for (int i = 0; i < input_names.size(); i++) {
        arg_types.push_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
    }
    arg_types.erase(arg_types.begin());

    json raw_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/models/whisper_tiny/whisper_vocab.json");
    json added_tokens_raw_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/models/whisper_tiny/whisper_added_tokens.json");
    for (auto const& [token, idx] : raw_json.items()) {
        if (token == "<|startofprev|>") {
            start_of_prev_id = idx;
        } else if (token == "<|startoftranscript|>") {
            start_of_transcript_id = idx;
        } else if (token == "<|endoftext|>") {
            end_of_text_id = idx;
        }


        id_to_token_map[idx] = token;
    }

    for (auto const& [token, idx] : added_tokens_raw_json.items()) {
        if (token == "<|startofprev|>") {
            start_of_prev_id = idx;
        } else if (token == "<|startoftranscript|>") {
            start_of_transcript_id = idx;
        }
        id_to_token_map[idx] = token;
    }
}

std::shared_ptr<std::vector<Ort::Value>> AudioTranscriber::audioToValueVector(std::vector<float> &float_vector, py::object &extractor)
{
    py::array_t<float> data = py::array_t<float>(py::cast(float_vector));

    py::object rdict = extractor.attr("__call__")(data, "sampling_rate"_a=16000, "return_tensors"_a="np");
    py::array_t<float> rdata = rdict.attr("get")("input_features");
    py::buffer_info rdata_buf = rdata.request();
    const auto rdata_size = rdata_buf.shape;
    // float *carray = rdata.mutable_data();

    std::shared_ptr<std::vector<Ort::Value>> input_tensors = std::make_shared<std::vector<Ort::Value>>();
    input_tensors->emplace_back(buffer_to_tensor<float>(rdata_buf));
    return input_tensors;
}

std::shared_ptr<std::vector<Ort::Value>> AudioTranscriber::prepareInputs(std::vector<float> &input_values) {
    return audioToValueVector(input_values, feature_extractor);
}

void AudioTranscriber::prepareInputsAndPush(std::vector<float> &input_values) {
    std::shared_ptr<std::vector<Ort::Value>> input_tensor = audioToValueVector(input_values, feature_extractor);
    if (add_args.size()) {
        for (int i = 0; i < add_args.size(); i++) {
            //std::cout << arg_types[i] << std::endl;
            if (arg_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || arg_types[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                float var = std::get<float>(add_args[i]);
                input_tensor->emplace_back(float_to_tensor(var));
            }
            else {
                int32_t var = std::get<int>(add_args[i]);
                input_tensor->emplace_back(int_to_tensor(var));
            }
        }
    }
    data_queue.push(input_tensor);
}

bool AudioTranscriber::isReadyForRun() {
    return !data_queue.empty() && !atomic_wait.load();
}

void AudioTranscriber::runModelAsync(std::vector<Ort::Value> &output_values)
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
        this);
    std::cout << "Done!" << std::endl;
}

void AudioTranscriber::runModelAsync(std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_values)
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
        this);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    for (int i = 0; i < 20; ++i) {
        std::this_thread::sleep_for(dur);
    }

    std::cout << ms_int.count() << "ms\n";
    // std::cout << "Done!" << std::endl;
}

void AudioTranscriber::postProcess(Ort::Value& value) {
    //py::capsule buffer_handle([](){});
    //py::array_t<float> res_array = py::array_t<float>(value.GetTensorTypeAndShapeInfo().GetShape(), value.GetTensorData<float>(), buffer_handle);	


    //py::list res_decoded = feature_extractor.attr("batch_decode")(res_array, "skip_special_tokens"_a=true).cast<py::list>();
    std::string res = processIDstoToken(value.GetTensorData<int>(), value.GetTensorTypeAndShapeInfo().GetElementCount(), true);
    std::cout << res << std::endl;
    //out_queue.push(res_decoded[0].cast<std::string>());
    out_queue.push(res);
}

void AudioTranscriber::mainRunCallback(void *obj, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr)
{
    AudioTranscriber* curr_obj = reinterpret_cast<AudioTranscriber*>(obj);
    Ort::Status status(status_ptr);
    if (!status.IsOK())
    {
        std::cout << "ERROR running model inference: " << status.GetErrorMessage() << std::endl;
        exit(-1);
    }
    Ort::Value output_value(outputs[0]);
    std::cout << "callback" << std::endl;
    curr_obj->postProcess(output_value);
    curr_obj->data_queue.pop();

    output_value.release();
    atomic_wait.store(false);
}

bool AudioTranscriber::isSpecialToken(int id) {
    return (id == start_of_prev_id || id == start_of_transcript_id);
}

std::string AudioTranscriber::processIDstoToken(std::vector<int>& ids, bool skip_special_tokens = true) {
    std::string res = "";
    for (int& id : ids) {
        if (skip_special_tokens && isSpecialToken(id)) {
            continue;
        }
        
        res = res + id_to_token_map[id];
    }
    return res;
}

std::string AudioTranscriber::processIDstoToken(const int* ids_pointer, size_t ids_size, bool skip_special_tokens = true) {
    std::string res = "";

    int* ptr = const_cast<int*>(ids_pointer);
    try {
        for (int i = 0; i < ids_size; i++) {
            int id = *ptr;
            //std::cout << "ID: " << id << std::endl;
            if (id == end_of_text_id) {
                break;
            }
            if (skip_special_tokens && isSpecialToken(id)) {
                continue;
            }
            
            res = res + id_to_token_map[id];
            ptr++;
        }
        
        //wchar_t t = L'Ġ';
        res = std::regex_replace(res, std::regex("Ġ"), " ");
    }
    catch (...) {
        std::cerr << "Error decoding output." << std::endl;
    }
    return res;
}