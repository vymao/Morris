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

std::atomic_bool AudioTranscriber::realtime_atomic_wait = false;
std::string AudioTranscriber::layered_string_out = "";

AudioTranscriber::AudioTranscriber(
    std::string model,
    std::shared_ptr<Ort::Session> new_session,
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
    py::object &extractor,
    const int32_t layered_multiplier,
    const struct TranscriberInferenceParams &model_args,
    const bool use_layered_transcription = false,
    const bool use_realtime_transcription = true) : AudioModelBase(model, new_session, global_allocator, extractor),
                                                                        add_args(model_args),
                                                                        multiplier(layered_multiplier),
                                                                        use_realtime(use_realtime_transcription),
                                                                        use_layered(use_layered_transcription)
{
    has_secondary_data_queue = true;

    json raw_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/models/whisper_tiny/whisper_vocab.json");
    json added_tokens_raw_json = parseJSON("/Users/victor/Desktop/Morris/cpp/main/models/whisper_tiny/whisper_added_tokens.json");
    for (auto const &[token, idx] : raw_json.items())
    {
        if (token == "<|startofprev|>")
        {
            start_of_prev_id = idx;
        }
        else if (token == "<|startoftranscript|>")
        {
            start_of_transcript_id = idx;
        }
        else if (token == "<|endoftext|>")
        {
            end_of_text_id = idx;
        }

        id_to_token_map[idx] = token;
    }

    for (auto const &[token, idx] : added_tokens_raw_json.items())
    {
        if (token == "<|startofprev|>")
        {
            start_of_prev_id = idx;
        }
        else if (token == "<|startoftranscript|>")
        {
            start_of_transcript_id = idx;
        }
        else if (token == "<|endoftext|>")
        {
            end_of_text_id = idx;
        }
        id_to_token_map[idx] = token;
    }

    out_queues.emplace_back(std::deque<std::string>());
    out_queues.emplace_back(std::deque<std::string>());
}

std::shared_ptr<AudioTensor> AudioTranscriber::audioToValueVector(std::vector<float> &float_vector, py::object &extractor)
{
    py::array_t<float> data = py::array_t<float>(py::cast(float_vector));

    py::object rdict = extractor.attr("__call__")(data, "sampling_rate"_a = 16000, "return_tensors"_a = "np");
    py::array_t<float> rdata = rdict.attr("get")("input_features");
    py::buffer_info rdata_buf = rdata.request();

    std::shared_ptr<AudioTensor> audio_tensor = std::make_shared<AudioTensor>();
    audio_tensor->setValueTensor(std::move(rdata_buf));
    return audio_tensor;
}

std::shared_ptr<AudioTensor> AudioTranscriber::prepareInputs(std::vector<float> &input_values)
{
    std::shared_ptr<AudioTensor> input_tensor = audioToValueVector(input_values, feature_extractor);
    if ((out_queues[0].size() == multiplier) && (layered_data_queue.size() >= multiplier) && use_layered)
    {
        std::cout << "layered" << std::endl;
        std::vector<float> combined;
        for (int i = 0; i < multiplier; i++)
        {
            std::vector<float> curr = *layered_data_queue.front();
            combined.insert(combined.end(), curr.begin(), curr.end());
            layered_data_queue.pop();
        }
        std::shared_ptr<AudioTensor> layered_tensor = audioToValueVector(combined, feature_extractor);
        *input_tensor = batchConcatAudioTensors(*input_tensor, *layered_tensor);
    } 
    

    std::vector<Ort::Value> add_args_tensors;
    for (int i = 0; i < input_names.size(); i++)
    {
        if (input_names[i] == "max_length") {
            add_args_tensors.emplace_back(int_to_tensor(&add_args.max_length));
        }
        else if (input_names[i] == "min_length") {
            add_args_tensors.emplace_back(int_to_tensor(&add_args.min_length));
        }
        else if (input_names[i] == "num_beams") {
            add_args_tensors.emplace_back(int_to_tensor(&add_args.num_beams));
        }
        else if (input_names[i] == "num_return_sequences") {
            add_args_tensors.emplace_back(int_to_tensor(&add_args.num_return_sequences));
        }
        else if (input_names[i] == "length_penalty") {
            add_args_tensors.emplace_back(float_to_tensor(&add_args.length_penalty));
        } else if (input_names[i] == "repetition_penalty") {
            add_args_tensors.emplace_back(float_to_tensor(&add_args.repetition_penalty));
        }
    }
    input_tensor->prepareValueTensorInference(std::move(add_args_tensors));
    return input_tensor;
}

void AudioTranscriber::prepareInputsAndPush(std::shared_ptr<std::vector<float>> input_values)
{
    data_queue.push(prepareInputs(*input_values));
    layered_data_queue.push(input_values);
}

bool AudioTranscriber::isReadyForRun()
{
    return !data_queue.empty() && !realtime_atomic_wait.load();
}

std::string AudioTranscriber::getLayeredOutput()
{
    return layered_string_out;
}

std::string AudioTranscriber::getTotalOutput()
{
    std::string res;
    for (std::string &p : out_queues[0])
    {
        res = res + p;
    }
    return layered_string_out + res;
}

void AudioTranscriber::runModelAsync()
{
    if (use_realtime) {
        std::shared_ptr<std::vector<Ort::Value>> ort_outputs = std::make_shared<std::vector<Ort::Value>>();
        ort_outputs->emplace_back(Ort::Value{nullptr});
        out_value_queue.push(ort_outputs);

        std::shared_ptr<AudioTensor> input_tensors = data_queue.front();
        realtime_atomic_wait.store(true);

        // if (data_queue.size() > 5) {
        //     std::cout << "\033[1;31m" + "Fallen behind in transcription, skipping inference." + "\033[0m\n";
        //     return;
        // }
        std::cout << "Queue size: " << data_queue.size() << std::endl;

        std::cout << "Running transcriber async..." << std::endl;
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
}

void AudioTranscriber::mainRunCallback(void *obj, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr)
{
    AudioTranscriber *curr_obj = reinterpret_cast<AudioTranscriber *>(obj);
    Ort::Status status(status_ptr);
    if (!status.IsOK())
    {
        std::cout << "ERROR running model inference: " << status.GetErrorMessage() << std::endl;
        exit(-1);
    }

    std::cout << "callback" << std::endl;
    Ort::Value output_value(outputs[0]);
    std::shared_ptr<AudioTensor> curr_data = curr_obj->data_queue.front();
    int batch_size = (int) *(curr_data->getShape());
    int max_length = static_cast<int>(output_value.GetTensorTypeAndShapeInfo().GetElementCount()) / batch_size;
    int* ptr = const_cast<int*>(output_value.GetTensorData<int>());

    std::string res;
    for (int i = 0; i < batch_size; i++) {
        int* start_ptr = ptr + (max_length * i);
        curr_obj->out_queues[i].push_back(curr_obj->processIDstoToken(start_ptr, start_ptr + max_length));
        if (i > 0) {
            for (int j = 0; j < curr_obj->multiplier; j++) {
                curr_obj->out_queues[i - 1].pop_front();
            }
        }
        res = curr_obj->out_queues[i].back();
    }
    
    if (curr_data->isLayered()) {
        res = "\033[1;33m" + res + "\033[0m\n";
    } else {
        res = "\033[1;32m" + res + "\033[0m\n";
    }
    std::cout << res;
    curr_obj->out_value_queue.pop();
    curr_obj->data_queue.pop();

    output_value.release();
    realtime_atomic_wait.store(false);
}

bool AudioTranscriber::isSpecialToken(int id)
{
    return (id == start_of_prev_id || id == start_of_transcript_id);
}

std::string AudioTranscriber::processIDstoToken(int* start_ptr, const int* end_ptr, bool skip_special_tokens)
{
    std::string res = "";
    try
    {
        for (int *ptr = const_cast<int*>(start_ptr); ptr < end_ptr; ptr++) {
            int id = *ptr;
            //std::cout << "ID: " << id << std::endl;
            if (id == end_of_text_id)
            {
                break;
            }
            ptr++;
            if (skip_special_tokens && isSpecialToken(id))
            {
                continue;
            }
            res = res + id_to_token_map[id];
        }            


        // wchar_t t = L'Ġ';
        res = std::regex_replace(res, std::regex("Ġ"), " ");
    }
    catch (...)
    {
        std::cerr << "Error decoding output." << std::endl;
    }
    return res;
}