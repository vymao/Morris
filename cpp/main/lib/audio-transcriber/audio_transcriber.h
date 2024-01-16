#ifndef AUDIO_TRANSCRIBER
#define AUDIO_TRANSCRIBER

#include <string>
#include <queue>
#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <variant>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <deque>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

#include "main/lib/model/model_base.h"
#include "main/lib/model/audio_tensor.h"

namespace py = pybind11;
using json = nlohmann::json;
using namespace audio;

struct TranscriberInferenceParams {
    int32_t max_length = 20;
	int32_t min_length = 1;
	int32_t num_beams = 1;
	int32_t num_return_sequences = 1;
	float length_penalty = 1.0f;
	float repetition_penalty = 1.0f;
};

class AudioTranscriber : public AudioModelBase
{
public:
    AudioTranscriber(
        std::string model,
        std::shared_ptr<Ort::Session> new_session,
        std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
        py::object &extractor,
        const int32_t layered_multiplier,
        const struct TranscriberInferenceParams& model_args,
        const bool use_layered_transcription,
        const bool use_realtime_transription);
    void runModelAsync();
    static void mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);
    static void layeredRunCallback(void *obj, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);
    std::shared_ptr<AudioTensor> prepareInputs(std::vector<float> &input_values);
    void prepareInputsAndPush(std::shared_ptr<std::vector<float>> input_values);
    bool isReadyForRun();
    std::string getLayeredOutput();
    std::string getTotalOutput();

    static std::atomic_bool realtime_atomic_wait;
    std::deque<std::string> realtime_out_queue;
    std::queue<std::string> layered_out_queue;
    std::vector<std::deque<std::string>> out_queues;
    static std::string layered_string_out;
private:
    struct TranscriberInferenceParams add_args;
    std::unordered_map<int, std::string> id_to_token_map;
    int start_of_prev_id;
    int start_of_transcript_id;
    int end_of_text_id;
    bool use_layered;
    bool use_realtime;
    int32_t multiplier;
    std::queue<std::shared_ptr<std::vector<float>>> layered_data_queue;

    std::string postProcess(Ort::Value& value);
    bool isSpecialToken(int id);
    std::shared_ptr<AudioTensor> audioToValueVector(std::vector<float> &float_vector, py::object &extractor);
    std::string processIDstoToken(int* start_ptr, const int* end_ptr, bool skip_special_tokens = true);

};

#endif