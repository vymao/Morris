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

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

#include "main/lib/model/model_base.h"

namespace py = pybind11;
using json = nlohmann::json;

class AudioTranscriber : public AudioModelBase
{
public:
    AudioTranscriber(
        std::string model,
        std::shared_ptr<Ort::Session> new_session,
        std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
        py::object &extractor,
        const std::vector<std::variant<int32_t, float>>& model_args);
    void runModelAsync(std::vector<Ort::Value> &output_values);
    void runModelAsync(std::vector<Ort::Value> &input_tensors, 
        std::vector<Ort::Value> &output_values);
    static void mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);
    std::shared_ptr<std::vector<Ort::Value>> prepareInputs(std::vector<float> &input_values);
    void prepareInputsAndPush(std::vector<float> &input_values);
    bool isReadyForRun();

    static std::atomic_bool atomic_wait;
    std::queue<std::string> out_queue;
private:
    std::vector<std::variant<int, float>> add_args;
    std::vector<ONNXTensorElementDataType> arg_types;
    std::unordered_map<int, std::string> id_to_token_map;
    int start_of_prev_id;
    int start_of_transcript_id;
    int end_of_text_id;


    void postProcess(Ort::Value& value);
    bool isSpecialToken(int id);
    std::string processIDstoToken(std::vector<int>& ids, bool skip_special_tokens);
    std::shared_ptr<std::vector<Ort::Value>> audioToValueVector(std::vector<float> &float_vector, py::object &extractor);
    std::string processIDstoToken(const int* ids_pointer, size_t ids_size, bool skip_special_tokens);

};
