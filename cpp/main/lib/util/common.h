#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>

#include "onnxruntime_cxx_api.h"

using json = nlohmann::json;
namespace py = pybind11;

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t layer_multiplier = 4;
    int32_t capture_id = 1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t audio_sampling_rate = 16000;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool use_llm       = false;

    std::string language  = "en";
    std::string classifier_model     = "";
    std::string transcriber_model    = "";
    std::string fname_out;
};

std::string print_shape(const std::vector<std::int64_t> &v);

int ortValueToTorchAndArgmax(Ort::Value& value_tensor);

Ort::Value pyArrayToTorchAndConcat(py::array_t<float>& left_mat, std::tuple<int> left_size, py::array_t<float>& right_mat, std::tuple<int> right_size);

json parseJSON(std::string file);
