// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include <onnxruntime_cxx_api.h>

#include <pybind11/embed.h> 
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cstdio>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <queue>

#include "lib/util/feature_extractor.h"
#include "lib/wake-classifier/wake_classifier.h"
#include "lib/audio-streamer/audio_stream.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

namespace py = pybind11;
namespace fs = std::filesystem;

constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
static const char* const kOrtSessionOptionsConfigUseEnvAllocators = "session.use_env_allocators";

void whisper_print_usage(int argc, char **argv, const whisper_params &params);

bool whisper_params_parse(int argc, char **argv, whisper_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help")
        {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-sr" || arg == "--audio-samplerate")
        {
            params.audio_sampling_rate = std::stoi(argv[++i]);
        }
        else if (arg == "--step")
        {
            params.step_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--length")
        {
            params.length_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--keep")
        {
            params.keep_ms = std::stoi(argv[++i]);
        }
        else if (arg == "-c" || arg == "--capture")
        {
            params.capture_id = std::stoi(argv[++i]);
        }
        else if (arg == "-mt" || arg == "--max-tokens")
        {
            params.max_tokens = std::stoi(argv[++i]);
        }
        else if (arg == "-ac" || arg == "--audio-ctx")
        {
            params.audio_ctx = std::stoi(argv[++i]);
        }
        else if (arg == "-vth" || arg == "--vad-thold")
        {
            params.vad_thold = std::stof(argv[++i]);
        }
        else if (arg == "-fth" || arg == "--freq-thold")
        {
            params.freq_thold = std::stof(argv[++i]);
        }
        else if (arg == "-su" || arg == "--speed-up")
        {
            params.speed_up = true;
        }
        else if (arg == "-tr" || arg == "--translate")
        {
            params.translate = true;
        }
        else if (arg == "-nf" || arg == "--no-fallback")
        {
            params.no_fallback = true;
        }
        else if (arg == "-ps" || arg == "--print-special")
        {
            params.print_special = true;
        }
        else if (arg == "-kc" || arg == "--keep-context")
        {
            params.no_context = false;
        }
        else if (arg == "-l" || arg == "--language")
        {
            params.language = argv[++i];
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-f" || arg == "--file")
        {
            params.fname_out = argv[++i];
        }
        else if (arg == "-tdrz" || arg == "--tinydiarize")
        {
            params.tinydiarize = true;
        }
        else if (arg == "-sa" || arg == "--save-audio")
        {
            params.save_audio = true;
        }
        else if (arg == "-ng" || arg == "--no-gpu")
        {
            params.use_gpu = false;
        }

        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char **argv, const whisper_params &params)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -sr N,    --audio-samplerate [%-7d] sample rate for audio\n", params.audio_sampling_rate);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n", params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n", params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n", params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n", params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n", params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n", params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n", params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n", params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n", params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n", params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n", params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n", params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n", params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "\n");
}

int main(int argc, char **argv)
{
    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false)
    {
        return 1;
    }

    const int thread_pool_size = std::thread::hardware_concurrency();
    std::cout << "Threads: " << thread_pool_size << std::endl;

    py::scoped_interpreter guard{};
    py::module_ transformers_module = py::module_::import("transformers");
    py::object AutoFeatureExtractor = py::getattr(transformers_module, "AutoFeatureExtractor");
    py::object extractor = AutoFeatureExtractor.attr("from_pretrained")("MIT/ast-finetuned-speech-commands-v2");


    OrtEnv* environment;
    OrtThreadingOptions* thread_opts = nullptr;
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    g_ort->CreateThreadingOptions(&thread_opts);
    g_ort->SetGlobalIntraOpNumThreads(thread_opts, 4);
    g_ort->CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_WARNING, "onnx_global_threadpool", thread_opts, &environment);

    OrtArenaCfg* arena_cfg = nullptr;
    OrtMemoryInfo* meminfo = nullptr;
    const char* keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes", "max_power_of_two_extend_bytes"};
    const size_t values[] = {0 /*let ort pick default max memory*/, 0, 0, 0, 0, 0};
    g_ort->CreateArenaCfgV2(keys, values, 5, &arena_cfg);

    std::vector<const char*> provider_keys, provider_values;
    g_ort->CreateAndRegisterAllocatorV2(environment, kCpuExecutionProvider, meminfo, arena_cfg, provider_keys.data(), provider_values.data(), 0);


    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");

    Ort::Env env = Ort::Env(environment);
    env.UpdateEnvWithCustomLogLevel(ORT_LOGGING_LEVEL_WARNING);

    //Ort::ThrowOnError(RegisterCustomOps(static_cast<OrtSessionOptions*>(session_options), OrtGetApiBase()));
    std::shared_ptr<Ort::Session> session_ptr = std::make_shared<Ort::Session>(env, params.model.c_str(), session_options);

    auto output_shapes = session_ptr->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shapes: " << print_shape(output_shapes) << std::endl;
 

    std::shared_ptr<Ort::AllocatorWithDefaultOptions> allocator_ptr = std::make_shared<Ort::AllocatorWithDefaultOptions>();

    WakeClassifier wake_classifier = WakeClassifier("MIT/ast-finetuned-speech-commands-v2", session_ptr, allocator_ptr, extractor);

    std::vector<std::int64_t> input_shapes;
    for (std::size_t i = 0; i < session_ptr->GetInputCount(); i++)
    {
        input_shapes = session_ptr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    // print name/shape of outputs
    std::vector<std::string> output_names;
    for (std::size_t i = 0; i < session_ptr->GetOutputCount(); i++)
    {
        auto output_shapes = session_ptr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    }

    for (int i = 0; i < input_shapes.size(); i++)
    {
        input_shapes[i] = abs(input_shapes[i]);
    }

    //////////

    std::queue<std::shared_ptr<std::vector<float>>> data_queue;
    std::queue<std::shared_ptr<std::vector<Ort::Value>>> classifier_data_queue;
    std::queue<std::shared_ptr<std::vector<Ort::Value>>> classifier_out;
    
    std::thread audio_stream_t(createAndRunAudioStream, std::ref(params), std::ref(data_queue));


    //std::cout << "Generating " << input_shapes.back() << " numbers..." << std::endl;
    // generate random numbers in the range [0, 255]
    //int total_number_elements = 93680;
    //std::vector<float> input_tensor_values(total_number_elements);
    //std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&]
    //              { return rand() % 255; });
    //std::cout << "Generated." << std::endl;
    //Matrix data {input_tensor_values};
    //py::buffer_info data_buffer = getBufferInfoForMatrix(data);
    //std::cout << "Creating pybind11 array..." << std::endl;
    //py::array_t<float> data_buffer = getFloatArrayforMatrix(data);
    //std::cout << "Done" << std::endl;

    //float *carray = rdata.mutable_data();

    while (true) {
        if (!data_queue.empty()) {
            std::cout << "Preparing inputs..." << std::endl;
            std::shared_ptr<std::vector<float>> data = data_queue.front();
            classifier_data_queue.push(wake_classifier.prepareInputs(*data));
            data_queue.pop();
            std::cout << "Done" << std::endl;
        }

        if (!classifier_data_queue.empty() && !wake_classifier.atomic_wait.load()) {
            std::shared_ptr<std::vector<Ort::Value>> ort_outputs = std::make_shared<std::vector<Ort::Value>>();
            ort_outputs->emplace_back(Ort::Value{nullptr});
            //wake_classifier.runModelSync(input_tensors);
            wake_classifier.runModelAsync(*ort_outputs, classifier_data_queue);
            classifier_out.push(ort_outputs);

        } else {
            std::chrono::duration<double, std::milli> dur{50};
            std::this_thread::sleep_for(dur);
        }
    }
    std::cout << "End." << std::endl;
    // run(params);
    audio_stream_t.join();
    return 0;
}