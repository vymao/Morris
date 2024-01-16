#include <string>
#include <queue>
#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include <onnxruntime_cxx_api.h>

#include "main/lib/model/model_base.h"
#include "main/lib/model/audio_tensor.h"

namespace py = pybind11;
using namespace audio;

class WakeClassifier : public AudioModelBase
{
public:
    WakeClassifier(std::string model,
                   std::shared_ptr<Ort::Session> new_session,
                   std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
                   py::object &extractor);
    void runModelAsync();
    int getNumOutputNames();
    static void mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);
    std::shared_ptr<AudioTensor> prepareInputs(std::vector<float> &input_values);
    void prepareInputsAndPush(std::shared_ptr<std::vector<float>> input_values);
    bool isReadyForRun();
    

    static std::atomic_bool atomic_wait;
    static std::atomic_bool wakeup;

private:
    std::shared_ptr<AudioTensor> audioToValueVector(std::vector<float> &float_vector, py::object &extractor);

};
