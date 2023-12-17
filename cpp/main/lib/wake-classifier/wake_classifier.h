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

namespace py = pybind11;

class WakeClassifier
{
public:
    WakeClassifier(std::string model,
                   std::shared_ptr<Ort::Session> new_session,
                   std::shared_ptr<Ort::AllocatorWithDefaultOptions> global_allocator,
                   py::object &extractor);
    std::vector<Ort::Value> runModelSync(std::vector<Ort::Value> &input_tensors);
    void runModelAsync(std::vector<Ort::Value> &output_values,
                       std::queue<std::shared_ptr<std::vector<Ort::Value>>> &data_queue);
    void runModelAsync(std::vector<Ort::Value> &input_tensors, 
        std::vector<Ort::Value> &output_values);
    int getNumOutputNames();
    static void mainRunCallback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr);
    static std::shared_ptr<std::vector<Ort::Value>> audioToValueVector(std::vector<float> &float_vector, py::object &extractor);
    std::shared_ptr<std::vector<Ort::Value>>  prepareInputs(std::vector<float> &input_values);

    std::thread::id caller_tid;
    static std::atomic_bool atomic_wait;

private:
    std::string model_name;
    std::shared_ptr<Ort::Session> session;
    std::shared_ptr<Ort::AllocatorWithDefaultOptions> allocator;
    py::object feature_extractor;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char *> input_names_arrays;
    std::vector<const char *> output_names_arrays;

    template <typename Function>
    std::vector<std::string> getInputOrOutputNames(size_t size, Function name_allocator_func);
    std::vector<const char *> getInputOrOutputNameArray(std::vector<std::string> &name_vector);
};
