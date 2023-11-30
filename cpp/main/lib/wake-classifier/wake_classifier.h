#include <string>
#include <queue>
#include <onnxruntime_cxx_api.h>


namespace py = pybind11;


class WakeClassifier {
    public:
        WakeClassifier(std::string model, std::queue<std::vector<float>>& data_queue, std::shared_ptr<Ort::Session> new_session);
        std::vector<Ort::Value> runModel(std::vector<Ort::Value> input_tensors);
    private:
        std::string model_name;
        std::shared_ptr<Ort::Session> session; 
        std::queue<std::vector<float>> raw_audio_queue;
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

        template <typename Function>
        std::vector<const char *> WakeClassifier::getInputOrOutputNames(size_t size, Function name_allocator_func);

        std::vector<Ort::Value> audioToValueVector(std::vector<float>& float_vector, py::object& extractor);

};
