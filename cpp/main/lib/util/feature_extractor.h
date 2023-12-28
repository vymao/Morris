#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <onnxruntime_cxx_api.h>

namespace py = pybind11;

class Matrix {
public:
    Matrix(std::vector<float>& raw_data) {
        m_data = &raw_data;
    }
    float *data() { return m_data->data(); }
    size_t size() { return m_data->size(); }
private:
    std::vector<float> *m_data;
};

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape);

template <typename T>
Ort::Value buffer_to_tensor(py::buffer_info &data);

py::buffer_info getBufferInfoForMatrix(Matrix& m);

py::array_t<float> getFloatArrayforMatrix(Matrix& m);

std::vector<Ort::Value> rawAudioToValueVector(std::vector<float>& float_vector, py::object& extractor);

Ort::Value int_to_tensor(int32_t value);

Ort::Value float_to_tensor(float value);