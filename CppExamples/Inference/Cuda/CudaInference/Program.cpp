#include <iostream>
#include <string>
#include "onnxruntime_cxx_api.h"
#include "cuda_provider_factory.h"

#include <vector>
#include <stdexcept>

int main() {
    try {
        const std::string modelFilePath = "<YOUR MODEL>";
        std::wstring wModelFilePath(modelFilePath.begin(), modelFilePath.end());

        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "CUDA_Debug");
        Ort::SessionOptions sessionOptions;
        
        // CUDA execution provider (device index 0 means the first core).
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
        if (status != nullptr) {
            const char* errorMessage = Ort::GetApi().GetErrorMessage(status);
            std::cerr << "Failed to append CUDA execution provider: " << errorMessage << std::endl;
            return -1;
        }
        std::cout << "CUDA execution provider appended successfully." << std::endl;

        Ort::Session session(env, wModelFilePath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session.GetInputCount();
        std::vector<std::string> input_names;
        for (size_t i = 0; i < num_inputs; i++) {
            auto name_alloc = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(std::string(name_alloc.get()));
        }

        std::cout << "Input Names:" << std::endl;
        for (const auto& name : input_names)
            std::cout << "  " << name << std::endl;

        // --- Prepare Dummy Input Data ---
        std::vector<int64_t> input_ids_shape = { 1, 3 };
        std::vector<int64_t> input_ids_data = { 1, 2, 3 };

        std::vector<int64_t> position_ids_shape = { 1, 3 };
        std::vector<int64_t> position_ids_data = { 0, 1, 2 };

        std::vector<int64_t> attention_mask_shape = { 1, 3 };
        std::vector<int64_t> attention_mask_data = { 1, 1, 1 };

        // past_key_values: assume empty past with sequence_length = 0.
        std::vector<int64_t> past_shape = { 1, 32, 0, 96 };
        std::vector<uint16_t> past_data; 

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_data.data(), input_ids_data.size(),
            input_ids_shape.data(), input_ids_shape.size());
        Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, position_ids_data.data(), position_ids_data.size(),
            position_ids_shape.data(), position_ids_shape.size());
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_data.data(), attention_mask_data.size(),
            attention_mask_shape.data(), attention_mask_shape.size());

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(position_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));

        for (size_t i = 3; i < num_inputs; i++) {
            Ort::Value past_tensor = Ort::Value::CreateTensor(
                memory_info,
                reinterpret_cast<void*>(past_data.data()),
                past_data.size() * sizeof(uint16_t),
                past_shape.data(), past_shape.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            input_tensors.push_back(std::move(past_tensor));
        }

        // We need C-style strings for input.
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names)
            input_names_cstr.push_back(name.c_str());

        // Retrieve output names.
        size_t num_outputs = session.GetOutputCount();
        std::vector<std::string> output_names;
        for (size_t i = 0; i < num_outputs; i++) {
            auto name_alloc = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(std::string(name_alloc.get()));
        }
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names)
            output_names_cstr.push_back(name.c_str());

        // Run Inference
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
            input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
            output_names_cstr.data(), output_names_cstr.size());

        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            float* logits_data = output_tensors[0].GetTensorMutableData<float>();
            std::cout << "Logits output, first few values:" << std::endl;
            for (size_t i = 0; i < 10; i++) {
                std::cout << "  " << logits_data[i] << std::endl;
            }
        }
        else {
            std::cout << "No logits output tensor found." << std::endl;
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
