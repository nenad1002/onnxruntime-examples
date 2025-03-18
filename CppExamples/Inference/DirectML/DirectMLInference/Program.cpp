#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"

#include <vector>
#include <string>
#include <string_view>
#include <stdexcept>

int main() {
    try {
        // Specify the model file path (make sure DirectML can execute it).
        constexpr std::string_view modelFileConstant = "<YOUR_ONNX_MODEL>";

        std::wstring wideString(modelFileConstant.begin(), modelFileConstant.end());
        std::basic_string<ORTCHAR_T> modelFile(wideString);

        // Retrieve the DirectML API pointer.
        OrtApi const& ortApi = Ort::GetApi();
        const OrtDmlApi* ortDmlApi = nullptr;
        OrtStatusPtr status = ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
        if (status != nullptr || ortDmlApi == nullptr) {
            const char* errorMessage = ortApi.GetErrorMessage(status);
            std::cerr << "Error retrieving DML API pointer: " << errorMessage << std::endl;
            return -1;
        }
        std::cout << "DML API pointer successfully retrieved." << std::endl;

        // Create ONNX Runtime environment and session options.
        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "DML_Debug");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // required for DML because it cannot schedule on GPU in parallel.
        sessionOptions.DisableMemPattern(); // Also DirectML specific

        // Append the first DirectML provider you find (index 0 represents the 1st GPU, in case there are multiple GPUs, you could increase the index)
        OrtStatus* status2 = ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);
        if (status2 != nullptr) {
            std::wcerr << L"Failed to append DirectML execution provider." << std::endl;
            return -1;
        }

        Ort::Session session(env, modelFile.c_str(), sessionOptions);
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
        // 1. input_ids: assume shape [1, 3], dummy values {1, 2, 3}
        std::vector<int64_t> input_ids_shape = { 1, 3 };
        std::vector<int64_t> input_ids_data = { 1, 2, 3 };

        // 2. position_ids: shape [1, 3], dummy values {0, 1, 2}
        std::vector<int64_t> position_ids_shape = { 1, 3 };
        std::vector<int64_t> position_ids_data = { 0, 1, 2 };

        // 3. attention_mask: shape [1, 3], dummy values {1, 1, 1}
        std::vector<int64_t> attention_mask_shape = { 1, 3 };
        std::vector<int64_t> attention_mask_data = { 1, 1, 1 };

        // 4. past_key_values:
        //    For each past key/value, assume type float16 which is most commonly used with ONNX.
        //    If your ONNX model uses different type, please change this.
        //    The model I used follows the following structure, hence this is hardcoded: [1, 32, sequence_length, 96].
        //    We also assume an empty past with sequence_length = 0 since otherwise this sample becomes more complicated.
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
