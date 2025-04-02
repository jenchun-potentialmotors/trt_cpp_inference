/* Copyright 2025 Potential Motors.
  @authors {Jen-Chun Wang} */
#include "Inference.h"
#include "InferenceEngine.h"
#include "Tensor.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace oros {

class TensorrtEngine final : public InferenceEngine {
public:
  TensorrtEngine(const Inference::Options &options);
  ~TensorrtEngine() override;

  Tensor infer(const Tensor &input_tensor) const override;

  TensorrtEngine() = delete;
  TensorrtEngine(const TensorrtEngine &) = delete;
  TensorrtEngine(TensorrtEngine &&) = delete;
  TensorrtEngine &operator=(const TensorrtEngine &) = delete;
  TensorrtEngine &operator=(TensorrtEngine &&) = delete;

private:
  std::string enginePath_;
  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime *)>>
      runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine,
                  std::function<void(nvinfer1::ICudaEngine *)>>
      engine_;
  std::unique_ptr<nvinfer1::IExecutionContext,
                  std::function<void(nvinfer1::IExecutionContext *)>>
      context_;
  cudaStream_t stream_;
  std::vector<std::unique_ptr<void, std::function<void(void *)>>> inputBuffers_;
  std::vector<std::unique_ptr<void, std::function<void(void *)>>>
      outputBuffers_;
  std::vector<size_t> inputBufferSizes_;
  std::vector<size_t> outputBufferSizes_;
  std::vector<void *> bindings_;

  bool initialize();
  bool loadEngine();
  size_t getDataTypeSize(nvinfer1::DataType dtype);
  bool allocateBuffers();
  void setTensorAddresses();
  void cleanup();
  bool executeInference(const std::vector<float> &inputData,
                        std::vector<float> &outputData) const;
};

} // namespace oros
