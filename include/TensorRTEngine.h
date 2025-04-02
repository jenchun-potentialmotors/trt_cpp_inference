#ifndef TENSORRTENGINE_H
#define TENSORRTENGINE_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ILogger;
enum class DataType;
enum class TensorIOMode;
} // namespace nvinfer1

class CudaMemory {
public:
  static std::unique_ptr<void, std::function<void(void *)>>
  allocateDevice(size_t size);

  static std::unique_ptr<void, std::function<void(void *)>>
  allocateHost(size_t size);
};

class TensorrtEngine {
public:
  TensorrtEngine(const std::string &enginePath);
  ~TensorrtEngine();

  bool initialize();
  bool executeInference(const std::vector<float> &inputData,
                        std::vector<float> &outputData);

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

  bool loadEngine();
  size_t getDataTypeSize(nvinfer1::DataType dtype);
  bool allocateBuffers();
  void setTensorAddresses();
  void cleanup();
};

#endif // TENSORRTENGINE_H