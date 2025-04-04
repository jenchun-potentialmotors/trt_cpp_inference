#include "../include/TensorRTEngine.h"
#include <cstring>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
} logger;

std::unique_ptr<void, std::function<void(void *)>>
CudaMemory::allocateDevice(size_t size) {
  void *deviceMemory = nullptr;
  if (cudaMalloc(&deviceMemory, size) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }
  return std::unique_ptr<void, std::function<void(void *)>>(deviceMemory,
                                                            [](void *ptr) {
                                                              if (ptr)
                                                                cudaFree(ptr);
                                                            });
}

std::unique_ptr<void, std::function<void(void *)>>
CudaMemory::allocateHost(size_t size) {
  void *hostMemory = nullptr;
  if (cudaMallocHost(&hostMemory, size) != cudaSuccess) {
    throw std::runtime_error("cudaMallocHost failed");
  }
  return std::unique_ptr<void, std::function<void(void *)>>(
      hostMemory, [](void *ptr) {
        if (ptr)
          cudaFreeHost(ptr);
      });
}

TensorrtEngine::TensorrtEngine(const std::string &enginePath)
    : enginePath_(enginePath), runtime_(createInferRuntime(logger),
                                        [](IRuntime *runtime) {
                                          if (runtime)
                                            delete runtime;
                                        }),
      engine_(nullptr,
              [](ICudaEngine *engine) {
                if (engine)
                  delete engine;
              }),
      context_(nullptr,
               [](IExecutionContext *context) {
                 if (context)
                   delete context;
               }),
      stream_(nullptr) {}

TensorrtEngine::~TensorrtEngine() { cleanup(); }

bool TensorrtEngine::initialize() {
  std::ifstream engineFile(enginePath_, std::ios::binary | std::ios::ate);
  bool engineExists = engineFile.good() && engineFile.tellg() > 0;
  engineFile.close();

  if (engineExists) {
    if (!loadEngine()) {
      return false;
    }
  } else {
    std::cerr << "Engine file not found, building engine is not implemented."
              << std::endl;
    return false;
  }

  context_.reset(engine_.get()->createExecutionContext());
  if (!context_) {
    std::cerr << "Failed to create execution context." << std::endl;
    cleanup();
    return false;
  }

  if (!allocateBuffers()) {
    std::cerr << "Failed to allocate buffers." << std::endl;
    cleanup();
    return false;
  }

  setTensorAddresses();
  cudaStreamCreate(&stream_);

  return true;
}

bool TensorrtEngine::executeInference(const std::vector<float> &inputData,
                                         std::vector<float> &outputData) {
  if (!engine_ || !context_) {
    std::cerr << "Engine or context not initialized." << std::endl;
    return false;
  }

  std::memcpy(inputBuffers_[0].get(), inputData.data(), inputBufferSizes_[0]);

  for (size_t i = 0; i < inputBuffers_.size() / 2; ++i) {
    cudaMemcpyAsync(inputBuffers_[2 * i + 1].get(), inputBuffers_[2 * i].get(),
                    inputBufferSizes_[i], cudaMemcpyHostToDevice, stream_);
  }

  if (!context_.get()->enqueueV3(stream_)) {
    std::cerr << "Error during inference." << std::endl;
    return false;
  }

  for (size_t i = 0; i < outputBuffers_.size() / 2; ++i) {
    cudaMemcpyAsync(outputBuffers_[2 * i].get(),
                    outputBuffers_[2 * i + 1].get(), outputBufferSizes_[i],
                    cudaMemcpyDeviceToHost, stream_);
  }
  cudaStreamSynchronize(stream_);

  outputData.resize(outputBufferSizes_[0] / sizeof(float));
  std::memcpy(outputData.data(), outputBuffers_[0].get(),
              outputBufferSizes_[0]);

  return true;
}

bool TensorrtEngine::loadEngine() {
  std::ifstream file(enginePath_, std::ios::binary);
  if (!file.good()) {
    std::cerr << "Error opening engine file: " << enginePath_ << std::endl;
    return false;
  }
  file.seekg(0, file.end);
  size_t fsize = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(fsize);
  file.read(engineData.data(), fsize);
  file.close();

  runtime_.reset(createInferRuntime(logger));
  engine_.reset(
      runtime_.get()->deserializeCudaEngine(engineData.data(), fsize));

  if (!engine_) {
    std::cerr << "Failed to deserialize CUDA engine." << std::endl;
    return false;
  }

  std::cout << "Engine loaded from " << enginePath_ << std::endl;
  return true;
}

size_t TensorrtEngine::getDataTypeSize(DataType dtype) {
  switch (dtype) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kHALF:
    return 2;
  case DataType::kINT8:
    return 1;
  case DataType::kINT32:
    return 4;
  default:
    return 4; // Default to 4 (float)
  }
}

bool TensorrtEngine::allocateBuffers() {
  int nbBindings = engine_.get()->getNbIOTensors();
  bindings_.resize(nbBindings);
  inputBuffers_.resize(0);
  outputBuffers_.resize(0);
  inputBufferSizes_.resize(0);
  outputBufferSizes_.resize(0);

  for (int i = 0; i < nbBindings; i++) {
    const char *tensorName = engine_.get()->getIOTensorName(i);
    Dims dims = engine_.get()->getTensorShape(tensorName);
    size_t vol = 1;
    for (int j = 0; j < dims.nbDims; j++) {
      vol *= dims.d[j];
    }

    DataType dtype = engine_.get()->getTensorDataType(tensorName);
    size_t typeSize = getDataTypeSize(dtype);
    size_t memSize = vol * typeSize;

    auto hostMem = CudaMemory::allocateHost(memSize);
    auto deviceMem = CudaMemory::allocateDevice(memSize);

    bindings_[i] = deviceMem.get();

    if (engine_.get()->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
      inputBuffers_.push_back(std::move(hostMem));
      inputBuffers_.push_back(std::move(deviceMem));
      inputBufferSizes_.push_back(memSize);
    } else {
      outputBuffers_.push_back(std::move(hostMem));
      outputBuffers_.push_back(std::move(deviceMem));
      outputBufferSizes_.push_back(memSize);
    }
  }
  return true;
}

void TensorrtEngine::setTensorAddresses() {
  for (size_t i = 0; i < bindings_.size(); i++) {
    const char *tensorName = engine_.get()->getIOTensorName(i);
    context_.get()->setTensorAddress(tensorName, bindings_[i]);
    std::cout << "  Set address for tensor: " << tensorName << std::endl;
  }
}

void TensorrtEngine::cleanup() {
  if (stream_)
    cudaStreamDestroy(stream_);
}