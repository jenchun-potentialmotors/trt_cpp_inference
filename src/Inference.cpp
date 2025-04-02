/* Copyright 2023 Potential Motors.
  @authors {Vinicius de A. Lima} */
#include <stdexcept>
#include <memory>
#include <string>
#include "Inference.h"
#include "Tensor.h"
#include "InferenceEngine.h"
// #include "src/ai/OnnxRuntimeEngine.h"
// #ifdef ENABLE_TFLITE
// #include "src/ai/TfLiteEngine.h"
// #endif
#ifdef ENABLE_TENSORRT
#include "TensorrtEngine.h"
#endif

namespace {

//TODO: All this input validation is still an achitectural challenge.
// On one side it makes sense to have it in the implementation instead of
// in the interface since but on the other hand that means it will be
// scatterer across the code base making it difficult for someone to
// understand everything we are validating against.
// For now, the main input validations are here but it is probably
// not sustainable.
//
// Originally the idea was to have all validation done in the Configuration
// module. I guess is still the best alternative.
void file_exists(const std::filesystem::path& path)
{
  if (not std::filesystem::exists(path))
    throw std::invalid_argument(std::string(path) + " doesn't exist.");
}

inline bool is_not_arm_device()
{
    bool is_not_arm = true;
#if defined(__arm__) || defined(__aarch64__)
    is_not_arm = false;
#endif
    return is_not_arm;
}


// void validate_user_inputs(const oros::Inference::Options& options)
// {
//   file_exists(options.model_path);

//   if (options.device == oros::Inference::Options::Device::gpu)
//     throw std::invalid_argument("Execution device gpu not supported yet.");

//   // validate options when onnx_runtime is selected.
//   if (options.engine == oros::Inference::Options::Engine::onnxruntime) {
//     if (options.device != oros::Inference::Options::Device::cpu)
//       throw std::invalid_argument("Execution device"
//         " not supported with onnxruntime yet.");
//   }
//   // validate options when tflite is selected
//   else  { // then (options.engine == oros::Inference::Options::tflite)
//     if (options.device == oros::Inference::Options::Device::npu
//           && is_not_arm_device())
//         throw std::invalid_argument(
//           "Execution device npu is only supported in ARM architectures.");
//   }
// }

} // anonymous namespace

namespace oros {


Inference::Options::Options(const std::filesystem::path& model_path,
  Inference::Options::Engine engine,
  Inference::Options::Device device,
  Inference::Options::Threads threads)
    : model_path(model_path), engine(engine), device(device), threads(threads)
{
}



struct Inference::Impl {
  std::shared_ptr<InferenceEngine> engine;
};


Inference::Inference(const Inference::Options& options)
  : pimpl_(std::make_unique<Inference::Impl>())
{
//   validate_user_inputs(options);

//   if (options.engine == Options::Engine::onnxruntime)
//     //TODO pass execution device to onnxruntime
//     pimpl_->engine = std::make_shared<OnnxRuntimeEngine>(options);

#ifdef ENABLE_TENSORRT
  if (options.engine == Options::Engine::tensorrt)
    pimpl_->engine = std::make_shared<TensorrtEngine>(options);
#endif

// #ifdef ENABLE_TFLITE
//   else if (options.engine == Options::Engine::tflite)
//     pimpl_->engine = std::make_shared<TfLiteEngine>(options);
// #endif
//   else
//     throw std::invalid_argument("Provided inference engine is not enabled.");
}


Inference::~Inference() = default;


Tensor Inference::infer(const Tensor& input) const
{
  return pimpl_->engine->infer(input);
}

} //namespace oros
