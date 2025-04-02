/* Copyright 2023 Potential Motors.
  @authors {Vinicius de A. Lima} */
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include "Tensor.h"

namespace oros {

class Inference {
public:
  /** Common execution option passed to any Inference implementation. */
  struct Options {
    std::string model_path;
    /** Runtime engine (backend) used for inferences. */
    enum Engine {onnxruntime, tflite, tensorrt} engine;
    /** Device to execute on. */
    enum Device {cpu, gpu, npu} device;
    /** If device=Device::cpu set the number of inter/intra threads. */
    struct Threads {
      uint32_t inter;
      uint32_t intra;
    } threads;

    Options() = delete;

    /** Complete construction.
      @param model_path is the path to an inference model.
        Models supported are onnx and tflite.
      @param engine the backend engine to execute the model, options are:
        Engine::onnxruntime, Engine::tflite.
      @param device is the hardware to execute the model, options are:
        Device::cpu, Device::npu.
      @param threads is a Threads structure definint the number of iter and
        intra threads. Defaults to 1 inter and 1 intra thread. */
    Options(const std::filesystem::path& model_path, Options::Engine engine,
      Options::Device device, Options::Threads threads);
  };

  /** Initialize the inference selecting an inference Engine.

    @param options @see oros::Inference::Options */
  Inference(const Inference::Options& options);

  /** Make a inference.

    For convenience the base class provides an implementation that can
    be called explicitly by derived classes.

    @param input is a Tensor with the input data.
    @return a Tensor with the output data. */
  virtual Tensor infer(const Tensor& input) const = 0;

  // Polymorphic class, rule of 6
  virtual ~Inference();
  Inference() = delete;
  Inference(const Inference&) = delete;
  Inference(Inference&&) = delete;
  Inference& operator=(const Inference&) = delete;
  Inference& operator=(Inference&&) = delete;

protected:
  class Impl;
  std::unique_ptr<Inference::Impl> pimpl_;
};

} // namespace oros
