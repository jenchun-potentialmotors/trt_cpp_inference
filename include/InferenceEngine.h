/* Copyright 2023 Potential Motors.
  @authors {Vinicius de A. Lima} */
#pragma once
#include "Tensor.h"

namespace oros {

/** Base class for abstracting different inference engines. */
class InferenceEngine {
public:
  /** Run the inference using the underlying execution library. */
  virtual Tensor infer(const Tensor&) const = 0;

  /* Rule of 6 */
  virtual ~InferenceEngine() = default;
  InferenceEngine() {};
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine(InferenceEngine&&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  InferenceEngine& operator=(InferenceEngine&&) = delete;
};

} // namespace oros
