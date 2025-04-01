/* Copyright 2024 Potential Motors.
  @authors {Vinicius de A. Lima} */
#pragma once
#include "../include/Image.h"
#include <array>
#include <cinttypes>
#include <memory>

namespace oros {

class Tensor {
public:
  /** Constructs a tensor from a image. */
  Tensor(const Image &, const std::array<float, 3> &mean,
         const std::array<float, 3> &stddev);

  /** Constructs a tensor to store a number of element regardless of shape. */
  Tensor(const uint32_t size);

  /** Direct access to the underlying contiguous storage. */
  float *data();

  // Need to be defaulted/copyable/movable, so rule of 6
  Tensor();
  Tensor(const Tensor &);
  Tensor(Tensor &&);
  Tensor &operator=(const Tensor &);
  Tensor &operator=(Tensor &&);
  ~Tensor();

  // Gives some compatibility with C++ containers
  float at(uint32_t idx) const;
  float &at(uint32_t idx);
  float operator[](uint32_t idx) const;
  float &operator[](uint32_t idx);

  const float *cbegin() const;
  const float *begin() const;
  float *begin();

  const float *cend() const;
  const float *end() const;
  float *end();

  uint32_t size() const;
  explicit operator std::vector<float>() const;

protected:
  class Impl;
  std::unique_ptr<Tensor::Impl> pimpl_;
};

} // namespace oros
