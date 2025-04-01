/* Copyright 2024 Potential Motors.
  @authors {Vinicius de A. Lima} */
#include "../include/Tensor.h"
#include "../include/Image.h"
#include <vector>

namespace {

inline std::vector<float> image_to_tensor(const oros::Image &img,
                                          const std::array<float, 3> &mean,
                                          const std::array<float, 3> stddev) {
  constexpr uint32_t depth = 3;
  const uint32_t &width = img.width();
  const uint32_t &height = img.height();
  const std::array<float, 3> scale{1.f / stddev[0], 1.f / stddev[1],
                                   1.f / stddev[2]};
  std::vector<float> data(width * height * depth);
  for (uint32_t v = 0; v < height; ++v)
    for (uint32_t u = 0; u < width; ++u)
      for (uint32_t c = 0; c < depth; ++c) {
        float e = img.at(u, v, c);
        data[c * height * width + v * width + u] = (e - mean[c]) * scale[c];
      }
  return data;
}

} // anonymous namespace

namespace oros {

struct Tensor::Impl {
  std::vector<float> data;
};

Tensor::Tensor(const Image &img, const std::array<float, 3> &mean,
               const std::array<float, 3> &stddev)
    : Tensor() {
  pimpl_->data = image_to_tensor(img, mean, stddev);
}

Tensor::Tensor(const uint32_t size) : Tensor() { pimpl_->data.resize(size); }

float *Tensor::data() { return pimpl_->data.data(); }

// Rule of 6
Tensor::Tensor() : pimpl_(std::make_unique<Tensor::Impl>()) {}

Tensor::Tensor(const Tensor &t) : Tensor() { *pimpl_ = *(t.pimpl_); }

Tensor::Tensor(Tensor &&t) : Tensor() { *pimpl_ = std::move(*(t.pimpl_)); }

Tensor &Tensor::operator=(const Tensor &t) {
  Tensor temp(t);
  std::swap(pimpl_, temp.pimpl_);
  return *this;
}

Tensor &Tensor::operator=(Tensor &&t) {
  std::swap(pimpl_, t.pimpl_);
  return *this;
}

Tensor::~Tensor() = default;

// Compatibility with C++ containers
float Tensor::at(uint32_t idx) const { return pimpl_->data.at(idx); }

float &Tensor::at(uint32_t idx) { return pimpl_->data.at(idx); }

float Tensor::operator[](uint32_t idx) const { return pimpl_->data[idx]; }

float &Tensor::operator[](uint32_t idx) { return pimpl_->data[idx]; }

const float *Tensor::cbegin() const { return &(*pimpl_->data.cbegin()); }

const float *Tensor::begin() const { return cbegin(); }

float *Tensor::begin() { return &(*pimpl_->data.begin()); }

const float *Tensor::cend() const { return &(*pimpl_->data.cend()); }

const float *Tensor::end() const { return cend(); }

float *Tensor::end() { return &(*pimpl_->data.end()); }

uint32_t Tensor::size() const { return pimpl_->data.size(); }

Tensor::operator std::vector<float>() const { return pimpl_->data; }

}; // namespace oros
