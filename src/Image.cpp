/* Copyright 2023 Potential Motors.
  @authors {Vincius de A. Lima} */
#include "../include/Image.h"
#include <opencv2/opencv.hpp>

namespace oros {

struct Image::Impl {
  cv::Mat data;
};

Image::Image() : pimpl_(std::make_unique<Image::Impl>()) {}

Image::Image(const cv::Mat &mat) : Image() { pimpl_->data = mat; }

Image::Image(const std::string &path) : Image() {
  pimpl_->data = cv::imread(path, cv::IMREAD_COLOR);
}

uint8_t &Image::at(const uint32_t u, const uint32_t v, const uint32_t c) {
  uint32_t depth = pimpl_->data.elemSize();
  return pimpl_->data.at<cv::Vec3b>(v, u)[2u - c];
}

const uint8_t &Image::at(const uint32_t u, const uint32_t v,
                         const uint32_t c) const {
  return const_cast<Image *>(this)->at(u, v, c);
}

Image Image::crop(const std::array<uint32_t, 2> &p0,
                  const std::array<uint32_t, 2> &p1) const {
  return pimpl_->data(
      {static_cast<int32_t>(p0[1]), static_cast<int32_t>(p1[1])},
      {static_cast<int32_t>(p0[0]), static_cast<int32_t>(p1[0])});
}

bool Image::empty() const { return pimpl_->data.empty(); }

Image Image::resize(uint32_t target_width, uint32_t target_height) const {
  cv::Mat out;
  cv::resize(pimpl_->data, out, cv::Size(target_width, target_height), 0, 0,
             cv::INTER_AREA);
  return out;
}

uint32_t Image::height() const {
  return static_cast<uint32_t>(pimpl_->data.rows);
}

uint32_t Image::width() const {
  return static_cast<uint32_t>(pimpl_->data.cols);
}

void Image::show(bool block) const {
  cv::Mat mat = pimpl_->data.clone();
  cv::imshow("oros::Image", mat);
  if (block)
    cv::waitKey();
  else
    cv::waitKey(1);
}

/* boilerplate */
Image::~Image() = default;

Image::Image(const Image &a) : Image() { *pimpl_ = *(a.pimpl_); }

Image::Image(Image &&a) : Image() { *pimpl_ = std::move(*(a.pimpl_)); }

Image &Image::operator=(const Image &a) {
  Image temp(a);
  std::swap(pimpl_, temp.pimpl_);
  return *this;
}

Image &Image::operator=(Image &&a) {
  std::swap(pimpl_, a.pimpl_);
  return *this;
}

} // namespace oros
