/* Copyright 2023 Potential Motors.
  @authors {Vinicius de A. Lima} */
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace oros {

/** Abstracts a image-like object.

  Images are stored as 8 bits 3 channels RGB color format.
  The origin is at the top left corner and pixel coordinates are
  represented by the set {u,v}. The component u runs along the width and
  the component v along the height. */
class Image {
public:
  /** Create an empty image. */
  Image();

  /** Implicit conversion from from OpenCV image.

    The opencv image will be converted to 8 bits 3 channels format. */
  Image(const cv::Mat &);

  /** Construct from an image in the filesystem. */
  Image(const std::string &path);

  /** Reference to the value of the channel of a pixel.

    @param u is the horizontal coordinate from the top left corner.
    @param v is the vertical coordinate from the top left corner.
    @param c is the color coordinate in RGB order. */
  uint8_t &at(const uint32_t u, const uint32_t v, const uint32_t c);
  const uint8_t &at(const uint32_t u, const uint32_t v, const uint32_t c) const;

  /** Rectangular crop of the image.

    The rectangle is defined by its top-to-bottom, left-to-right diagonal.
    The upper boundary is not included, meaning, the row and column defined
    by the second point p1 will not be part of the cropped image.

    @param p0 is the pixel coordinate of the first point.
    @param p1 is the pixel coordinate of the second point.*/
  Image crop(const std::array<uint32_t, 2> &p0,
             const std::array<uint32_t, 2> &p1) const;

  /** Check if the image is empty. */
  bool empty() const;

  /** Resize to the target size not keeping aspect ratio. */
  Image resize(uint32_t target_width, uint32_t target_height) const;

  /** The height of the image in number of pixels. */
  uint32_t height() const;

  /** The width of the image in number of pixels. */
  uint32_t width() const;

  /** Display the image in the screen.

    Use for debug purposes only. */
  [[deprecated("show should be used only for debugging.")]] void
  show(bool block = true) const;

  /* Boilerplate */
  ~Image();
  Image(const Image &);
  Image(Image &&);
  Image &operator=(const Image &);
  Image &operator=(Image &&);

private:
  class Impl;
  std::unique_ptr<Image::Impl> pimpl_;
};

} // namespace oros
