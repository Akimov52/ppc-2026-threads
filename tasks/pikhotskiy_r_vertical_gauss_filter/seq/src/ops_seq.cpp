#include "pikhotskiy_r_vertical_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace pikhotskiy_r_vertical_gauss_filter {

namespace {
constexpr int kKernelNorm = 16;
constexpr int kStripeDivider = 8;

constexpr int ClampIndex(int value, int upper_bound) noexcept {
  if (upper_bound <= 0) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value >= upper_bound) {
    return upper_bound - 1;
  }
  return value;
}

constexpr std::size_t ToLinearIndex(int x, int y, int width) noexcept {
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
}

std::uint8_t NormalizeAndRoundUp(int sum) {
  return static_cast<std::uint8_t>((sum + (kKernelNorm - 1)) / kKernelNorm);
}
}  // namespace

PikhotskiyRVerticalGaussFilterSEQ::PikhotskiyRVerticalGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool PikhotskiyRVerticalGaussFilterSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.width <= 0 || in.height <= 0) {
    return false;
  }
  const auto expected_size = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height);
  if (in.data.size() != expected_size) {
    return false;
  }
  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::PreProcessingImpl() {
  const auto &in = GetInput();
  width_ = in.width;
  height_ = in.height;
  stripe_width_ = std::max(1, width_ / kStripeDivider);

  source_ = in.data;
  vertical_buffer_.assign(source_.size(), 0);
  result_buffer_.assign(source_.size(), 0);
  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::RunImpl() {
  const auto expected_size = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  if (width_ <= 0 || height_ <= 0 || source_.size() != expected_size || vertical_buffer_.size() != expected_size ||
      result_buffer_.size() != expected_size) {
    return false;
  }

  for (int x_begin = 0; x_begin < width_; x_begin += stripe_width_) {
    const int x_end = std::min(width_, x_begin + stripe_width_);
    RunVerticalPassForStripe(x_begin, x_end);
  }

  for (int x_begin = 0; x_begin < width_; x_begin += stripe_width_) {
    const int x_end = std::min(width_, x_begin + stripe_width_);
    RunHorizontalPassForStripe(x_begin, x_end);
  }

  return true;
}

bool PikhotskiyRVerticalGaussFilterSEQ::PostProcessingImpl() {
  GetOutput().width = width_;
  GetOutput().height = height_;
  GetOutput().data = result_buffer_;
  return true;
}

void PikhotskiyRVerticalGaussFilterSEQ::RunVerticalPassForStripe(int x_begin, int x_end) {
  for (int y = 0; y < height_; ++y) {
    const int y_top = ClampIndex(y - 1, height_);
    const int y_bottom = ClampIndex(y + 1, height_);

    for (int x = x_begin; x < x_end; ++x) {
      const std::size_t center = ToLinearIndex(x, y, width_);
      const std::size_t top = ToLinearIndex(x, y_top, width_);
      const std::size_t bottom = ToLinearIndex(x, y_bottom, width_);
      vertical_buffer_[center] =
          static_cast<int>(source_[top]) + (2 * static_cast<int>(source_[center])) + static_cast<int>(source_[bottom]);
    }
  }
}

void PikhotskiyRVerticalGaussFilterSEQ::RunHorizontalPassForStripe(int x_begin, int x_end) {
  for (int y = 0; y < height_; ++y) {
    for (int x = x_begin; x < x_end; ++x) {
      const int x_left = ClampIndex(x - 1, width_);
      const int x_right = ClampIndex(x + 1, width_);
      const std::size_t center = ToLinearIndex(x, y, width_);
      const std::size_t left = ToLinearIndex(x_left, y, width_);
      const std::size_t right = ToLinearIndex(x_right, y, width_);
      const int weighted_sum = vertical_buffer_[left] + (2 * vertical_buffer_[center]) + vertical_buffer_[right];
      result_buffer_[center] = NormalizeAndRoundUp(weighted_sum);
    }
  }
}

}  // namespace pikhotskiy_r_vertical_gauss_filter
