#include "kolotukhin_a_gaussian_blur/tbb/include/ops_tbb.hpp"

#include <tbb/parallel_for.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "kolotukhin_a_gaussian_blur/common/include/common.hpp"

namespace kolotukhin_a_gaussian_blur {

KolotukhinAGaussinBlureTBB::KolotukhinAGaussinBlureTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KolotukhinAGaussinBlureTBB::ValidationImpl() {
  const auto &pixel_data = std::get<0>(GetInput());
  const auto img_width = std::get<1>(GetInput());
  const auto img_height = std::get<2>(GetInput());

  return static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width) == pixel_data.size();
}

bool KolotukhinAGaussinBlureTBB::PreProcessingImpl() {
  const auto img_width = std::get<1>(GetInput());
  const auto img_height = std::get<2>(GetInput());

  GetOutput().assign(static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width), 0);
  return true;
}

bool KolotukhinAGaussinBlureTBB::RunImpl() {
  const auto &pixel_data = std::get<0>(GetInput());
  const auto img_width = std::get<1>(GetInput());
  const auto img_height = std::get<2>(GetInput());

  auto &output = GetOutput();

  std::vector<std::uint8_t> temp(pixel_data.size());

  tbb::parallel_for(0, img_height, [&](int y) {
    for (int x = 0; x < img_width; x++) {
      int sum = GetPixel(pixel_data, img_width, img_height, x - 1, y) +
                2 * GetPixel(pixel_data, img_width, img_height, x, y) +
                GetPixel(pixel_data, img_width, img_height, x + 1, y);
      temp[y * img_width + x] = static_cast<std::uint8_t>(sum / 4);
    }
  });

  tbb::parallel_for(0, img_height, [&](int y) {
    for (int x = 0; x < img_width; x++) {
      int sum = GetPixel(temp, img_width, img_height, x, y - 1) + 2 * GetPixel(temp, img_width, img_height, x, y) +
                GetPixel(temp, img_width, img_height, x, y + 1);
      output[y * img_width + x] = static_cast<std::uint8_t>(sum / 4);
    }
  });

  return true;
}

std::uint8_t KolotukhinAGaussinBlureTBB::GetPixel(const std::vector<std::uint8_t> &pixel_data, int img_width,
                                                  int img_height, int pos_x, int pos_y) {
  std::size_t x = static_cast<std::size_t>(std::max(0, std::min(pos_x, img_width - 1)));
  std::size_t y = static_cast<std::size_t>(std::max(0, std::min(pos_y, img_height - 1)));
  return pixel_data[(y * static_cast<std::size_t>(img_width)) + x];
}

bool KolotukhinAGaussinBlureTBB::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_gaussian_blur
