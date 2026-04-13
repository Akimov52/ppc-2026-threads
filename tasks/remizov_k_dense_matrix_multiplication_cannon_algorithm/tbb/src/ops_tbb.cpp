#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/tbb/include/ops_tbb.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/common/include/common.hpp"

namespace remizov_k_dense_matrix_multiplication_cannon_algorithm {

RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::ValidationImpl() {
  const auto &input_data = GetInput();

  int block_dim = std::get<0>(input_data);
  const auto &mat_a = std::get<1>(input_data);
  const auto &mat_b = std::get<2>(input_data);

  if (block_dim <= 0) {
    return false;
  }
  if (mat_a.empty() || mat_b.empty()) {
    return false;
  }

  size_t n = mat_a.size();
  if (n != mat_a[0].size()) {
    return false;
  }
  if (n != mat_b.size() || n != mat_b[0].size()) {
    return false;
  }

  return (n % static_cast<size_t>(block_dim) == 0);
}

}  // namespace remizov_k_dense_matrix_multiplication_cannon_algorithm
