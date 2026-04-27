#include "zavyalov_a_complex_sparse_matrix_mult/stl/include/ops_stl.hpp"

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include "util/include/util.hpp"
#include "zavyalov_a_complex_sparse_matrix_mult/common/include/common.hpp"

namespace zavyalov_a_compl_sparse_matr_mult {

SparseMatrix ZavyalovAComplSparseMatrMultSTL::MultiplicateWithStl(const SparseMatrix &matr_a,
                                                                  const SparseMatrix &matr_b) {
  if (matr_a.width != matr_b.height) {
    throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
  }

  int num_threads = ppc::util::GetNumThreads();
  size_t total = matr_a.Count();

  std::vector<std::map<std::pair<size_t, size_t>, Complex>> local_maps(num_threads);

  auto worker = [&](int tid, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t row_a = matr_a.row_ind[i];
      size_t col_a = matr_a.col_ind[i];
      Complex val_a = matr_a.val[i];

      for (size_t j = 0; j < matr_b.Count(); ++j) {
        if (col_a == matr_b.row_ind[j]) {
          local_maps[tid][{row_a, matr_b.col_ind[j]}] += val_a * matr_b.val[j];
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  size_t chunk = (total + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    size_t start = t * chunk;
    size_t end = std::min(start + chunk, total);
    if (start < total) {
      threads.emplace_back(worker, t, start, end);
    }
  }

  for (auto &th : threads) {
    th.join();
  }

  std::map<std::pair<size_t, size_t>, Complex> mp;
  for (auto &lm : local_maps) {
    for (auto &[key, value] : lm) {
      mp[key] += value;
    }
  }

  SparseMatrix res;
  res.width = matr_b.width;
  res.height = matr_a.height;
  for (const auto &[key, value] : mp) {
    res.val.push_back(value);
    res.row_ind.push_back(key.first);
    res.col_ind.push_back(key.second);
  }

  return res;
}

ZavyalovAComplSparseMatrMultSTL::ZavyalovAComplSparseMatrMultSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ZavyalovAComplSparseMatrMultSTL::ValidationImpl() {
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());
  return matr_a.width == matr_b.height;
}

bool ZavyalovAComplSparseMatrMultSTL::PreProcessingImpl() {
  return true;
}

bool ZavyalovAComplSparseMatrMultSTL::RunImpl() {
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());

  GetOutput() = MultiplicateWithStl(matr_a, matr_b);

  return true;
}

bool ZavyalovAComplSparseMatrMultSTL::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_compl_sparse_matr_mult
