#pragma once

#include "zavyalov_a_complex_sparse_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zavyalov_a_compl_sparse_matr_mult {

class ZavyalovAComplSparseMatrMultSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit ZavyalovAComplSparseMatrMultSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static SparseMatrix MultiplicateWithStl(const SparseMatrix &matr_a, const SparseMatrix &matr_b);

};

}  // namespace zavyalov_a_compl_sparse_matr_mult
