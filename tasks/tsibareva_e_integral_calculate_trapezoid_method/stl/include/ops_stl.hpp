#pragma once

#include "tsibareva_e_integral_calculate_trapezoid_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tsibareva_e_integral_calculate_trapezoid_method {

class TsibarevaEIntegralCalculateTrapezoidMethodSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit TsibarevaEIntegralCalculateTrapezoidMethodSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsibareva_e_integral_calculate_trapezoid_method
