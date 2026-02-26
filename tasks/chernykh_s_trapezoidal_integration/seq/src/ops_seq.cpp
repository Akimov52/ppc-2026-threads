#include "chernykh_s_trapezoidal_integration/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "chernykh_s_trapezoidal_integration/common/include/common.hpp"

namespace chernykh_s_trapezoidal_integration {

ChernykhSTrapezoidalIntegrationSEQ::ChernykhSTrapezoidalIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

// NOLINTNEXTLINE(misc-no-recursion)
void ChernykhSTrapezoidalIntegrationSEQ::RecursiveMethod(std::size_t dim, std::vector<double> &current_point,
                                                         double current_coeff, const IntegrationInType &input,
                                                         double &total_sum) {
  if (dim == input.limits.size()) {
    total_sum += input.func(current_point) * current_coeff;
    return;
  }

  double a = input.limits[dim].first;
  double b = input.limits[dim].second;
  std::size_t n = input.steps[dim];
  double h = (b - a) / static_cast<double>(n);

  for (std::size_t i = 0; i <= n; i++) {
    current_point[dim] = a + (static_cast<double>(i) * h);
    double local_coeff = (i == 0 || i == n) ? 0.5 : 1.0;
    RecursiveMethod(dim + 1, current_point, local_coeff * current_coeff, input, total_sum);
  }
}

bool ChernykhSTrapezoidalIntegrationSEQ::ValidationImpl() {
  const auto &input = this->GetInput();

  if (input.limits.empty()) {
    return false;
  }

  if (input.limits.size() != input.steps.size()) {
    return false;
  }
  return std::ranges::all_of(input.steps, [](int s) { return s > 0; });
}

bool ChernykhSTrapezoidalIntegrationSEQ::PreProcessingImpl() { return true; }

bool ChernykhSTrapezoidalIntegrationSEQ::RunImpl() {
  const auto &input = this->GetInput();
  std::size_t dims = input.limits.size();
  std::vector<double> current_point(dims);
  double total_sum = 0.0;

  RecursiveMethod(0, current_point, 1.0, input, total_sum);
  double h_prod = 1.0;
  for (std::size_t i = 0; i < dims; ++i) {
    h_prod *= (input.limits[i].second - input.limits[i].first) / static_cast<double>(input.steps[i]);
  }

  GetOutput() = total_sum * h_prod;

  return true;
}

bool ChernykhSTrapezoidalIntegrationSEQ::PostProcessingImpl() { return true; }

}  // namespace chernykh_s_trapezoidal_integration