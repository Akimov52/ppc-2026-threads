#include "tasks/gusev_d_double_sort_even_odd_batcher_omp/omp/include/ops_omp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads {
namespace {

constexpr int kRadixPasses = 8;
constexpr int kBitsPerByte = 8;
constexpr size_t kRadixBuckets = 256;
constexpr uint64_t kBucketMask = 0xFFULL;

using Block = std::vector<ValueType>;
using BlockList = std::vector<Block>;

struct BlockRange {
  size_t begin = 0;
  size_t end = 0;
};

uint64_t DoubleToSortableKey(ValueType value) {
  const auto bits = std::bit_cast<uint64_t>(value);
  const auto sign_mask = uint64_t{1} << 63;
  return (bits & sign_mask) == 0 ? bits ^ sign_mask : ~bits;
}

size_t GetBucketIndex(ValueType value, int shift) {
  return static_cast<size_t>((DoubleToSortableKey(value) >> shift) & kBucketMask);
}

void BuildPrefixSums(std::array<size_t, kRadixBuckets>& count) {
  size_t prefix = 0;
  for (auto& value : count) {
    const auto current = value;
    value = prefix;
    prefix += current;
  }
}

void RadixSortDoubles(Block& data) {
  if (data.size() < 2) {
    return;
  }

  Block buffer(data.size());
  auto* src = &data;
  auto* dst = &buffer;

  for (int byte = 0; byte < kRadixPasses; ++byte) {
    std::array<size_t, kRadixBuckets> count{};
    const auto shift = byte * kBitsPerByte;

    for (ValueType value : *src) {
      count.at(GetBucketIndex(value, shift))++;
    }
    BuildPrefixSums(count);

    for (ValueType value : *src) {
      const auto bucket = GetBucketIndex(value, shift);
      (*dst)[count.at(bucket)++] = value;
    }

    std::swap(src, dst);
  }

  if (src != &data) {
    data = std::move(*src);
  }
}

void SplitByGlobalParity(const Block& source, size_t global_offset, Block& even, Block& odd) {
  even.clear();
  odd.clear();
  even.reserve((source.size() + 1) / 2);
  odd.reserve(source.size() / 2);

  for (size_t i = 0; i < source.size(); ++i) {
    if (((global_offset + i) & 1U) == 0U) {
      even.push_back(source[i]);
    } else {
      odd.push_back(source[i]);
    }
  }
}

Block InterleaveParityGroups(size_t total_size, const Block& even, const Block& odd) {
  Block result(total_size);
  size_t even_index = 0;
  size_t odd_index = 0;

  for (size_t i = 0; i < total_size; ++i) {
    if ((i & 1U) == 0U) {
      result[i] = even[even_index++];
    } else {
      result[i] = odd[odd_index++];
    }
  }

  return result;
}

void OddEvenFinalize(Block& result) {
  for (size_t phase = 0; phase < result.size(); ++phase) {
    const auto start = phase & 1U;
    for (size_t i = start; i + 1 < result.size(); i += 2) {
      if (result[i] > result[i + 1]) {
        std::swap(result[i], result[i + 1]);
      }
    }
  }
}

void SplitBlocksByParity(const Block& left, const Block& right, Block& left_even, Block& left_odd, Block& right_even,
                         Block& right_odd) {
  SplitByGlobalParity(left, 0, left_even, left_odd);
  SplitByGlobalParity(right, left.size(), right_even, right_odd);
}

void MergeParityGroups(const Block& left_even, const Block& right_even, const Block& left_odd, const Block& right_odd,
                       Block& merged_even, Block& merged_odd) {
  merged_even.clear();
  merged_odd.clear();
  merged_even.reserve(left_even.size() + right_even.size());
  merged_odd.reserve(left_odd.size() + right_odd.size());

  std::ranges::merge(left_even, right_even, std::back_inserter(merged_even));
  std::ranges::merge(left_odd, right_odd, std::back_inserter(merged_odd));
}

Block MergeBatcherEvenOdd(const Block& left, const Block& right) {
  Block left_even;
  Block left_odd;
  Block right_even;
  Block right_odd;

  SplitBlocksByParity(left, right, left_even, left_odd, right_even, right_odd);

  Block merged_even;
  Block merged_odd;
  MergeParityGroups(left_even, right_even, left_odd, right_odd, merged_even, merged_odd);

  auto result = InterleaveParityGroups(left.size() + right.size(), merged_even, merged_odd);
  OddEvenFinalize(result);
  return result;
}

BlockRange GetBlockRange(size_t block_index, size_t block_count, size_t total_size) {
  return {
      .begin = (block_index * total_size) / block_count,
      .end = ((block_index + 1) * total_size) / block_count,
  };
}

size_t GetBlockCount(size_t input_size) {
  if (input_size == 0) {
    return 0;
  }

  const auto omp_threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
  return std::max<size_t>(1, std::min(input_size, omp_threads));
}

void FillAndSortBlock(const std::vector<ValueType>& input, Block& block, BlockRange range) {
  block.assign(input.begin() + static_cast<std::ptrdiff_t>(range.begin), input.begin() + static_cast<std::ptrdiff_t>(range.end));
  RadixSortDoubles(block);
}

BlockList MakeSortedBlocks(const std::vector<ValueType>& input) {
  const auto block_count = GetBlockCount(input.size());
  const auto total_size = input.size();

  BlockList blocks(block_count);

#pragma omp parallel for schedule(static) if(block_count > 1)
  for (long long block = 0; block < static_cast<long long>(block_count); ++block) {
    const auto index = static_cast<size_t>(block);
    FillAndSortBlock(input, blocks[index], GetBlockRange(index, block_count, total_size));
  }

  return blocks;
}

Block MergeBlocks(BlockList blocks) {
  if (blocks.empty()) {
    return {};
  }

  while (blocks.size() > 1) {
    const auto pair_count = blocks.size() / 2;
    BlockList next((blocks.size() + 1) / 2);

#pragma omp parallel for schedule(static) if(pair_count > 1)
    for (long long pair = 0; pair < static_cast<long long>(pair_count); ++pair) {
      const auto index = static_cast<size_t>(pair);
      next[index] = MergeBatcherEvenOdd(blocks[index * 2], blocks[(index * 2) + 1]);
    }

    if ((blocks.size() & 1U) != 0U) {
      next.back() = std::move(blocks.back());
    }

    blocks = std::move(next);
  }

  return blocks.empty() ? std::vector<double>{} : std::move(blocks.front());
}

}  // namespace

DoubleSortEvenOddBatcherOMP::DoubleSortEvenOddBatcherOMP(const InType& in) : BaseTask(in) {
  internal_order_test_ = true;
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DoubleSortEvenOddBatcherOMP::ValidationImpl() { return GetOutput().empty(); }

bool DoubleSortEvenOddBatcherOMP::PreProcessingImpl() {
  input_data_ = GetInput();
  result_data_.clear();
  return true;
}

bool DoubleSortEvenOddBatcherOMP::RunImpl() {
  if (input_data_.empty()) {
    result_data_.clear();
    return true;
  }

  auto blocks = MakeSortedBlocks(input_data_);
  result_data_ = MergeBlocks(std::move(blocks));
  return true;
}

bool DoubleSortEvenOddBatcherOMP::PostProcessingImpl() {
  SetOutput(result_data_);
  return true;
}

}  // namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads
