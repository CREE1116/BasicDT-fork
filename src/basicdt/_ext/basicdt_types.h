#pragma once

#include <cstdint>
#include <vector>

// BasicDTTree — fitted axis-aligned tree (sparse heap layout, serializable).

struct BasicDTTree {
  int max_depth   = 0;
  int K           = 0;
  int D           = 0;
  int total_nodes = 0;
  std::vector<int>     split_feature;   // [total_nodes]: split feature index (-1 if leaf)
  std::vector<float>   split_threshold; // [total_nodes]: threshold value
  std::vector<float>   leaf_values;     // [total_nodes × K]
  std::vector<uint8_t> is_leaf;         // [total_nodes]
  std::vector<float>   split_gain;      // [total_nodes]
  std::vector<int>     left_child;      // [total_nodes]: left child pointer (-1 if leaf)
  std::vector<int>     right_child;     // [total_nodes]: right child pointer (-1 if leaf)

  int D_num = 0;
  std::vector<float> na_means;  // [D]: numeric μ_f impute; cat cols: NaN-category rank
  std::vector<std::vector<std::pair<int, float>>> cat_ranks;  // [D_cat]: raw value → rank sorted by raw value

  // Pre-flattened categorical ranks cache for O(1) array lookup in predict
  std::vector<std::vector<float>> flat_cat_ranks;
  std::vector<int> cat_min_val;
  std::vector<uint8_t> use_flat_lookup;
};
