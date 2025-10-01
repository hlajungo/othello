#pragma once

#include <cstdint>

constexpr uint64_t MASK_11111110 = 0xfefefefefefefefeull;
constexpr uint64_t MASK_01111111 = 0x7f7f7f7f7f7f7f7full;
constexpr uint64_t MASK_11111111 = 0xffffffffffffffffull;

constexpr int SEED = 20250821;
constexpr uint64_t BLACK_INIT = (1ull << (8 * 3 + 4)) | (1ull << (8 * 4 + 3));
constexpr uint64_t WHITE_INIT = (1ull << (8 * 3 + 3)) | (1ull << (8 * 4 + 4));

