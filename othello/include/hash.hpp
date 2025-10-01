#pragma once
#include <const.h>
#include <dbg.hpp>
#include <game.hpp>
#include <random>
class Hash_ctx {
private:
  void hash_init() {
    std::mt19937_64 rng(seed);
    for(int player= 0; player < 2; ++player)
      for(int square= 0; square < 64; ++square) zobrist_hash[player][square]= rng();
    hash_side= rng();
  }
public:
  Hash_ctx(int seed): seed(seed) { hash_init(); }
  uint64_t zobrist_hash[2][64]; // Ramdom board for hashing, index is
                                // [player_num][board_size]
  uint64_t hash_side;           // Random num for the one to move
  int seed;                     // Seed for random
};
template<typename Hash_ctx> class Zobrist_hash_impl {
public:
  Zobrist_hash_impl(Hash_ctx& hash_ctx): hash_ctx(hash_ctx) {}
  /*
   * @brief filling random integer for zobrist_hash and hash_side
   *
   * std::mt19937_64 using Mersenne Twister Algorithm, generate uint64.
   * XOR hash 容易碰撞嘛？
   */
  uint64_t hash_position(const Position& pos) {
    // check for nullptr
    // assert (pos_ptr);
    // auto& pos = *pos_ptr;
    uint64_t hash= 0;
    const auto& black= pos.black;
    const auto& white= pos.white;
    const auto& is_black_move= pos.is_black_move;
    uint64_t bitboard= black;
    // XOR to all black
    while(bitboard) {
      // 第一個 1 的 offset (第一個 1 之前的低位 0 的數量)。
      int square= __builtin_ctzll(bitboard);
      hash^= zobrist_hash[0][square];
      // 清除第一個低位 1
      bitboard&= bitboard - 1;
    }
    // XOR to all white
    bitboard= white;
    while(bitboard) {
      int square= __builtin_ctzll(bitboard);
      hash^= zobrist_hash[1][square];
      bitboard&= bitboard - 1;
    }
    if(is_black_move) hash^= hash_side; // 輪到黑走
    return hash;
  }
private:
  Hash_ctx& hash_ctx;
  // renaming for hash_ctx
  using zobrist_hash_t= decltype(Hash_ctx::zobrist_hash);
  zobrist_hash_t& zobrist_hash= hash_ctx.zobrist_hash;
  using hash_side_t= decltype(Hash_ctx::hash_side);
  hash_side_t& hash_side= hash_ctx.hash_side;
  using seed_t= decltype(Hash_ctx::seed);
  const seed_t& seed= hash_ctx.seed;
};
// concept for Hash ad Hash_method_t
// concept 像多態中虛函數能強制別人實現 API，使用 concept
// 能在模板注入時，確保該類型存在 API。
template<typename T>
concept Hashable= requires(T target, uint64_t h, Position pos) {
  { target.hash_position(h, pos) } -> std::same_as<void>;
};
template<Hashable Hash_impl> class Hash {
  Hash_impl& hash;
};
/*
 * @brief 已有雜湊值 h，計算下完這一步後的新雜湊
 * Using previous hash h, is_black_move, move_square and flips_mask to hash.
 */
/*
uint64_t
hash_after_move (uint64_t h,
                 bool is_black_move,
                 int move_square,
                 uint64_t flips_mask)
{
  // 先 XOR side：因為換手
  h ^= hash_side;
  int me = is_black_move ? 0 : 1; // 0=黑,1=白
  int opp = is_black_move ? 1 : 0;
  // 放上新棋
  h ^= zobrist_hash[me][move_square];
  // 被翻轉的每一顆子：從對手 → 我方
  uint64_t bitboard = flips_mask;
  while (bitboard)
  {
    int square = __builtin_ctzll (bitboard);
    h ^= zobrist_hash[opp][square]; // 先移除對手
    h ^= zobrist_hash[me][square];  // 加上我方
    bitboard &= bitboard - 1;
  }
  return h;
}
// 過手（pass）只需 XOR side
inline uint64_t
hash_after_pass (uint64_t h)
{
  return h ^ hash_side;
}
*/
//// C++11 enum:type, normaly it use int(4bytes), we using uint8(1byte).
//// enum class forcing type and only using Bound::BOUND_EXACT to access and
//// forbid implicit type convertion.
// enum class Bound : uint8_t
//{
// EXACT,
// ALPHA,
// BETA
//};
// struct TTEntry
//{
// uint64_t key;             // 或者 16~32bit tag
// int16_t value;            // 節點值（注意相對根的手數轉換）
// int16_t depth;            // 搜索深度
// uint8_t bound;            // EXACT / ALPHA / BETA
// uint8_t best_move_square; // 可選：最佳著法（0..63，或 64=pass）
// uint16_t age;             // 可選：時代/輪次，用於替換策略
//};
// struct TT
//{
// TTEntry* table;
// size_t size; // table size, need to be 2^N
// uint16_t age = 0;
// void
// init (size_t bytes)
//{
// size_t n = 1;
//// not <=, because last time will finished the calculation.
// while ((n * sizeof (TTEntry)) < bytes)
// n <<= 1; // n*=2
// size = n;
//// C++11 API, 64 = Aligned borders, ensure the start address is 64*N
//// The size has to be 64*N
// table = (TTEntry*)aligned_alloc (64, size * sizeof (TTEntry));
// memset (table, 0, size * sizeof (TTEntry));
//}
/*
 * @brief hash key -> offset in array "table"
 */
// inline TTEntry*
// probe (uint64_t key)
//{
//// key & (size - 1) is equal to (key % size) but faster
// return &table[key & (size - 1)];
//}
//// depth younger replace strategy
// inline void
// store (uint64_t key, int value, int depth, Bound bound, int best_square)
//{
// TTEntry* e = probe (key);
//// collusion happened or has better data, then replace old one.
// if (e->key != key || depth >= e->depth || age != e->age)
//{
// e->key = key;
// e->value = (int16_t)value;
// e->depth = (int16_t)depth;
// e->bound = (uint8_t)bound;
// e->best_move_square = (uint8_t)best_square;
// e->age = age;
//}
//}
// inline bool
// load (uint64_t key, TTEntry& out)
//{
// TTEntry* e = probe (key);
// if (e->key == key)
//{
// out = *e;
// return true;
//}
// return false;
//}
// inline void
// new_age ()
//{
//++age;
//}
//};
