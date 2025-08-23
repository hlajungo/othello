#include <bitset>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

#include <stack>
#include <tuple>

#include "dbg.hpp"
#include "type.hpp"

const uint64_t MASK_11111110 = 0xfefefefefefefefeull;
const uint64_t MASK_01111111 = 0x7f7f7f7f7f7f7f7full;
const uint64_t MASK_11111111 = 0xffffffffffffffffull;

constexpr int SEED = 20250821;

/*
 * 編碼
 * bit 0 = A1
 * bit 1 = B1
 * ...
 * bit 7 = H1
 * bit 8 = A2
 * ...
 * bit 63 = H8
 *
 * 棋盤長這樣
 *   HGFEDCBA
 * 8 H8     A8
 * 7
 * 6
 * 5   ...
 * 4
 * 3
 * 2
 * 1 H1     A1
 */

#include <cstdint>
#include <vector>

typedef struct Move
{
  uint64_t flipped_mask; /*flipped squares*/
  int square;            /*square played*/
} Move;

/*
 * 方向向量：
 * 右 = shift << 1
 * 左 = shift >> 1
 * 上 = shift >> 8
 * 下 = shift << 8
 * 右上 = shift >> 7
 * 左上 = shift >> 9
 * 右下 = shift << 9
 * 左下 = shift << 7
 */

/*
 * Logic:
 * O = player X = opponent, . = available move, _ = empty
 * 考慮
 * .XXX XXXO x8
 * player 使用 <<1 變為
 * 0000 0010 x8
 * 0111 1110 & 0000 0010 = 0000 0010 x8 = tmp
 *
 * 進迴圈
 * 1000 0000 & 0000 0100 = 0000 0000, moves 不變
 * 0111 1110 & 0000 0100 = 0000 0100
 *
 * 一直重複，直到
 *
 * 0111 1110 & 0100 0000 = 0100 0000 = tmp
 * 1000 0000 & 1000 0000 = 1000 0000, moves |= 1000 0000
 *
 * 這種執行是由位運算帶來並行性，它能夠自動的檢查任何 .X~XO。
 * 會不斷重複直到，向右移動不帶來任何重疊
 */

// 你不能在算 moveable_mask 時，一起算單獨的 flipped_mask 和
// square。這是由於位運算並行性，你得在算完後，單獨的計算。
void
get_move (uint64_t& moveable_mask,
          const uint64_t& player,
          const uint64_t& opponent)
{
  // right, left, down, up
  constexpr int dir_1d[] = { 1, -1, 8, -8, 9, 7, -7, -9 };
  constexpr uint64_t mask_1d[]
      = { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111,
          MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };

  uint64_t empty = ~(player | opponent);
  for (int dir = 0; dir < 8; ++dir)
  {
    int shift = dir_1d[dir];
    uint64_t mask = mask_1d[dir];

    // 嘗試走一格, 尋找能壓到對手的格子
    uint64_t tmp = opponent & mask
                   & (shift > 0 ? (player << shift) : (player >> -shift));
    // 壓到格子了
    while (tmp)
    {
      // 嘗試走一格，是空的，就加入解
      uint64_t moveable
          = empty & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
      if (moveable)
      {
        moveable_mask |= moveable;
      }
      // 繼續走
      tmp = opponent & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
    }
  }
}

void
get_flipped_mask (array_1d_t<Move>& move_1d,
                  const uint64_t& moveable_mask,
                  const uint64_t& player,
                  const uint64_t& opponent)
{
  // right, left, down, up
  constexpr int dir_1d[] = { 1, -1, 8, -8, 9, 7, -7, -9 };
  constexpr uint64_t mask_1d[]
      = { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111,
          MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };

  uint64_t moves = moveable_mask;
  while (moves)
  {
    // lowest 1 offset
    int square = __builtin_ctzll (moves);
    // lowest 1 mask
    uint64_t move_bit = 1ull << square;
    uint64_t flipped_mask = 0;

    for (int dir = 0; dir < 8; ++dir)
    {
      uint64_t mask = mask_1d[dir];
      int shift = dir_1d[dir];

      uint64_t captured = 0;
      uint64_t tmp = (shift > 0 ? (move_bit << shift) : (move_bit >> -shift));

      while (tmp & mask & opponent)
      {
        captured |= tmp;
        tmp = (shift > 0 ? (tmp << shift) : (tmp >> -shift));
      }
      if (tmp & mask & player)
      {
        flipped_mask |= captured;
      }
    }

    move_1d.push_back ({ flipped_mask, square });
    // moveable_mask |= move_bit;
    moves &= moves - 1; // clear lowest 1
  }
}

void
DOUT_bitboard (uint64_t bitboard)
{
  std::bitset<64> binary (bitboard);
  for (int i = 63; i >= 0; --i)
  {
    DOUT << binary[i];
    if (i % 8 == 0)
    {
      DOUT << "\n";
    }
  }
}

/**
 * @brief Count the number of bits set to one in an unsigned long long.
 *
 * This is the classical popcount function.
 * Since 2007, it is part of the instruction set of some modern CPU,
 * (>= barcelona for AMD or >= nelhacem for Intel). Alternatively,
 * a fast SWAR algorithm, adding bits in parallel is provided here.
 * This function is massively used to count discs on the board,
 * the mobility, or the stability.
 *
 * @param b 64-bit integer to count bits of.
 * @return the number of bits set.
 */

#ifndef POPCOUNT
int
popcount (unsigned long long b)
{
  int c;
  b = b - ((b >> 1) & 0x5555555555555555ull);
  b = ((b >> 2) & 0x3333333333333333ull) + (b & 0x3333333333333333ull);
  b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0Full;
  c = (b * 0x0101010101010101ull) >> 56;
  return c;
}
#endif

bool
is_legal (uint64_t player, uint64_t opponent)
{
  uint64_t overlay = player & opponent;
  if (overlay != 0)
  {
    DOUT << "not legal, overlay\n";
    return false;
  }

  int bit_num = popcount (player) + popcount (opponent);
  if (bit_num < 4)
  {
    DOUT << "not legal, too less bit\n";
    return false;
  }

  return true;
}

uint64_t zobrist_hash[2][64]; /*zobrist_hash[player num][board size]*/
uint64_t hash_side;           /*the one to move*/

/*
 * @brief filling random integer for zobrist_hash and hash_side
 *
 * std::mt19937_64 using Mersenne Twister Algorithm, generate uint64.
 *
 * XOR hash method 好嗎？容易碰撞嘛？
 */
void
zobrist_init ()
{
  std::mt19937_64 rng (SEED);
  for (int player = 0; player < 2; ++player)
    for (int square = 0; square < 64; ++square)
      zobrist_hash[player][square] = rng ();
  hash_side = rng ();
}

uint64_t
hash_position (uint64_t black, uint64_t white, bool is_black_move)
{
  uint64_t h = 0;
  uint64_t bitboard = black;

  // XOR to all black
  while (bitboard)
  {
    // 第一個 1 的 offset (第一個 1 之前的低位 0 的數量)。
    int square = __builtin_ctzll (bitboard);
    h ^= zobrist_hash[0][square];
    // 清除第一個低位 1
    bitboard &= bitboard - 1;
  }

  // XOR to all white
  bitboard = white;
  while (bitboard)
  {
    int square = __builtin_ctzll (bitboard);
    h ^= zobrist_hash[1][square];
    bitboard &= bitboard - 1;
  }
  if (is_black_move)
    h ^= hash_side; // 輪到黑走
  return h;
}

/*
 * @brief 已有雜湊值 h，計算下完這一步後的新雜湊
 * Using previous hash h, is_black_move, move_square and flips_mask to hash.
 */
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

// C++11 enum:type, normaly it use int(4bytes), we using uint8(1byte).
// enum class forcing type and only using Bound::BOUND_EXACT to access and
// forbid implicit type convertion.
enum class Bound : uint8_t
{
  EXACT,
  ALPHA,
  BETA
};

struct TTEntry
{
  uint64_t key;             // 或者 16~32bit tag
  int16_t value;            // 節點值（注意相對根的手數轉換）
  int16_t depth;            // 搜索深度
  uint8_t bound;            // EXACT / ALPHA / BETA
  uint8_t best_move_square; // 可選：最佳著法（0..63，或 64=pass）
  uint16_t age;             // 可選：時代/輪次，用於替換策略
};

struct TT
{
  TTEntry* table;
  size_t size; // table size, need to be 2^N
  uint16_t age = 0;

  void
  init (size_t bytes)
  {
    size_t n = 1;
    // not <=, because last time will finished the calculation.
    while ((n * sizeof (TTEntry)) < bytes)
      n <<= 1; // n*=2
    size = n;
    // C++11 API, 64 = Aligned borders, ensure the start address is 64*N
    // The size has to be 64*N
    table = (TTEntry*)aligned_alloc (64, size * sizeof (TTEntry));
    memset (table, 0, size * sizeof (TTEntry));
  }

  /*
   * @brief hash key -> offset in array "table"
   */
  inline TTEntry*
  probe (uint64_t key)
  {
    // key & (size - 1) is equal to (key % size) but faster
    return &table[key & (size - 1)];
  }

  // depth younger replace strategy
  inline void
  store (uint64_t key, int value, int depth, Bound bound, int best_square)
  {
    TTEntry* e = probe (key);
    // collusion happened or has better data, then replace old one.
    if (e->key != key || depth >= e->depth || age != e->age)
    {
      e->key = key;
      e->value = (int16_t)value;
      e->depth = (int16_t)depth;
      e->bound = (uint8_t)bound;
      e->best_move_square = (uint8_t)best_square;
      e->age = age;
    }
  }

  inline bool
  load (uint64_t key, TTEntry& out)
  {
    TTEntry* e = probe (key);
    if (e->key == key)
    {
      out = *e;
      return true;
    }
    return false;
  }

  inline void
  new_age ()
  {
    ++age;
  }
};

// int alpha_beta(Position pos, int depth, int alpha, int beta) {
// uint64_t key = zobrist_hash(pos);

//// 1. 查 TT
// TTEntry* entry = tt_probe(key);
// if (entry && entry->depth >= depth) {
// if (entry->flag == EXACT) return entry->score;
// if (entry->flag == LOWERBOUND && entry->score >= beta) return entry->score;
// if (entry->flag == UPPERBOUND && entry->score <= alpha) return entry->score;
//}

//// 2. 遞迴搜索子節點
// int bestScore = -INF;
// Move bestMove;
// for (Move m : generate_moves(pos)) {
// int score = -alpha_beta(apply(pos, m), depth - 1, -beta, -alpha);
// if (score > bestScore) {
// bestScore = score;
// bestMove = m;
//}
// if (bestScore >= beta) break; // β 剪枝
// alpha = std::max(alpha, bestScore);
//}

//// 3. 存入 TT
// int flag = (bestScore <= alpha) ? UPPERBOUND :
//(bestScore >= beta)  ? LOWERBOUND : EXACT;
// tt_store(key, depth, bestScore, flag, bestMove);

// return bestScore;
//}

struct TTState
{
  uint64_t black;
  uint64_t white;
  bool is_black_move;
};

int
gen_database (int move_num, const char* filename)
{
  uint64_t init_black = 0x0000000810000000ull; // 初始黑
  uint64_t init_white = 0x0000001008000000ull; // 初始白
  bool init_black_move = true;

  FILE* file = fopen (filename, "w");
  if (!file)
  {
    perror ("Cannot open file");
    return -1;
  }

  // 顯式棧
  std::stack<TTState> stk;
  stk.push ({ init_black, init_white, init_black_move });

  while (!stk.empty ())
  {
    TTState state = stk.top ();
    stk.pop ();

    uint64_t black = state.black;
    uint64_t white = state.white;
    bool is_black_move = state.is_black_move;

    uint64_t legal_moves;
    get_move (legal_moves,
              is_black_move ? black : white,
              is_black_move ? white : black);

    // 遍歷每個合法位置
    while (legal_moves)
    {
      uint64_t move = legal_moves & -legal_moves; // 取最右邊的 1
      legal_moves &= legal_moves - 1;             // 清掉這個 bit

      // 決定要移動的位置了，現在下該位置後，black, white 會發生什麼？

      uint64_t new_black = black;
      uint64_t new_white = white;

      if (is_black_move)
        new_black ^= move;
      else
        new_white ^= move;

      int num_discs = __builtin_popcountll (new_black | new_white);
      if (num_discs == move_num)
      {
        uint64_t h = hash_position (new_black, new_white, !is_black_move);
        DOUT << h << "\n";
        fwrite (&h, sizeof (h), 1, file);
      }
      else if (num_discs < move_num)
      {
        // 推入棧中，稍後再展開
        stk.push ({ new_black, new_white, !is_black_move });
      }
    }
  }

  fclose (file);
  return 0;
}

int
main ()
{
  // gen_database (10, "db10.txt");

  uint64_t init_black
      = 0b00000000000000000000000000000000000000000000000000000001ull; // 初始黑
  uint64_t init_white
      = 0b00000000000000000000000000000000000000000000000001111110ull; // 初始白
  uint64_t moveable_mask = 0;
  get_move (moveable_mask, init_black, init_white);

   DOUT << std::bitset<64> (moveable_mask) << "\n";

  array_1d_t<Move> move_1d;
  get_flipped_mask (move_1d, moveable_mask, init_black, init_white);

  for (auto& i : move_1d)
  {
    DOUT << i.square;
    DOUT_bitboard (i.flipped_mask);
  }
}

/*int
main ()
{
  zobrist_init ();

  uint64_t black = (1ull << (8 * 3 + 4)) | (1ull << (8 * 4 + 3));
  uint64_t white = (1ull << (8 * 3 + 3)) | (1ull << (8 * 4 + 4));
  bool is_black_move = true;

  uint64_t h = hash_position (black, white, is_black_move);
  std::cout << "Initial zobrist hash = " << std::hex << h << std::dec << "\n";

  // 建立 TT (16 MB)
  TT tt;
  tt.init (16 * 1024 * 1024);

  // 存一筆 (模擬搜尋結果)
  tt.store (h, 50, 5, Bound::EXACT, 19); // value=50, depth=5, best move=19

  // 嘗試拿 h 讀取
  TTEntry entry;
  if (tt.load (h, entry))
  {
    std::cout << "TT hit! value=" << entry.value << " depth=" << entry.depth
              << " best move square=" << (int)entry.best_move_square << "\n";
  }
  else
  {
    std::cout << "TT miss!\n";
  }

  // 模擬下一步 (黑走在 19，翻轉假設只有 27)
  uint64_t h2 = hash_after_move (h, is_black_move, 19, (1ull << 27));
  std::cout << "After move zobrist hash = " << std::hex << h2 << std::dec
            << "\n";

  return 0;
}*/
