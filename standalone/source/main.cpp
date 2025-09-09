#include <fs_util.hpp>
#include <game.hpp>
#include <hash.hpp>
#include <move.hpp>
#include <type.hpp>

/*
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

void
fn (bool is_legal, const std::string& db_path, const Game_ctx& game_ctx);

int
main ()
{
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash_impl (hash_ctx);
  // 局面上有 5 子
  Game_ctx game_ctx_5 (
      0b0000000000000000000000000000100000011100000000000000000000000000,
      0b0000000000000000000000000001000000000000000000000000000000000000,
      false);

  Game_impl<Game_ctx, Zobrist_hash_impl<Hash_ctx> > game_impl (game_ctx_5,
                                                               hash_impl);
  // 16 子(包含)以下使用查 db
  // db 路徑, db prefix, 16 子解答
  auto legality = game_impl.is_legal ("./db", "db", 16);
  if (legality == true)
  {
    DOUT << "legal\n";
  }
  else
  {
    DOUT << "illegal\n";
  }

  // 局面上有 15 子
  Game_ctx game_ctx_15 (
      0b0000000000100000001100000001100000111100001000000000000000000000,
      0b0000000000000000010000000010010000000000000110000000000000000000,
      false);

  game_impl.game_ctx = game_ctx_15;
  legality = game_impl.is_legal ("./db", "db", 16);
  if (legality == true)
  {
    DOUT << "legal\n";
  }
  else
  {
    DOUT << "illegal\n";
  }
}

// int
// main ()
//{
// zobrist_init ();

// uint64_t black = (1ull << (8 * 3 + 4)) | (1ull << (8 * 4 + 3));
// uint64_t white = (1ull << (8 * 3 + 3)) | (1ull << (8 * 4 + 4));
// bool is_black_move = true;

// uint64_t hash = 0;
// hash_position (hash, black, white, is_black_move);
// std::cout << "Initial zobrist hash = " << std::hex << hash << std::dec <<
// "\n";

//// 建立 TT (16 MB)
// TT tt;
// tt.init (16 * 1024 * 1024);

//// 存一筆 (模擬搜尋結果)
// tt.store (hash, 50, 5, Bound::EXACT, 19); // value=50, depth=5, best
// move=19

//// 嘗試拿 hash 讀取
// TTEntry entry;
// if (tt.load (hash, entry))
//{
// std::cout << "TT hit! value=" << entry.value << " depth=" << entry.depth
//<< " best move square=" << (int)entry.best_move_square << "\n";
//}
// else
//{
// std::cout << "TT miss!\n";
//}

//// 模擬下一步 (黑走在 19，翻轉假設只有 27)
// uint64_t h2 = hash_after_move (hash, is_black_move, 19, (1ull << 27));
// std::cout << "After move zobrist hash = " << std::hex << h2 << std::dec
//<< "\n";

// return 0;
//}
