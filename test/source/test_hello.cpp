#include <filesystem>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>

#include <doctest/doctest.h>

#include <game.hpp>
#include <hash.hpp>
#include <move.hpp>

#include <database.hpp>

// #include <iostream>
// #include <sstream>
// #include <streambuf>

/*
TEST_CASE ("Hello")
{
  // 儲存原始 cout buffer
  std::streambuf* old_buf = std::cout.rdbuf ();

  // 準備攔截的輸出流
  std::ostringstream captured_output;
  std::cout.rdbuf (captured_output.rdbuf ()); // 將 cout 輸出轉向 stringstream

  // 呼叫被測函數
  hello_from_template ();

  // 還原 cout，避免後續攔截
  std::cout.rdbuf (old_buf);

  // 比對輸出
  CHECK (captured_output.str () == "Hello from template!\n");
}
*/

/*
 * @test Test class `Move` function `get_move` and `get_flipped_mask`
 *
 * Checking where can black go, and if black go, what will be flipped.
 */
TEST_CASE ("Move")
{
  Move_impl move_impl;

  Game_ctx game_ctx_init (
      0b00000000000000000000000000000000000000000000000000000001ull,
      0b00000000000000000000000000000000000000000000000001111110ull,
      true);

  uint64_t moveable_mask = 0;
  move_impl.get_move (moveable_mask, game_ctx_init);

  // the position black can play
  CHECK (moveable_mask == 0b10000000);

  array_1d_t<Flip_ctx> move_1d;
  move_impl.get_flipped_mask (move_1d, moveable_mask, game_ctx_init);

  // the position black played
  CHECK (move_1d[0].square == 7);
  // after play, what will be flipped
  CHECK (move_1d[0].flip_mask == 0b01111110);
}

/*
 * @test Test class `Hash_ctx` constructor and class `Zobrist_hash_impl`
 * function `hash_position`.
 *
 * The seed is 20250821, if seed changed, test will not pass.
 *
 */
TEST_CASE ("Hash_ctx, Zobrist_hash_impl")
{
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash (hash_ctx);

  Game_ctx game_ctx_init ((1ull << (8 * 3 + 4)) | (1ull << (8 * 4 + 3)),
                          (1ull << (8 * 3 + 3)) | (1ull << (8 * 4 + 4)),
                          true);

  uint64_t hash_num = 0;
  hash.hash_position (hash_num, game_ctx_init);

  // hash number by seed 20250821
  CHECK (hash_num == 7872878270880865676);
}

/*
 * @test Test class `Database` function `gen_database`
 *
 * Trying to generate a database with all possible 5 piece, then compare it with
 * ans file `db5`.
 *
 */

TEST_CASE ("Database")
{
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash_impl (hash_ctx);

  Move_impl move_impl;
  Database<Move_impl, Zobrist_hash_impl<Hash_ctx> > db (move_impl, hash_impl);

  // test dir
  std_fs::path test_dir = std_fs::path (__FILE__).parent_path ().parent_path ();

  // ans dir (fixture dir)
  std_fs::path ans_dir = test_dir / "fixture" / "piece_db";

  // db output dir
  std_fs::path db_path = test_dir / "test_piece_db";
  try_create_dir (db_path);

  auto ret = db.gen_database (db_path.string (), "db", 5, 5, false);
  REQUIRE (ret == 0);

  CHECK (
      is_file_equal ((ans_dir / "db5").string (), (db_path / "db5").string ()));
}

/*
 * @test Test class `Game_impl` function `is_legal`
 * Using legal bitboard to test database generated db is working
 */

TEST_CASE ("Game_impl is_legal")
{
  // test dir
  std_fs::path test_dir = std_fs::path (__FILE__).parent_path ().parent_path ();

  // ans dir (fixture dir)
  std_fs::path ans_dir = test_dir / "fixture" / "piece_db";

  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash_impl (hash_ctx);

  // legal position with 5 piece, it should use db5
  Game_ctx game_ctx_5 (
      0b0000000000000000000000000000100000011100000000000000000000000000,
      0b0000000000000000000000000001000000000000000000000000000000000000,
      false);

  Game_impl<Game_ctx, Zobrist_hash_impl<Hash_ctx> > game_impl (game_ctx_5,
                                                               hash_impl);
  auto legality = game_impl.is_legal (ans_dir.string(), "db", 16);
  CHECK (legality == true);

  // legal position with 15 piece, it should use db15
  Game_ctx game_ctx_15 (
      0b0000000000100000001100000001100000111100001000000000000000000000,
      0b0000000000000000010000000010010000000000000110000000000000000000,
      false);

  game_impl.game_ctx = game_ctx_15;
  legality = game_impl.is_legal (ans_dir.string (), "db", 16);
  CHECK (legality == true);
}

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
