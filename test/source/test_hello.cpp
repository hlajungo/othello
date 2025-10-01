#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>
#include <doctest/doctest.h>
#include <fs_util.hpp>
#include <game.hpp>
#include <hash.hpp>
#include <move.hpp>
#include <generator.hpp>
// #include <iostream>
// #include <sstream>
// #include <streambuf>
/* @brief Test Position
 * It will test default function and helper functions
 */
TEST_CASE("Position") {
  uint64_t black= (get_mask(4, 3) | get_mask(3, 4));
  uint64_t white= (get_mask(3, 3) | get_mask(4, 4));
  /* Test constructor
   */
  Position pos({ { 4, 3 }, { 3, 4 } }, { { 3, 3 }, { 4, 4 } }, true);
  CHECK(pos.get_black() == black);
  CHECK(pos.get_white() == white);
  CHECK(pos.get_is_black_move() == true);
  /* Test copy constructor
   */
  Position pos2(pos);
  CHECK(pos2.get_black() == black);
  CHECK(pos2.get_white() == white);
  CHECK(pos2.get_is_black_move() == true);
  /* Test copy assignment
   */
  Position pos3= pos;
  CHECK(pos3.get_black() == black);
  CHECK(pos3.get_white() == white);
  CHECK(pos3.get_is_black_move() == true);
  /* Test helper function
   */
  CHECK(pos.get_player() == black);
  CHECK(pos.get_opponent() == white);
  CHECK(pos.get_empty() == ~(black | white));
  CHECK(pos.get_piece_num() == 4);
  Position pos4({ { 4, 3 }, { 3, 4 }, { 5, 5 } }, { { 3, 3 }, { 4, 4 } }, true);
  CHECK(pos.get_score() == 0);
  CHECK(pos4.get_score() == 1);
}
/*
 * @brief Test get_moveable_mask, get_flip_ctx_1d and flip
 * It will trying a pos, do a flip based on get_moveable_mask
 */
TEST_CASE("Flip_1") {
  Position pos(0b00000000000000000000000000000000000000000000000000000001ull, 0b00000000000000000000000000000000000000000000000001111110ull, true);
  Move_impl move_impl;
  uint64_t moveable_mask= move_impl.get_moveable_mask(pos);
  // the position black can play
  CHECK(moveable_mask == 0b10000000);
  array_1d_t<Flip_ctx> flip_ctx_1d= move_impl.get_flip_ctx_1d(pos, moveable_mask);
  CHECK(flip_ctx_1d.size() == 1);
  // the position black played
  CHECK(flip_ctx_1d[0].square == 7);
  // after play, what will be flipped
  CHECK(flip_ctx_1d[0].flip_mask == 0b01111110);
  move_impl.flip(pos, flip_ctx_1d[0]);
  CHECK(pos.black == 0b00000000000000000000000000000000000000000000000011111111ull);
  CHECK(pos.white == 0b00000000000000000000000000000000000000000000000000000000ull);
  CHECK(pos.is_black_move == false);
}
/*
 * @brief Test try_play
 * It will trying to play. When fail, return 1, when success, return 0, and pos is flipped.
 */
TEST_CASE("Flip_2") {
  Move_impl move_impl;
  Position pos(0b00000000000000000000000000000000000000000000000000000001ull, 0b00000000000000000000000000000000000000000000000001111110ull, true);
  CHECK(1 == move_impl.try_play(pos, 1));
  CHECK(0 == move_impl.try_play(pos, 7));
  CHECK(pos.black == 0b00000000000000000000000000000000000000000000000011111111ull);
  CHECK(pos.white == 0b00000000000000000000000000000000000000000000000000000000ull);
  CHECK(pos.is_black_move == false);
}
/*
 * @brief Testing hashing position
 *
 * The seed is 20250821, if seed changed, test will not pass.
 *
 */
TEST_CASE("Hash_ctx, Zobrist_hash_impl") {
  Hash_ctx hash_ctx(SEED);
  Zobrist_hash_impl<Hash_ctx> hash(hash_ctx);
  Position pos_init((1ull << (8 * 3 + 4)) | (1ull << (8 * 4 + 3)), (1ull << (8 * 3 + 3)) | (1ull << (8 * 4 + 4)), true);
  uint64_t hash_num= hash.hash_position(pos_init);
  // by seed 20250821
  CHECK(hash_num == 7872878270880865676);
}
/*
 * @test Test class `Pos_gen` function `gen_database`
 *
 * Trying to generate a database with all possible 5 piece, then compare it with
 * ans file `db5`.
 *
 */
/* The test is commented due to passing github action, the test require local answer set at "$REPO_PATH/test/fixture/piece_db/db*"
 */
#if 0
TEST_CASE ("Pos_gen")
{
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash_impl (hash_ctx);
  Move_impl move_impl;
  Pos_gen<Move_impl, Zobrist_hash_impl<Hash_ctx> > db (move_impl, hash_impl);
  // test dir
  std_fs::path test_dir = std_fs::path (__FILE__).parent_path ().parent_path ();
  // ans dir (fixture dir)
  std_fs::path ans_dir = test_dir / "fixture" / "piece_db";
  // db output dir
  std_fs::path db_path = test_dir / "test_piece_db";
  fs_util.try_mkdir (db_path);
  auto ret = db.gen_database (db_path.string (), "db", 5, 5, false);
  REQUIRE (ret == 0);
  CHECK (fs_util.is_file_equal ((ans_dir / "db5").string (),
                                (db_path / "db5").string ()));
}
#endif
/*
 * @test Test class `Game_impl` function `is_legal`
 * Using legal bitboard to test database generated db is working
 */
/* The test is commented due to passing github action, the test require local answer set at "$REPO_PATH/test/fixture/piece_db/db*"
 */
#if 0
TEST_CASE("Game_impl is_legal") {
  // test root dir
  std_fs::path test_dir = std_fs::path (__FILE__).parent_path ().parent_path ();
  // ans dir (fixture dir)
  std_fs::path ans_dir = test_dir / "fixture" / "piece_db";
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash_impl (hash_ctx);
  // legal position with 5 piece, it should use db5
  Position pos_piece_5 (
      0b0000000000000000000000000000100000011100000000000000000000000000,
      0b0000000000000000000000000001000000000000000000000000000000000000,
      false);
  using Game = Game_impl<Position, Zobrist_hash_impl<Hash_ctx> >;
  Game game (pos_piece_5, hash_impl);
  auto legality = game.is_legal (ans_dir.string (), "db");
  CHECK (legality == true);
  // legal position with 15 piece, it should use db15
  Position pos_piece_15 (
      0b0000000000100000001100000001100000111100001000000000000000000000,
      0b0000000000000000010000000010010000000000000110000000000000000000,
      false);
  game.pos = pos_piece_15;
  legality = game.is_legal (ans_dir.string (), "db");
  CHECK (legality == true);
}
#endif
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
