#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include <doctest.h>

#include <doctest/doctest.h>
#include <hello.h>

#include <move.h>

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
 * @test Test get_move and get_flipped_mask
 *
 * black = X
 * white = O
 * board = -OOOOOOX
 *
 * goal:
 * get_move should return
 * moveable_mask = X-------
 *
 * get_flipped_mask should return
 * square = 7 (the played square in seq 76543210)
 * flipped_mask = -XXXXXXX-
 */
TEST_CASE ("get_move get_flipped_mask")
{
  uint64_t init_black
      = 0b00000000000000000000000000000000000000000000000000000001ull;
  uint64_t init_white
      = 0b00000000000000000000000000000000000000000000000001111110ull;

  uint64_t moveable_mask = 0;
  get_move (moveable_mask, init_black, init_white);

  CHECK (moveable_mask == 0b10000000);

  array_1d_t<Move> move_1d;
  get_flipped_mask (move_1d, moveable_mask, init_black, init_white);

  CHECK (move_1d[0].square == 7);
  CHECK (move_1d[0].flipped_mask == 0b01111110);
}
