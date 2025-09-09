#pragma once
#include <const.h>
#include <dbg.hpp>
#include <game.hpp>
#include <type.hpp>

class Flip_ctx
{
public:
  uint64_t flip_mask; /*flipped squares*/
  int square;         /*square played*/
};

class Move_impl
{
public:
  /*方向向量：
  右 = shift << 1
  左 = shift >> 1
  上 = shift >> 8
  下 = shift << 8
  右上 = shift >> 7
  左上 = shift >> 9
  右下 = shift << 9
  左下 = shift << 7
  */

  /*
   * get_moveable_mask 演算法說明
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
   *
   * 你不能在算 moveable_mask 時，一起算單獨的 flip_mask 和
   * square。這是由於位運算並行性，你得在算完後，單獨的計算。
   */

  void
  get_moveable_mask (uint64_t& moveable_mask,
            const uint64_t& player,
            const uint64_t& opponent)
  {
    // right, left, down, up, right down, left down, right up, left up
    constexpr int dir_1d[] = { 1, -1, 8, -8, 9, 7, -7, -9 };
    constexpr uint64_t mask_1d[]
        = { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111,
            MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };

    uint64_t empty = ~(player | opponent);
    for (int dir = 0; dir < 8; ++dir)
    {
      const int& shift = dir_1d[dir];
      const uint64_t& mask = mask_1d[dir];

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
  get_flip_mask (uint64_t& flip_mask,
                 const int square,
                 const uint64_t player,
                 const uint64_t opponent)
  {
    // right, left, down, up
    constexpr const int dir_1d[] = { 1, -1, 8, -8, 9, 7, -7, -9 };
    constexpr const uint64_t mask_1d[]
        = { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111,
            MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };

    uint64_t move_mask = 1ull << square;

    for (int dir = 0; dir < 8; ++dir)
    {
      uint64_t mask = mask_1d[dir];
      int shift = dir_1d[dir];

      uint64_t captured = 0;
      // 嘗試走一步
      uint64_t tmp = (shift > 0 ? (move_mask << shift) : (move_mask >> -shift));

      // 壓到對手
      while (tmp & mask & opponent)
      {
        captured |= tmp;
        tmp = (shift > 0 ? (tmp << shift) : (tmp >> -shift));
      }
      // 壓到自己，完成"自己對手自己"夾心
      if (tmp & mask & player)
      {
        flip_mask |= captured;
      }
    }
  }
  void
  get_flip_mask_1d (array_1d_t<Flip_ctx>& flip_1d,
                    const uint64_t& moveable_mask,
                    const uint64_t& player,
                    const uint64_t& opponent)
  {
    uint64_t moves = moveable_mask;
    while (moves)
    {
      // lowest 1 offset
      int square = __builtin_ctzll (moves);
      uint64_t flip_mask = 0;
      get_flip_mask (flip_mask, square, player, opponent);

      moves &= moves - 1; // clear lowest 1

      flip_1d.push_back ({ flip_mask, square });
    }
  }

  /*
   * @brief Wrapper for Game_ctx
   */
  void
  get_moveable_mask (uint64_t& moveable_mask, const Game_ctx& game_ctx)
  {
    if (game_ctx.is_black_move == true)
    {
      get_moveable_mask (moveable_mask, game_ctx.black, game_ctx.white);
    }
    else
    {
      get_moveable_mask (moveable_mask, game_ctx.white, game_ctx.black);
    }
  }

  /*
   * @brief Get flip_mask for every positition in flip_1d
   */
  void
  get_flip_mask_1d (array_1d_t<Flip_ctx>& flip_1d,
                    const uint64_t& moveable_mask,
                    const Game_ctx& game_ctx)
  {
    if (game_ctx.is_black_move == true)
    {
      get_flip_mask_1d (flip_1d, moveable_mask, game_ctx.black, game_ctx.white);
    }
    else
    {
      get_flip_mask_1d (flip_1d, moveable_mask, game_ctx.white, game_ctx.black);
    }
  }

  /*
   * 找出所有連續三子組合
   * */
  // void
  // get_removeable (uint64_t& removeable_mask,
  // const uint64_t& player,
  // const uint64_t& opponent)
  //{
  //// right, left, down, up, right down, left down, right up, left up
  // constexpr int dir_1d[] = { 1, -1, 8, -8, 9, 7, -7, -9 };
  // constexpr uint64_t mask_1d[]
  //= { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111,
  // MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };

  // uint64_t empty = ~(player | opponent);
  // for (int dir = 0; dir < 8; ++dir)
  //{
  // const int& shift = dir_1d[dir];
  // const uint64_t& mask = mask_1d[dir];

  //// 嘗試走一格, 尋找能壓到自己的格子
  // uint64_t tmp = opponent & mask
  //& (shift > 0 ? (player << shift) : (player >> -shift));
  //// 壓到格子了
  // while (tmp)
  //{
  //// 嘗試走一格，是空的，就加入解
  // uint64_t moveable
  //= empty & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
  // if (moveable)
  //{
  // moveable_mask |= moveable;
  //}
  //// 繼續走
  // tmp = opponent & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
  //}
  //}
};
