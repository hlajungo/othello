#pragma once
#include <Position.hpp>
#include <const.h>
#include <dbg.hpp>
#include <type.hpp>
struct Flip_ctx {
  Flip_ctx(): flip_mask(0), square(0) {}
  Flip_ctx(uint64_t flip_mask, int square): flip_mask(flip_mask), square(square) {}
  Flip_ctx(Flip_ctx&& o) {
    flip_mask= o.flip_mask;
    square= o.square;
  }
  uint64_t flip_mask; /*flipped squares*/
  uint8_t square;     /*square played,[0-63]*/
};
/*
 * @brief Bitboard move utility, all thing related with playing chess in here.
 */
class Move_impl {
public:
  /*
   * 位運算帶來並行性，它能夠自動的檢查任何像是 ".XXXO" 的匹配，並累加 X
   * 會不斷重複直到，向右移動不帶來任何重疊
   *
   * 你不能在算 moveable_mask 時，一起算單獨的 flip_mask 和
   * square。這是由於位運算並行性，你得在算完後，單獨的計算。
   */
  uint64_t get_moveable_mask(const Position& pos) {
    /*
     * dir
     * 右 = shift << 1
     * 左 = shift >> 1
     * 上 = shift >> 8
     * 下 = shift << 8
     * 右上 = shift >> 7
     * 左上 = shift >> 9
     * 右下 = shift << 9
     * 左下 = shift << 7
     */
    constexpr const int dir_1d[]= { 1, -1, 8, -8, 9, 7, -7, -9 };
    constexpr const uint64_t mask_1d[]= { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111, MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };
    uint64_t moveable_mask= 0;
    uint64_t player= pos.get_player();
    uint64_t opponent= pos.get_opponent();
    uint64_t empty= pos.get_empty();
    for(int dir= 0; dir < 8; ++dir) {
      const int shift= dir_1d[dir];
      const uint64_t mask= mask_1d[dir];
      // 嘗試走一格, 尋找能壓到對手的格子
      uint64_t tmp= opponent & mask & (shift > 0 ? (player << shift) : (player >> -shift));
      // 壓到格子了
      while(tmp) {
        // 嘗試走一格，是空的，就加入解
        uint64_t answer_mask= empty & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
        if(answer_mask) { moveable_mask|= answer_mask; }
        // 繼續走
        tmp= opponent & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
      }
    }
    return moveable_mask;
  }
  /*
   * @brief At position play square
   * @return Flip mask of played position
   */
  Flip_ctx get_flip_ctx(const Position& pos, const int square) {
    Flip_ctx flip_ctx(0, square);
    uint64_t player= pos.get_player();
    uint64_t opponent= pos.get_opponent();
    // right, left, down, up
    constexpr const int dir_1d[]= { 1, -1, 8, -8, 9, 7, -7, -9 };
    constexpr const uint64_t mask_1d[]= { MASK_11111110, MASK_01111111, MASK_11111111, MASK_11111111, MASK_11111110, MASK_01111111, MASK_11111110, MASK_01111111 };
    uint64_t move_mask= 1ull << square;
    //std::cout << move_mask << "\n";
    for(int dir= 0; dir < 8; ++dir) {
      uint64_t mask= mask_1d[dir];
      int shift= dir_1d[dir];
      uint64_t captured= 0;
      // 嘗試壓到對手
      uint64_t tmp= opponent & mask & (shift > 0 ? (move_mask << shift) : (move_mask >> -shift));
      //std::cout << "B " << tmp << "\n";
      // 壓到對手
      while(tmp) {
        // 對手加到解答
        captured|= tmp;
        // 嘗試走一步壓到自己
        uint64_t ans= player & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
        if(ans) {
          flip_ctx.flip_mask|= captured;
          break;
        }
        // 繼續壓到對手
        tmp= opponent & mask & (shift > 0 ? (tmp << shift) : (tmp >> -shift));
        //std::cout << "A " << tmp << "\n";
      }
      //std::cout << player << " " << tmp << "\n";
      // 壓到自己，完成"自己對手自己"夾心
      // if(tmp & player) { flip_ctx.flip_mask|= captured; std::cout << "HIII" <<flip_ctx.flip_mask << "\n"; }
    }
    return flip_ctx;
  }
  /*
   * @brief Play every bit in moveable_mask, and get its Flip_ctx
   */
  array_1d_t<Flip_ctx> get_flip_ctx_1d(const Position& pos, const uint64_t moveable_mask) {
    uint64_t moves= moveable_mask;
    array_1d_t<Flip_ctx> flip_ctx_1d;
    while(moves) {
      // lowest 1 offst
      int square= __builtin_ctzll(moves);
      flip_ctx_1d.push_back(get_flip_ctx(pos, square));
      // clear lowest 1
      moves&= moves - 1;
    }
    return flip_ctx_1d;
  }
  /*
   * @brief Using flip_ctx flip the pos
   */
  void flip(Position& pos, Flip_ctx& flip_ctx) {
    if(pos.is_black_move) {
      pos.black|= 1ull << flip_ctx.square;
      pos.black|= flip_ctx.flip_mask;
      pos.white&= ~flip_ctx.flip_mask;
    } else {
      pos.white|= 1ull << flip_ctx.square;
      pos.white|= flip_ctx.flip_mask;
      pos.black&= ~flip_ctx.flip_mask;
    }
    pos.is_black_move= !pos.is_black_move;
  }
  void play(Position& pos, const int square) {
    auto flip_ctx= get_flip_ctx(pos, square);
    if(flip_ctx.flip_mask != 0) flip(pos, flip_ctx);
  }
  int try_play(Position& pos, const int square) {
    uint64_t moveable_mask= get_moveable_mask(pos);
    // pos is moveable
    if(moveable_mask & (1ull << square)) {
      play(pos, square);
      return 0;
    }
    // pos is invaild
    else {
      std::cerr << "Error: A invaild try_play\n";
      return 1;
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
