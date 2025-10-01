#pragma once
#include <observer.hpp>
#include <type.hpp>

/* @brief This part of comment is for a departed feature for Position, the cache system for calc get_empty etc
 */
// template <typename Position>
// class Game_cache : public Observer
//{
// public:
// explicit Game_cache (const Position& game_ctx)
//: game_ctx (game_ctx), need_update (true) [>Need update at begin<]
//{
//}
// void
// update () override
//{
// need_update = true;
//}
// uint64_t
// get_empty ()
//{
// if (need_update)
//{
// need_update = false;
// empty_mask = ~(game_ctx.black | game_ctx.white);
//}
// return empty_mask;
//}
// int
// get_piece_num ()
//{
// if (need_update)
//{
// need_update = false;
// piece_num = __builtin_popcountll (game_ctx.black | game_ctx.white);
//}
// return piece_num;
//}
// private:
// const Position& game_ctx; // Strongly bound resource
// bool need_update;
////  cached data
// uint64_t empty_mask;
// int piece_num;
//};
inline uint64_t get_mask(int x, int y) { return (1ull << (8 * y + x)); }
class Position {
public:
  /* Variable
   */
  uint64_t black;     // bitboard for black
  uint64_t white;     // bitboard for white
  bool is_black_move; // the one to move
public:
  /* Class func
   */
  /* @brief Using readable pair to init black/white
   */
  Position(array_1d_t<std::pair<int, int>> black, array_1d_t<std::pair<int, int>> white, bool is_black_move= true): black(0), white(0), is_black_move(is_black_move) {
    for(auto& piece: black) {
      auto x= piece.first;
      auto y= piece.second;
      this->black|= get_mask(x, y);
    }
    for(auto& piece: white) {
      auto x= piece.first;
      auto y= piece.second;
      this->white|= get_mask(x, y);
    }
  }
  /* @brief Using given data init Position
   */
  Position(const uint64_t black, const uint64_t white, const bool is_black_move= true): black(black), white(white), is_black_move(is_black_move) {}
  /* Copy constructor and assignment
   */
  Position(const Position& o): black(o.black), white(o.white), is_black_move(o.is_black_move) {}
  Position& operator=(const Position& o) {
    this->black= o.black;
    this->white= o.white;
    this->is_black_move= o.is_black_move;
    return *this;
  }
  ~Position() {}
  /* Setter Getter
   */
  void set_black(const uint64_t black) { this->black= black; }
  uint64_t get_black() const { return black; }
  void set_white(const uint64_t white) { this->white= white; }
  uint64_t get_white() const { return white; }
  void set_is_black_move(const uint64_t is_black_move) { this->is_black_move= is_black_move; }
  bool get_is_black_move() const { return is_black_move; }
  /* Other helper
   */
  uint64_t get_empty() const { return ~(black | white); }
  uint64_t get_player() const { return is_black_move ? black : white; }
  uint64_t get_opponent() const { return is_black_move ? white : black; }
  int get_piece_num() const { return __builtin_popcountll(black | white); }
  int get_score() const { return __builtin_popcountll(get_player()) - __builtin_popcountll(get_opponent()); }
};
