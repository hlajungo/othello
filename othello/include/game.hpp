#pragma once
#include <Position.hpp>
#include <algorithm>
#include <cstdint>
#include <dbg.hpp>
#include <fs_util.hpp>
#include <move.hpp>
#include <type.hpp>
template<typename Position, typename Hash_impl> class Game_impl {
public:
  Game_impl(Position& pos, Hash_impl& hash_impl): pos(pos), hash_impl(hash_impl) {}
  Position& pos;
  Hash_impl& hash_impl;
public:
  std::tuple<bool, std::string> get_db(const std::string& db_path, const std::string& db_prefix, int n) {
    // Get all file
    auto file_1d= fs_util.get_file_1d(db_path, db_prefix);
    // Get target file
    std_fs::path filename= std_fs::path(db_path) / (db_prefix + std::to_string(n));
    auto it= std::find(file_1d.begin(), file_1d.end(), filename);
    if(it != file_1d.end())
      return std::tuple<bool, std::string>(true, filename);
    else
      return std::tuple<bool, std::string>(false, "");
  }
  bool try_db(const std::string& db_path, const std::string& db_prefix) {
    int piece_num= pos.get_piece_num();
    auto [found_db, filename]= get_db(db_path, db_prefix, piece_num);
    if(found_db) {
      DOUT << "Found db: " << filename << "\n";
      uint64_t hash= hash_impl.hash_position(pos);
      const auto found= fs_util.is_in_file(filename, hash);
      if(found) {
        return true;
      } else {
        // std::cout << "Not in db\n";
        return false;
      }
    } else {
      std::cout << "Doesn't found db, using other method\n";
      return false;
    }
  }
  bool is_legal(const std::string& db_path, const std::string& db_prefix) {
    const auto& black= pos.black;
    const auto& white= pos.white;
    // Check overlay
    uint64_t overlay= black & white;
    if(overlay != 0) { std::cerr << "not legal, overlay\n"; }
    int piece_num= pos.get_piece_num();
    // Check piece
    if(piece_num < 4) { std::cerr << "not legal, at least has 4 on board\n"; }
    /* Check db */
    auto stat= try_db(db_path, db_prefix);
    if(stat) {
      std::cout << "Db found pos\n";
      return true;
    }
    // try 回溯分析
    return false;
  }
  // bool is_end() {}
  void try_play(Position& pos, const int square) {
    uint64_t moveable_mask= 0;
    get_moveable_mask(moveable_mask, pos);
    // pos is vaild
    if(moveable_mask & (1ull << square)) {
      Flip_ctx flip_ctx;
      get_flip_ctx(flip_ctx, pos, square);
      if(flip_ctx.flip_mask != 0) { flip(pos, flip_ctx); }
    }
    // pos is invaild
    else {
      std::cerr << "error: A invaild play played";
    }
  }
};
template<typename Game_impl, typename Hash_impl> class Game {
public:
  Game(Game_impl& g, Hash_impl& h): game(g), hash(h) {}
  Game_impl& game;
  Hash_impl& hash;
};
// 方向向量 idx is [左右][上下]
// constexpr int DIRS[8][2]= { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
// 簡單輔助：取得 bitboard 上第 (r,c) 是否有子
inline bool has_bit(uint64_t bb, int r, int c) { return bb & (1ULL << (r * 8 + c)); }
//// 核心：找出「可能的最後一步」
// void
// get_removeable (uint64_t& removeable_mask,
// const uint64_t& player,
// const uint64_t& opponent)
//{
// removeable_mask = 0ULL;
//// 掃描棋盤 (8x8)
// for (int r = 0; r < 8; r++)
//{
// for (int c = 0; c < 8; c++)
//{
// int idx = r * 8 + c;
//// 只處理玩家的子
// if (!(player & (1ULL << idx)))
// continue;
// bool could_be_last = false;
//// 嘗試 8 個方向，看這子是否可能是最後放下的
// for (auto& d : DIRS)
//{
// int dr = d[0], dc = d[1];
// int rr = r + dr, cc = c + dc;
// bool has_opponent_between = false;
//// 一路延伸
// while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8)
//{
// int next_idx = rr * 8 + cc;
// if (opponent & (1ULL << next_idx))
//{
// has_opponent_between = true;
//}
// else if (player & (1ULL << next_idx))
//{
//// 遇到自己 → 如果中間有對手，就合理
// if (has_opponent_between)
//{
// could_be_last = true;
//}
// break;
//}
// else
//{
//// 空格 → 不可能
// break;
//}
// rr += dr;
// cc += dc;
//}
//}
// if (could_be_last)
//{
// removeable_mask |= (1ULL << idx);
//}
//}
//}
//}
