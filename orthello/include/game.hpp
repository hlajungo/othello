#pragma once

#include <algorithm>
#include <cstdint>
#include <system_error>

#include <dbg.hpp>
#include <fs_util.hpp>
#include <type.hpp>

class Observer
{
public:
  virtual ~Observer () = default;
  virtual void
  update ()
      = 0;
};

template <typename Game_ctx>
class Game_cache : public Observer
{
public:
  explicit Game_cache (const Game_ctx& game_ctx) : game_ctx (game_ctx) {}

  void
  update () override
  {
    need_update = true;
  }

  uint64_t
  get_empty ()
  {
    if (need_update)
    {
      need_update = false;
      empty_mask = ~(game_ctx.black | game_ctx.white);
    }
    return empty_mask;
  }

  int
  get_piece_num ()
  {
    if (need_update)
    {
      need_update = false;
      piece_num = __builtin_popcountll (game_ctx.black | game_ctx.white);
    }
    return piece_num;
  }

private:
  const Game_ctx& game_ctx; // Strongly bound resource
  bool need_update = true;

  // data to cached
  uint64_t empty_mask;
  int piece_num;
};

class Game_ctx
{
private:
  // observer impl
  std::vector<Observer*> observer_1d;

  void
  notify ()
  {
    for (auto* obs : observer_1d)
    {
      obs->update ();
    }
  }

public:
  void
  add_observer (Observer* obs)
  {
    observer_1d.push_back (obs);
  }

public:
  // variable and setter
  Game_ctx (const uint64_t black,
            const uint64_t white,
            const bool is_black_move)
      : black (black), white (white), is_black_move (is_black_move){};

  uint64_t black; // bitboard for black
  uint64_t white; // bitboard for white
  bool is_black_move; // the one to move
                      //
  void
  set_black (const uint64_t black)
  {
    this->black = black;
    notify ();
  }

  void
  set_white (const uint64_t white)
  {
    this->white = white;
    notify ();
  }

  void
  set_is_black_move (const uint64_t is_black_move)
  {
    this->is_black_move = is_black_move;
    notify ();
  }

  Game_cache<Game_ctx>* cache_ptr = nullptr;
};

template <typename Game_ctx, typename Hash_impl>
class Game_impl
{
public:
  bool
  is_legal (const std::string& db_path,
            const std::string& db_prefix,
            const int& check_db_threshold)
  {
    const auto& black = game_ctx.black;
    const auto& white = game_ctx.white;
    // const auto& is_black_move = game_ctx.is_black_move;

    uint64_t overlay = black & white;
    if (overlay != 0)
    {
      DOUT << "not legal, overlay\n";
      return false;
    }

    int bit_num = __builtin_popcountll (black) + __builtin_popcountll (white);
    if (bit_num < 4)
    {
      DOUT << "not legal, at least has 4 on board\n";
      return false;
    }

    int num_discs = __builtin_popcountll (black | white);
    // DOUT << num_discs << "\n";
    //  直接查 db
    if (num_discs <= check_db_threshold)
    {
      array_1d_t<std::string> file_1d;
      get_file_1d (file_1d, db_path, db_prefix);
      // DOUT_array_1d(file_1d);

      std_fs::path filename
          = std_fs::path (db_path) / (db_prefix + std::to_string (num_discs));

      auto it = std::find (file_1d.begin (), file_1d.end (), filename);
      // found that file
      if (it != file_1d.end ())
      {
        const auto& filename = *it;
        DOUT << "Found " << filename << "\n";
        uint64_t hash = 0;
        hash_impl.hash_position (hash, game_ctx);
        // DOUT << "hash = " << hash << "\n";

        const auto found = is_in_file (filename, hash);
        if (found)
        {
          return true;
        }
        else
        {
          DOUT << "bitboard not in db\n";
          return false;
        }
      }
      else
      {
        std::error_code ec (ENOENT, std::generic_category ());
        std::cerr << "That db does not exist, but it should exist: "
                  << ec.message () << "\n";
      }
    }

    // try 回溯分析

    return false;
  }


  bool
  is_finish()
  {

  }

public:
  Game_impl (Game_ctx& game_ctx, Hash_impl& hash_impl)
      : game_ctx (game_ctx), hash_impl (hash_impl)
  {
  }
  Game_ctx& game_ctx;
  Hash_impl& hash_impl;
};

template <typename Game_impl, typename Hash_impl>
class Game
{
public:
  Game (Game_impl& g, Hash_impl& h) : game (g), hash (h) {}
  Game_impl& game;
  Hash_impl& hash;
};

// 方向向量 idx is [左右][上下]
constexpr int DIRS[8][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 },  { 0, -1 },
                             { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };

// 簡單輔助：取得 bitboard 上第 (r,c) 是否有子
inline bool
has_bit (uint64_t bb, int r, int c)
{
  return bb & (1ULL << (r * 8 + c));
}

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
