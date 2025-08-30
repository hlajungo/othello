#pragma once

#include <algorithm>
#include <cstdint>
#include <system_error>

#include <dbg.hpp>
#include <fs_util.hpp>
#include <type.hpp>

class Game_ctx
{
public:
  uint64_t black;     // bitboard for black
  uint64_t white;     // bitboard for white
  bool is_black_move; // the one to move
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

    // try

    return false;
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
