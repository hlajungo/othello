#pragma once

#include <chrono>
#include <stack>
#include <unordered_set>

#include <hash.hpp>
#include <move.hpp>

template <typename Move_impl, typename Hash_impl>
class Database
{
public:
  /*
   * @brief generate database with `move_num` nonempty square.
   * batch_size version, from `compute -> file repate` became `compute -> buffer
   * -> file repeat`
   * @param file_prefix 輸出檔案前綴
   * @param move_num_begin 起始棋子數
   * @param move_num_end 結束棋子數
   * @param batch_size 一次性寫入的 bytes 數 (必須是 8 的倍數)
   * (because uint64_t is 8 bytes)
   */
  int
  gen_database (const std::string& db_path,
                const std::string& db_prefix,
                const int& move_num_begin,
                const int& move_num_end,
                const bool& is_batch = false,
                const size_t& batch_size = 8)
  {
    assert (move_num_begin <= 5);
    assert (move_num_begin <= move_num_end);
    assert (move_num_end <= 64);
    assert (batch_size >= 8);
    assert (batch_size % sizeof (uint64_t) == 0);

    // batch
    std::vector<uint64_t> buffer;
    size_t batch_count = batch_size / sizeof (uint64_t);
    if (is_batch == true)
    {
      buffer.reserve (batch_count);
    }

    for (int move_num = move_num_begin; move_num <= move_num_end; ++move_num)
    {
      auto time_start = std::chrono::high_resolution_clock::now ();

      // 初始棋盤
      Game_ctx game_ctx_init (
          0x0000000810000000ull, 0x0000001008000000ull, true);

      std_fs::path db_name
          = std_fs::path (db_path) / (db_prefix + std::to_string (move_num));

      FILE* file = fopen (db_name.c_str (), "wb");
      if (!file)
      {
        perror ("Cannot open file");
        return -1;
      }

      // 顯式棧
      std::stack<Game_ctx> stk;
      std::unordered_set<uint64_t> seen;

      stk.push (game_ctx_init);
      while (!stk.empty ())
      {
        Game_ctx cur_ctx = stk.top ();
        stk.pop ();
        // uint64_t black = state.black;
        // uint64_t white = state.white;
        // bool is_black_move = state.is_black_move;

        uint64_t moveable_mask = 0;
        this->move_impl.get_move (moveable_mask, cur_ctx);

        // check for is game playable
        if (moveable_mask == 0)
        {
          uint64_t moveable_mask_2 = 0;
          cur_ctx.is_black_move = !cur_ctx.is_black_move;
          this->move_impl.get_move (moveable_mask_2, cur_ctx);

          // opponent can't move too, game finish
          if (moveable_mask_2 == 0)
          {
            continue;
          }
          // opponent can move, let game going with opponent turn
          stk.push (cur_ctx);
          continue;
        }

        array_1d_t<Flip_ctx> move_1d;
        this->move_impl.get_flipped_mask (move_1d, moveable_mask, cur_ctx);

        // trying go through all move
        for (auto& move : move_1d)
        {
          // play this move
          Game_ctx new_ctx (
              cur_ctx.black, cur_ctx.white, !cur_ctx.is_black_move);
          uint64_t square_mask = 1ull << move.square;

          // update new ctx with black go
          if (cur_ctx.is_black_move)
          {
            new_ctx.black |= square_mask;
            new_ctx.black |= move.flip_mask;
            new_ctx.white &= ~move.flip_mask;
          }
          // with white go
          else
          {
            new_ctx.white |= square_mask;
            new_ctx.white |= move.flip_mask;
            new_ctx.black &= ~move.flip_mask;
          }

          int num_discs = __builtin_popcountll (new_ctx.black | new_ctx.white);
          if (num_discs == move_num)
          {
            uint64_t hash_num = 0;
            hash_impl.hash_position (hash_num, new_ctx);
            // DOUT_bitboard (new_black);
            // DOUT << hash << "\n";

            // 此處有可能出現以前重複過的
            // it return std::pair<iterator, bool>
            if (seen.insert (hash_num).second)
            {
              // no batch method
              if (is_batch == false)
              {
                fwrite (&hash_num, sizeof (hash_num), 1, file);
              }
              // batch version
              else
              {
                buffer.push_back (hash_num);
                if (buffer.size () >= batch_count)
                {
                  fwrite (
                      buffer.data (), sizeof (uint64_t), buffer.size (), file);
                  buffer.clear ();
                }
              }
            }
          }
          else if (num_discs < move_num)
          {
            // 推入棧中，稍後再展開
            stk.push (new_ctx);
          }
        }
      }
      fclose (file);
      auto time_end = std::chrono::high_resolution_clock::now ();
      std::chrono::duration<double> time_elapsed = time_end - time_start;
      std::cout << move_num << " bits use " << time_elapsed.count () << " s"
                << std::endl;
    }
    return 0;
  }

  /**
   * @brief 讀取二進位的 hash 檔案，並以十進制輸出
   * @param filename 檔案名稱 (例如 "db2.txt")
   */
  void
  read_db (const char* filename)
  {
    FILE* file = fopen (filename, "rb"); // "rb" = read binary
    if (!file)
    {
      perror ("Cannot open file");
      return;
    }

    uint64_t h;
    size_t count = 0;
    while (fread (&h, sizeof (h), 1, file) == 1)
    {
      std::cout << count << ": " << h << "\n";
      count++;
    }

    fclose (file);
  }

public:
  Database (Move_impl& move_impl, Hash_impl& hash_impl)
      : move_impl (move_impl), hash_impl (hash_impl)
  {
  }

  Move_impl& move_impl;
  Hash_impl& hash_impl;
};

