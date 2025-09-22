#pragma once
#include <iostream>
#include <observer.hpp>
#include <type.hpp>

template <typename Game_ctx>
class Game_cache : public Observer
{
public:
  explicit Game_cache (const Game_ctx& game_ctx)
      : game_ctx (game_ctx), need_update (true) /*Need update at begin*/
  {
  }

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
  bool need_update;

  //  cached data
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
            const bool is_black_move,
            bool set_cache = true)
      : black (black), white (white), is_black_move (is_black_move),
        game_cache (nullptr)
  {
    if (set_cache && game_cache == nullptr)
    {
      game_cache = new Game_cache<Game_ctx> (*this);
    }
  }
  ~Game_ctx ()
  {
    delete game_cache;
  }

  uint64_t black;                   // bitboard for black
  uint64_t white;                   // bitboard for white
  bool is_black_move;               // the one to move
  Game_cache<Game_ctx>* game_cache; // ptr to game_cache

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

  Game_cache<Game_ctx>*
  get_game_cache () const
  {
    if (!game_cache)
    {
      throw std::runtime_error ("Trying to access not init game_cache");
    }
    return game_cache;
  }
};
