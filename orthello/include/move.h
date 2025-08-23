#pragma once

#include <const.h>
#include <type.hpp>
#include <bitset>

#include <dbg.hpp>

typedef struct Move
{
  uint64_t flipped_mask; /*flipped squares*/
  int square;            /*square played*/
} Move;


void
get_move (uint64_t& moveable_mask,
          const uint64_t& player,
          const uint64_t& opponent);


void
get_flipped_mask (array_1d_t<Move>& move_1d,
                  const uint64_t& moveable_mask,
                  const uint64_t& player,
                  const uint64_t& opponent);

