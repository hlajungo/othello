#include <database.hpp>
#include <hash.hpp>
#include <move.hpp>

int
main ()
{
  Hash_ctx hash_ctx (SEED);
  Zobrist_hash_impl<Hash_ctx> hash (hash_ctx);

  Move_impl move_impl;
  Database<Move_impl, Zobrist_hash_impl<Hash_ctx>> db(move_impl, hash);

  // generate db5 to db9, number is the disc on board.
  auto ret = db.gen_database("./go", "db", 5, 9, false);
  if (ret != 0)
  {
    DOUT << "Gen database go wrong\n";
  }
}





