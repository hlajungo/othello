#pragma once
#include <iostream>
#include <vector>

#include <cstdint>
#include <cassert>
//#include <type.hpp>

#ifdef DEBUG
#define DOUT std::cout
#else
#define DOUT 0 && std::cout
#endif

template <typename T>
void
DOUT_array_1d (std::vector<T> v)
{
  for (const auto& i : v)
  {
    DOUT << i << ' ';
  }
  DOUT << '\n';
}

template <typename T>
void
DOUT_array_2d (std::vector<std::vector<T> > vec)
{
  for (const auto& a_1d : vec)
  {
    for (const auto& v : a_1d)
    {
      DOUT << v << ' ';
    }
  }
  DOUT << '\n';
}

template <typename T>
void
DOUT_array_3d (std::vector<std::vector<std::vector<T> > > vec)
{
  for (const auto& a_2d : vec)
  {
    for (const auto& a_1d : a_2d)
    {
      for (const auto& v : a_1d)
      {
        DOUT << v << ' ';
      }
    }
  }
  DOUT << '\n';
}

template <typename T>
void
print_array_pair (std::vector<T> v)
{
  for (auto& i : v)
  {
    DOUT << i.first << ':' << i.second << '\n';
  }
}

// 用來展開 tuple 的 helper function
template <typename Tuple, size_t... Is>
void
print_tuple_impl (const Tuple& t, std::index_sequence<Is...>)
{
  ((DOUT << (Is == 0 ? "" : " ") << std::get<Is> (t)), ...);
  DOUT << '\n';
}

// 萬能轉發接口：列印任意型別的 tuple
template <typename... Args>
void
print_tuple (const std::tuple<Args...>& t)
{
  print_tuple_impl (t, std::index_sequence_for<Args...>{});
}

// 印出 vector<tuple<...>> 的函式
template <typename... Args>
void
prinTector_tuples (const std::vector<std::tuple<Args...> >& vec)
{
  for (const auto& t : vec)
  {
    print_tuple (t);
  }
}

inline void
DOUT_bitboard (uint64_t bitboard)
{
  std::bitset<64> binary (bitboard);
  for (int i = 63; i >= 0; --i)
  {
    DOUT << binary[i];
    if (i % 8 == 0)
    {
      DOUT << "\n";
    }
  }
  DOUT << "\n";
}
