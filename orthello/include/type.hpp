#pragma once
#include <exception>
#include <string>
#include <vector>

#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

template <class T>
T
type_converter (char* str_ptr, char** end)
{
  if constexpr (std::is_same_v<float, T>)
  {
    return strtof (str_ptr, end);
  }
  else if constexpr (std::is_same_v<double, T>)
  {
    return strtod (str_ptr, end);
  }
  else if constexpr (std::is_same_v<long double, T>)
  {
    return strtold (str_ptr, end);
  }
  // induce compile time error if none of the cases are fulfilled
  else
    static_assert (not std::is_same_v<T, T>, "Invalid template type");
}

template <typename T>
using array_1d_t = std::vector<T, std::allocator<T> >;

template <typename T>
using array_2d_t = std::vector<std::vector<T, std::allocator<T> > >;


