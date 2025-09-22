#pragma once
#include <filesystem>
#include <fstream>
#include <type.hpp>

namespace std_fs = std::filesystem;

inline void
get_file_1d (array_1d_t<std::string>& file_1d,
             std::string_view dir,
             std::string_view file_prefix)
{
  for (const auto& entry : std_fs::directory_iterator (dir))
  {
    if (entry.is_regular_file ())
    {
      std::string filename = entry.path ().filename ().string ();
      // 檢查前綴
      if (filename.rfind (file_prefix, 0) == 0)
      {
        file_1d.push_back (entry.path ().string ());
      }
    }
  }
}

template <typename T>
bool
is_in_file (std::string_view filename, const T& target)
{
  static_assert (std::is_trivially_copyable_v<T>,
                 "is_in_file only supports trivially copyable types");

  // fopen is C function, only `const char*` supported
  FILE* file = fopen (std::string (filename).c_str (), "rb");
  if (!file)
  {
    perror ("Cannot open file");
    return false;
  }

  T h;
  while (fread (&h, sizeof (h), 1, file) == 1)
  {
    if (h == target)
    {
      fclose (file);
      return true;
    }
  }
  fclose (file);
  return false;
}

bool
is_file_equal (const std::string& path1, const std::string& path2)
{
  std::ifstream f1 (path1, std::ios::binary);
  std::ifstream f2 (path2, std::ios::binary);

  if (!f1.is_open () || !f2.is_open ())
    return false;

  // 讀進 vector<char> 後比較
  std::vector<char> buf1 ((std::istreambuf_iterator<char> (f1)),
                          std::istreambuf_iterator<char> ());
  std::vector<char> buf2 ((std::istreambuf_iterator<char> (f2)),
                          std::istreambuf_iterator<char> ());

  return buf1 == buf2;
}

void
try_create_dir (std_fs::path& path)
{
  if (!std_fs::exists (path))
    std_fs::create_directories (path);
}

// Linux mmap optimize for reading file
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

template <typename T>
bool
is_in_file2 (std::string_view filename, const T& target)
{
  static_assert (std::is_trivially_copyable_v<T>,
                 "is_in_file only supports trivially copyable types");

  std_fs::path filepath (filename);

  int fd = ::open (filepath.c_str (), O_RDONLY);
  if (fd == -1)
  {
    std::perror ("open");
    return false;
  }

  struct stat st;
  if (::fstat (fd, &st) == -1)
  {
    std::perror ("fstat");
    ::close (fd);
    return false;
  }

  if (st.st_size < sizeof (T))
  {
    ::close (fd);
    return false;
  }

  void* addr = ::mmap (nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close (fd);
  if (addr == MAP_FAILED)
  {
    std::perror ("mmap");
    return false;
  }

  size_t count = st.st_size / sizeof (T);
  auto* data = static_cast<const T*> (addr);

  bool found = false;
  for (size_t i = 0; i < count; ++i)
  {
    if (data[i] == target)
    {
      found = true;
      break;
    }
  }

  ::munmap (addr, st.st_size);
  return found;
}

// windows mmap reading file
#if 0
#include <filesystem>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <windows.h>

namespace fs = std::filesystem;

template <typename T>
bool
is_in_file (std::string_view filename, const T& target)
{
  static_assert (std::is_trivially_copyable_v<T>,
                 "is_in_file only supports trivially copyable types");

  fs::path filepath (filename);

  HANDLE hFile = CreateFileW (filepath.wstring ().c_str (),
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              nullptr,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL,
                              nullptr);
  if (hFile == INVALID_HANDLE_VALUE)
  {
    std::cerr << "Cannot open file\n";
    return false;
  }

  LARGE_INTEGER fileSize;
  if (!GetFileSizeEx (hFile, &fileSize) || fileSize.QuadPart < sizeof (T))
  {
    CloseHandle (hFile);
    return false;
  }

  HANDLE hMapping
      = CreateFileMappingW (hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hMapping)
  {
    CloseHandle (hFile);
    std::cerr << "CreateFileMapping failed\n";
    return false;
  }

  void* addr = MapViewOfFile (hMapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle (hMapping);
  CloseHandle (hFile);

  if (!addr)
  {
    std::cerr << "MapViewOfFile failed\n";
    return false;
  }

  size_t count = static_cast<size_t> (fileSize.QuadPart / sizeof (T));
  auto* data = static_cast<const T*> (addr);

  bool found = false;
  for (size_t i = 0; i < count; ++i)
  {
    if (data[i] == target)
    {
      found = true;
      break;
    }
  }

  UnmapViewOfFile (addr);
  return found;
}
#endif
