cmake_minimum_required(VERSION 3.14...3.22)


set (TEST_PROJECT "hello")
set (TEST_PROJECT_SUFFIX "test")
project (${TEST_PROJECT}_${TEST_PROJECT_SUFFIX} LANGUAGES CXX)

# ---- Options ----

option (ENABLE_TEST_COVERAGE "Enable test coverage" ON)

# --- Import ----

include (../cmake/tools.cmake)
include (../cmake/CPM.cmake)

# ---- C++20 setting ----

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON) # outputting to the file
                                       # "build/compile_commands.json"


# ---- Package ----

CPMAddPackage ("gh:doctest/doctest@2.4.9")
CPMAddPackage ("gh:TheLartians/Format.cmake@1.7.3")

# package: hello

option (TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)
if (TEST_INSTALLED_VERSION)
  find_package (${TEST_PROJECT} CONFIG REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../install/project)
else ()
  # 這會在當前文件插入 ../CMakeLists.txt，意味著加入 target ${TEST_PROJECT}
  CPMAddPackage (NAME ${TEST_PROJECT} SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../my_project)
endif ()

# ---- Create binary ----

file (GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)

add_executable (${PROJECT_NAME} ${sources})
target_link_libraries (${PROJECT_NAME} doctest::doctest ${TEST_PROJECT}::${TEST_PROJECT})


# --- Add test ----
add_test (NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME} )

# enable compiler warnings
if (NOT TEST_INSTALLED_VERSION)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #target_compile_options(UseCudaLib PUBLIC -Wall -Wpedantic -Wextra -Werror)
    target_compile_options (${PROJECT_NAME} PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wpedantic -Wextra -Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra,-Werror>
    )
  elseif (MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif ()
endif ()

# ---- Add GreeterTests ----

enable_testing ()

# Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND
# ${PROJECT_NAME})

include (${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests (${PROJECT_NAME})

install (TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION tests
)

# ---- code coverage ----

if (ENABLE_TEST_COVERAGE AND NOT TEST_INSTALLED_VERSION)
  # "-ftest-coverage" for .gcno，控制流程圖與語法分析
  # "-fprofile-arcs" for .gcda, 真正執行統計次數
  # "--coverage" == "-ftest-coverage" + "-fprofile-arcs"
  target_compile_options (${TEST_PROJECT} PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options (${TEST_PROJECT} PUBLIC -fprofile-arcs -ftest-coverage)
endif ()
