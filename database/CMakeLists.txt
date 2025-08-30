cmake_minimum_required (VERSION 3.14...3.22)

project (gen_database LANGUAGES CXX)

# cpm: init
include (../cmake/CPM.cmake)

# ---- Package ----

# package: hello
option (HELLO_INSTALLED_VERSION "Use installed package or use source code" OFF)

if (HELLO_INSTALLED_VERSION)
  # "find_package" for using compiled package
  # CONFIG for finding helloConfig.cmake Not finding FindHello.cmake
  # REQUIRED for jump error when not found
  # list(APPEND CMAKE_PREFIX_PATH "/PATH") for simple path
  find_package (
    #hello CONFIG REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../install/project
    hello CONFIG REQUIRED
  )
else ()
  # "CPMAddPackage" for adding source code
  CPMAddPackage(
    NAME orthello
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../orthello
  )
endif ()



# package: cxxopts
cpmaddpackage (
  GITHUB_REPOSITORY
  jarro2783/cxxopts
  VERSION
  3.0.0
  OPTIONS
  "CXXOPTS_BUILD_EXAMPLES NO"
  "CXXOPTS_BUILD_TESTS NO"
  "CXXOPTS_ENABLE_INSTALL YES"
)

# ---- Compile & Install ----

file (
  GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp
)

add_executable (${PROJECT_NAME} ${sources})

target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG)

set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_link_libraries (${PROJECT_NAME} orthello::orthello cxxopts)

install (
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
