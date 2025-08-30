#!/bin/bash

# lazy project
lpcmake ()
{
  cmake -S orthello -B build/orthello
  [[ $? -ne 0 ]] && { echo "error: configure error"; return 1; }
  cmake --build build/orthello
  [[ $? -ne 0 ]] && { echo "error: build error"; return 1; }
}

# lazy standalone
lscmake ()
{
  cmake -S standalone/ -B build/standalone
  [[ $? -ne 0 ]] && { echo "error: configure error"; return 1; }
  cmake --build build/standalone
  [[ $? -ne 0 ]] && { echo "error: build error"; return 1; }
  ./build/standalone/project_standalone
}

ltcmake ()
{
  cmake -S test/ -B build/test
  [[ $? -ne 0 ]] && { echo "error: configure error"; return 1; }
  cmake --build build/test
  [[ $? -ne 0 ]] && { echo "error: build error"; return 1; }
  ctest --test-dir ./build/test --output-on-failure
  #./build/test/orthello_test
}

ldcmake ()
{
  cmake -S documentation/ -B build/documentation
  [[ $? -ne 0 ]] && { echo "error: configure error"; return 1; }
  cmake --build build/documentation --target GenerateDocs
  [[ $? -ne 0 ]] && { echo "error: build error"; return 1; }
  open build/documentation/doxygen/html/index.html
}
