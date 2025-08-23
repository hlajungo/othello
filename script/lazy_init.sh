#!/bin/bash

# lazy project
lpcmake ()
{
  cmake -S orthello -B build/orthello
  cmake --build build/orthello
}

# lazy standalone
lscmake ()
{
  cmake -S standalone/ -B build/standalone
  cmake --build build/standalone
  ./build/standalone/project_standalone
}

ltcmake ()
{
  cmake -S test/ -B build/test
  cmake --build build/test
  ctest --test-dir ./build/test --output-on-failure
  #./build/test/orthello_test
}

ldcmake ()
{
  cmake -S documentation/ -B build/documentation
  cmake --build build/documentation --target GenerateDocs
  open build/documentation/doxygen/html/index.html
}
