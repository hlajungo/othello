import os
import ycm_core


def DirectoryOfThisScript():
    return os.path.dirname(os.path.abspath(__file__))

flags = [
        '-Wall',
        '-O3',
        '-std=c++20',
        '-xc++',
        '-DNDEBUG',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-Wno-long-long',
        '-Wno-variadic-macros',
        '-fexceptions',
        '-ferror-limit=10000' ,
        '-I', DirectoryOfThisScript() + '/orthello/include',
        '-I', DirectoryOfThisScript() + '/orthello/source',
        '-I', DirectoryOfThisScript() + '/build/test/_deps/doctest-src',
        ]

def Settings ( filename, **kwargs ):
    return { 'flags': flags
            , 'do_cache': True
            }




