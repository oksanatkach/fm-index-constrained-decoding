# fm-index-constrained-decoding

## Installation Guide

### Linux

```commandline
mv cpp_modules_linux cpp_modules

sudo apt install swig

git clone https://github.com/simongog/sdsl-lite.git

env CFLAGS='-fPIC' CXXFLAGS='-fPIC' res/external/sdsl-lite/install.sh

pip install -r requirements.txt
```

### Mac

```commandline
mv cpp_modules_mac cpp_modules

brew install swig

git clone https://github.com/xxsds/sdsl-lite.git

swig -c++ -python -I/usr/local/include -outdir cpp_modules -o cpp_modules/fm_index_wrap.cxx cpp_modules/fm_index.i

clang++ -std=c++17 -fPIC -shared -undefined dynamic_lookup \                                                      
    cpp_modules/fm_index.cpp cpp_modules/fm_index_wrap.cxx \
    -Isdsl-lite/include \
    -o cpp_modules/_fm_index.so

pip install -r requirements.txt
```
