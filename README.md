# fm-index-constrained-decoding

## Installation Guide

### Linux

```commandline
git clone https://github.com/oksanatkach/fm-index-constrained-decoding.git

cd fm-index-constrained-decoding

mv cpp_modules_linux cpp_modules

apt update && apt install -y \
    build-essential \
    swig \
    cmake \
    python3.11-dev \
    libdivsufsort-dev \
    g++ \
    libnuma-dev

git clone https://github.com/simongog/sdsl-lite.git

env CFLAGS='-fPIC' CXXFLAGS='-fPIC' sdsl-lite/install.sh

swig -c++ -python -I/usr/include -outdir cpp_modules -o cpp_modules/fm_index_wrap.cxx cpp_modules/fm_index.i

# note: these root paths work if you spun a linux machine using Docker
# for an HPC system like Cirrus
g++ -std=c++17 -fPIC -shared cpp_modules/fm_index.cpp cpp_modules/fm_index_wrap.cxx /root/lib/libsdsl.a -ldivsufsort -ldivsufsort64 -I/root/include -I/usr/include/python3.11 -o cpp_modules/_fm_index.so

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
    -I sdsl-lite/include \
    -o cpp_modules/_fm_index.so

# create a python env here
pip install -r requirements.txt
```

### Building the FM-index (CLI)
To most straightforward way to build the FM-index is to use the script we have provided in `scripts/build_fm_index.py`! 
You only need  to put your retrieval corpus in a very simple TSV format as in the following example:
```
doc1    Doc 1   This is a sample document
doc2    Doc 2   This is another sample document
doc3    Doc 3   And here you find the final one
```
Fields are: 
* document id
* document title
* text 

Then you can build the FM-index with:
```commandline
FILE_I=test_data/sample_corpus.tsv
FILE_O=test_data/sample_corpus.fm_index

# optional, depends on what tokenizer is used
huggingface-cli login

python build_fm_index.py \
    $FILE_I $FILE_O \
    --hf_model meta-llama/Llama-3.2-1B-Instruct  \
    --jobs 40 --include_title
```
The parameter `--jobs` only speeds up the tokenization at the moment. `--include_title` only makes sense if your retrieval corpus has non-empty titles.


### Converting Wikipedia dump into an index

Download the wiki dump.
```commandline
# WARNING: this file is 24 GB
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# smaller file for testing
wget https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
```

Convert bz2 to tsv.
```commandline
python process_wikipedia_dump.py data/simplewiki-latest-pages-articles.xml.bz2 data/simplewiki.tsv
```
 
Build the FM index.
```commandline
FILE_I=data/simplewiki.tsv
FILE_O=data/simplewiki.fm_index
TKNSR=meta-llama/Llama-3.2-1B-Instruct

python build_fm_index.py $FILE_I $FILE_O --hf_model $TKNSR --jobs 40 --include_title

```
