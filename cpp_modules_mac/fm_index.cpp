// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "fm_index.hpp"

#include <sdsl/suffix_arrays.hpp>
#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>

using namespace sdsl;
using namespace std;


typedef csa_wt_int<> fm_index_type;
typedef uint64_t size_type;
typedef uint64_t value_type;
typedef uint64_t char_type;

FMIndex::FMIndex() {
    query_ = int_vector<>(4096);
}

FMIndex::~FMIndex() {}

void FMIndex::initialize(const vector<char_type> &data) {

    int_vector<> data2 = int_vector<>(data.size());
    for (size_type i = 0; i < data.size(); i++) data2[i] = data[i];
    construct_im(index, data2, 0);
    chars = vector<value_type>(index.wavelet_tree.sigma);
    rank_c_i = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_j = vector<size_type>(index.wavelet_tree.sigma);
}

void FMIndex::initialize_from_file(const string file, int width) {
    construct(index, file, width);
    chars = vector<value_type>(index.wavelet_tree.sigma);
    rank_c_i = vector<size_type>(index.wavelet_tree.sigma);
    rank_c_j = vector<size_type>(index.wavelet_tree.sigma);
}

size_type FMIndex::size() {
    return index.size();
}

const vector<size_type> FMIndex::backward_search_multi(const vector<char_type> query)
{
    vector<size_type> output;
    size_type l = 0;
    size_type r = index.size() - 1;  // Fixed: Use size() - 1 instead of size()
    for (size_type i = 0; i < query.size(); i++)
        backward_search(index, l, r, (char_type) query[i], l, r);
    output.push_back(l);
    output.push_back(r+1);
    return output;
}

const vector<size_type> FMIndex::backward_search_step(char_type symbol, size_type low, size_type high)
{
    vector<size_type> output;
    size_type new_low = 0;
    size_type new_high = 0;

    // Add bounds checking
    if (high >= index.size()) {
        high = index.size() - 1;
    }
    if (low > high) {
        // Invalid range
        output.push_back(0);
        output.push_back(0);
        return output;
    }

    backward_search(index, low, high, (char_type) symbol, new_low, new_high);

    // Check if the search failed
    if (new_low > new_high) {
        // No matches found, return empty range
        output.push_back(0);
        output.push_back(0);
        return output;
    }

    output.push_back(new_low);
    output.push_back(new_high + 1);
    return output;
}

const vector<char_type> FMIndex::distinct(size_type low, size_type high)
{
    vector<char_type> ret;
    if (low == high) return ret;

    // Add bounds checking
    if (high > index.size()) {
        high = index.size();
    }
    if (low >= high) return ret;

    size_type quantity;
    interval_symbols(index.wavelet_tree, low, high, quantity, chars, rank_c_i, rank_c_j);
    for (size_type i = 0; i < quantity; i++)
    {
        ret.push_back(chars[i]);
    }
    return ret;
}

const vector<char_type> FMIndex::distinct_count(size_type low, size_type high)
{
    vector<value_type> chars_ = vector<value_type>(index.wavelet_tree.sigma);
    vector<value_type> rank_c_i_ = vector<size_type>(index.wavelet_tree.sigma);
    vector<value_type> rank_c_j_ = vector<size_type>(index.wavelet_tree.sigma);

    vector<char_type> ret;
    if (low == high) return ret;

    // Add bounds checking
    if (high > index.size()) {
        high = index.size();
    }
    if (low >= high) return ret;

    size_type quantity;
    interval_symbols(index.wavelet_tree, low, high, quantity, chars_, rank_c_i_, rank_c_j_);
    for (size_type i = 0; i < quantity; i++)
    {
	
        ret.push_back(chars_[i]);
        ret.push_back((char_type) rank_c_j_[i] - rank_c_i_[i]);
    }
    return ret;
}

const vector<vector<char_type>> FMIndex::distinct_count_multi(vector<size_type> lows, vector<size_type> highs)
{
    vector<vector<char_type>> ret;
    vector<std::future<const vector<char_type>>> threads;


    for (size_type i = 0; i < lows.size(); i++) {
        threads.push_back(
            std::async(&FMIndex::distinct_count, this, lows[i], highs[i])
        );
    }

    for (size_type i = 0; i < lows.size(); i++) {
        ret.push_back(
            threads[i].get()
        );
    }

    return ret;
}

size_type FMIndex::locate(size_type row)
{
    if (row >= index.size()) return -1;
    return (size_type) index[row];
}

const vector<char_type> FMIndex::extract_text(size_type begin, size_type end)
{
    vector<char_type> ret;
    if (end <= begin) return ret;
    if (end > index.size()) end = index.size();

    size_type start = index.isa[end-1];  // Fixed: use end-1 since end is exclusive
    char_type symbol = index.bwt[start];
    ret.push_back(symbol);
    if (end - begin == 1) return ret;

    for (size_type i = 0; i < end-begin-1; i++)
    {
        vector<size_type> search_result = backward_search_step(symbol, start, start+1);
        start = search_result[0];
        if (start >= index.size()) break;  // Safety check
        symbol = index.bwt[start];
        ret.push_back(symbol);
    }
    return ret;
}

void FMIndex::save(const string path)
{
    store_to_file(index, path);
}

FMIndex load_FMIndex(const string path)
{
    FMIndex fm;
    load_from_file(fm.index, path);
    fm.chars = vector<value_type>(fm.index.wavelet_tree.sigma);
    fm.rank_c_i = vector<size_type>(fm.index.wavelet_tree.sigma);
    fm.rank_c_j = vector<size_type>(fm.index.wavelet_tree.sigma);
    return fm;
}