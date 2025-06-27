%module fm_index

%{
#include "fm_index.hpp"
%}

// Include SWIG's standard integer types
%include <stdint.i>

// Include vector support
%include "std_vector.i"
%include "std_string.i"

// Method 1: Define the types before using them in templates
// This ensures SWIG understands the exact mapping
%typedef uint64_t char_type;
%typedef uint64_t size_type;
%typedef uint64_t value_type;

// Method 2: Use the actual underlying types for templates
// This is more explicit and avoids typedef resolution issues
%template(CharVector) std::vector<uint64_t>;
%template(SizeVector) std::vector<uint64_t>;
%template(ValueVector) std::vector<uint64_t>;

// For the nested vector in distinct_count_multi
%template(CharVectorVector) std::vector<std::vector<uint64_t>>;

// Include the header after all type definitions
%include "fm_index.hpp"
