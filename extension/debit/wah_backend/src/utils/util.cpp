#include "utils/util.h"

// The single live global from the original util.cpp.  bitvector.cpp's OR
// fast path reads it (`if (enable_fence_pointer) res.m_vec.reserve(...)`)
// to decide whether to pre-allocate space for fence pointers.  Initialised
// to false, never flipped on by any code in this tree, so the fence path
// stays dormant.
bool enable_fence_pointer = false;
