
#ifndef GZ_SUDOKU_RUST_V1_H
#define GZ_SUDOKU_RUST_V1_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <cstring>      // For std::memset(), std::memcpy()
#include <vector>
#include <bitset>
#include <array>        // For std::array<T, Size>

#include "BasicSolver.h"
#include "Sudoku.h"
#include "SudokuTable.h"
#include "StopWatch.h"
#include "BitUtils.h"
#include "BitSet.h"
#include "PackedBitSet.h"
#include "BitArray.h"
#include "BitVec.h"

using namespace gzSudoku;

//
// Whether search no guess steps only?
//
#define RUST_V1_ONLY_NO_GUESS       0

namespace {

static const size_t kSearchMode = SearchMode::OneSolution;

static const bool kCheckSolvedRows = false;
static const bool kUseFastMode = false;

static const bool kReachSolutionsLimitToExit = true;
static const bool kReachSolutionsLimitToExit2 = true;

// Kill all in other blocks locked column / box
static const uint32_t nonconflicting_cells_neighbor_bands_by_locked_candidates[512] = {
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07767767767, 07766766766, 07765765765, 07767767767, 07763763763, 07767767767, 07767767767, 07767767767,
    07757757757, 07756756756, 07755755755, 07757757757, 07753753753, 07757757757, 07757757757, 07757757757,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07737737737, 07736736736, 07735735735, 07737737737, 07733733733, 07737737737, 07737737737, 07737737737,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07677677677, 07676676676, 07675675675, 07677677677, 07673673673, 07677677677, 07677677677, 07677677677,
    07667667667, 07666666666, 07665665665, 07667667667, 07663663663, 07667667667, 07667667667, 07667667667,
    07657657657, 07656656656, 07655655655, 07657657657, 07653653653, 07657657657, 07657657657, 07657657657,
    07677677677, 07676676676, 07675675675, 07677677677, 07673673673, 07677677677, 07677677677, 07677677677,
    07637637637, 07636636636, 07635635635, 07637637637, 07633633633, 07637637637, 07637637637, 07637637637,
    07677677677, 07676676676, 07675675675, 07677677677, 07673673673, 07677677677, 07677677677, 07677677677,
    07677677677, 07676676676, 07675675675, 07677677677, 07673673673, 07677677677, 07677677677, 07677677677,
    07677677677, 07676676676, 07675675675, 07677677677, 07673673673, 07677677677, 07677677677, 07677677677,
    07577577577, 07576576576, 07575575575, 07577577577, 07573573573, 07577577577, 07577577577, 07577577577,
    07567567567, 07566566566, 07565565565, 07567567567, 07563563563, 07567567567, 07567567567, 07567567567,
    07557557557, 07556556556, 07555555555, 07557557557, 07553553553, 07557557557, 07557557557, 07557557557,
    07577577577, 07576576576, 07575575575, 07577577577, 07573573573, 07577577577, 07577577577, 07577577577,
    07537537537, 07536536536, 07535535535, 07537537537, 07533533533, 07537537537, 07537537537, 07537537537,
    07577577577, 07576576576, 07575575575, 07577577577, 07573573573, 07577577577, 07577577577, 07577577577,
    07577577577, 07576576576, 07575575575, 07577577577, 07573573573, 07577577577, 07577577577, 07577577577,
    07577577577, 07576576576, 07575575575, 07577577577, 07573573573, 07577577577, 07577577577, 07577577577,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07767767767, 07766766766, 07765765765, 07767767767, 07763763763, 07767767767, 07767767767, 07767767767,
    07757757757, 07756756756, 07755755755, 07757757757, 07753753753, 07757757757, 07757757757, 07757757757,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07737737737, 07736736736, 07735735735, 07737737737, 07733733733, 07737737737, 07737737737, 07737737737,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07377377377, 07376376376, 07375375375, 07377377377, 07373373373, 07377377377, 07377377377, 07377377377,
    07367367367, 07366366366, 07365365365, 07367367367, 07363363363, 07367367367, 07367367367, 07367367367,
    07357357357, 07356356356, 07355355355, 07357357357, 07353353353, 07357357357, 07357357357, 07357357357,
    07377377377, 07376376376, 07375375375, 07377377377, 07373373373, 07377377377, 07377377377, 07377377377,
    07337337337, 07336336336, 07335335335, 07337337337, 07333333333, 07337337337, 07337337337, 07337337337,
    07377377377, 07376376376, 07375375375, 07377377377, 07373373373, 07377377377, 07377377377, 07377377377,
    07377377377, 07376376376, 07375375375, 07377377377, 07373373373, 07377377377, 07377377377, 07377377377,
    07377377377, 07376376376, 07375375375, 07377377377, 07373373373, 07377377377, 07377377377, 07377377377,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07767767767, 07766766766, 07765765765, 07767767767, 07763763763, 07767767767, 07767767767, 07767767767,
    07757757757, 07756756756, 07755755755, 07757757757, 07753753753, 07757757757, 07757757757, 07757757757,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07737737737, 07736736736, 07735735735, 07737737737, 07733733733, 07737737737, 07737737737, 07737737737,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07767767767, 07766766766, 07765765765, 07767767767, 07763763763, 07767767767, 07767767767, 07767767767,
    07757757757, 07756756756, 07755755755, 07757757757, 07753753753, 07757757757, 07757757757, 07757757757,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07737737737, 07736736736, 07735735735, 07737737737, 07733733733, 07737737737, 07737737737, 07737737737,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07767767767, 07766766766, 07765765765, 07767767767, 07763763763, 07767767767, 07767767767, 07767767767,
    07757757757, 07756756756, 07755755755, 07757757757, 07753753753, 07757757757, 07757757757, 07757757757,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07737737737, 07736736736, 07735735735, 07737737737, 07733733733, 07737737737, 07737737737, 07737737737,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777,
    07777777777, 07776776776, 07775775775, 07777777777, 07773773773, 07777777777, 07777777777, 07777777777
};

// Triads mask per row
static const uint32_t rowTriadsMaskTbl[512] = {
    0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
    2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7
};

// Keep all locked candidates
static const uint32_t keepLockedCandidatesTbl[512] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 07007070700, 07707070700, 07007770700, 07707770700,
    0, 0, 0, 0, 07077070700, 07777070700, 07777770700, 07777770700,
    0, 0, 07007700070, 07077700070, 0, 0, 07007770070, 07077770070,
    0, 0, 07707700070, 07777700070, 0, 0, 07777770070, 07777770070,
    0, 0, 07007700770, 07777700770, 07007070770, 07777070770, 07007770770, 07777770770,
    0, 0, 07707700770, 07777700770, 07077070770, 07777070770, 07777770770, 07777770770,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 07070007700, 07070707700, 07770007700, 07770707700,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 07077007700, 07777707700, 07777007700, 07777707700,
    0, 07070700007, 0, 07077700007, 0, 07070707007, 0, 07077707007,
    0, 07070700707, 0, 07777700707, 07070007707, 07070707707, 07777007707, 07777707707,
    0, 07770700007, 0, 07777700007, 0, 07777707007, 0, 07777707007,
    0, 07770700707, 0, 07777700707, 07077007707, 07777707707, 07777007707, 07777707707,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 07070077700, 07070777700, 07770777700, 07770777700,
    0, 0, 0, 0, 07007077700, 07707777700, 07007777700, 07707777700,
    0, 0, 0, 0, 07077077700, 07777777700, 07777777700, 07777777700,
    0, 07070700077, 07007700077, 07077700077, 0, 07070777077, 07007777077, 07077777077,
    0, 07070700777, 07707700777, 07777700777, 07070077777, 07070777777, 07777777777, 07777777777,
    0, 07770700777, 07007700777, 07777700777, 07007077777, 07777777777, 07007777777, 07777777777,
    0, 07770700777, 07707700777, 07777700777, 07077077777, 07777777777, 07777777777, 07777777777,
    0, 0, 0, 0, 0, 0, 0, 0,
    00, 0, 07700007070, 07700077070, 0, 0, 07770007070, 07770077070,
    00, 07700070007, 0, 07700077007, 0, 07707070007, 0, 07707077007,
    00, 07700070077, 07700007077, 07700077077, 0, 07777070077, 07777007077, 07777077077,
    00, 0, 0, 0, 0, 0, 0, 0,
    00, 0, 07707007070, 07777077070, 0, 0, 07777007070, 07777077070,
    00, 07770070007, 0, 07777077007, 0, 07777070007, 0, 07777077007,
    00, 07770070077, 07707007077, 07777077077, 0, 07777070077, 07777007077, 07777077077,
    00, 0, 0, 0, 0, 0, 0, 0,
    00, 0, 07700707070, 07700777070, 0, 0, 07770777070, 07770777070,
    00, 07700070707, 0, 07700777707, 07007070707, 07707070707, 07007777707, 07707777707,
    00, 07700070777, 07700707777, 07700777777, 07077070777, 07777070777, 07777777777, 07777777777,
    00, 0, 07007707070, 07077777070, 0, 0, 07007777070, 07077777070,
    00, 0, 07707707070, 07777777070, 0, 0, 07777777070, 07777777070,
    00, 07770070777, 07007707777, 07777777777, 07007070777, 07777070777, 07007777777, 07777777777,
    00, 07770070777, 07707707777, 07777777777, 07077070777, 07777070777, 07777777777, 07777777777,
    00, 0, 0, 0, 0, 0, 0, 0,
    00, 0, 07700007770, 07700777770, 07070007770, 07070777770, 07770007770, 07770777770,
    00, 07700770007, 0, 07700777007, 0, 07707777007, 0, 07707777007,
    00, 07700770777, 07700007777, 07700777777, 07077007777, 07777777777, 07777007777, 07777777777,
    00, 07070770007, 0, 07077777007, 0, 07070777007, 0, 07077777007,
    00, 07070770777, 07707007777, 07777777777, 07070007777, 07070777777, 07777007777, 07777777777,
    00, 07770770007, 0, 07777777007, 0, 07777777007, 0, 07777777007,
    00, 07770770777, 07707007777, 07777777777, 07077007777, 07777777777, 07777007777, 07777777777,
    00, 0, 0, 0, 0, 0, 0, 0,
    00, 0, 07700707770, 07700777770, 07070077770, 07070777770, 07770777770, 07770777770,
    00, 07700770707, 0, 07700777707, 07007077707, 07707777707, 07007777707, 07707777707,
    00, 07700770777, 07700707777, 07700777777, 07077077777, 07777777777, 07777777777, 07777777777,
    00, 07070770077, 07007707077, 07077777077, 0, 07070777077, 07007777077, 07077777077,
    00, 07070770777, 07707707777, 07777777777, 07070077777, 07070777777, 07777777777, 07777777777,
    00, 07770770777, 07007707777, 07777777777, 07007077777, 07777777777, 07007777777, 07777777777,
    00, 07770770777, 07707707777, 07777777777, 07077077777, 07777777777, 07777777777, 07777777777
};

// Get the masks of all row triads is single
static const uint32_t rowTriadsSingleMaskTbl[512] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0124, 0124, 0124, 0124, 0, 0, 0, 0, 0124, 0124, 0124, 0124,
    0, 0, 0142, 0142, 0, 0, 0142, 0142, 0, 0, 0142, 0142, 0, 0, 0142, 0142, 0, 0, 0142, 0142, 0124, 0124, 0100, 0100, 0, 0, 0142, 0142, 0124, 0124, 0100, 0100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0214, 0214, 0214, 0214, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0214, 0214, 0214, 0214,
    0, 0241, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0214, 0200, 0214, 0200, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0214, 0200, 0214, 0200,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0214, 0214, 0214, 0214, 0, 0, 0, 0, 0124, 0124, 0124, 0124, 0, 0, 0, 0, 04, 04, 04, 04,
    0, 0241, 0142, 040, 0, 0241, 0142, 040, 0, 0241, 0142, 040, 0214, 0200, 0, 0, 0, 0241, 0142, 040, 0124, 0, 0100, 0, 0, 0241, 0142, 040, 04, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0412, 0412, 0, 0, 0412, 0412, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0412, 0400, 0, 0421, 0412, 0400,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0412, 0412, 0, 0, 0412, 0412, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0412, 0400, 0, 0421, 0412, 0400,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0412, 0412, 0, 0, 0412, 0412, 0, 0421, 0, 0421, 0124, 020, 0124, 020, 0, 0421, 0412, 0400, 0124, 020, 0, 0,
    0, 0, 0142, 0142, 0, 0, 0142, 0142, 0, 0, 02, 02, 0, 0, 02, 02, 0, 0421, 0142, 0, 0124, 020, 0100, 0, 0, 0421, 02, 0, 0124, 020, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0412, 0412, 0214, 0214, 010, 010, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0, 0421, 0412, 0400, 0214, 0, 010, 0,
    0, 0241, 0, 0241, 0, 0241, 0, 0241, 0, 0241, 0412, 0, 0214, 0200, 010, 0, 0, 01, 0, 01, 0, 01, 0, 01, 0, 01, 0412, 0, 0214, 0, 010, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0412, 0412, 0214, 0214, 010, 010, 0, 0421, 0, 0421, 0124, 020, 0124, 020, 0, 0421, 0412, 0400, 04, 0, 0, 0,
    0, 0241, 0142, 040, 0, 0241, 0142, 040, 0, 0241, 02, 0, 0214, 0200, 0, 0, 0, 01, 0142, 0, 0124, 0, 0100, 0, 0, 01, 02, 0, 04, 0, 0, 0
};

// Get the masks of combine all columns in band is single
static const uint32_t combColumnSingleMaskTbl[512] = {
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0777, 0777, 0666, 0777, 0666, 0666, 0666,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0555, 0555, 0444, 0555, 0444, 0444, 0444, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0777, 0777, 0666, 0777, 0666, 0666, 0666,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0555, 0555, 0444, 0555, 0444, 0444, 0444, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0333, 0333, 0222, 0333, 0222, 0222, 0222,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0111, 0111, 00, 0111, 00, 00, 00, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0777, 0777, 0666, 0777, 0666, 0666, 0666,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0777, 0777, 0666, 0777, 0666, 0666, 0666, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 0555, 0555, 0444, 0555, 0444, 0444, 0444, 00, 0555, 0555, 0444, 0555, 0444, 0444, 0444,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0333, 0333, 0222, 0333, 0222, 0222, 0222,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0111, 0111, 00, 0111, 00, 00, 00, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0333, 0333, 0222, 0333, 0222, 0222, 0222,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0111, 0111, 00, 0111, 00, 00, 00, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 00, 00, 00, 00, 00, 00, 00, 00, 0333, 0333, 0222, 0333, 0222, 0222, 0222,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0333, 0333, 0222, 0333, 0222, 0222, 0222, 00, 0111, 0111, 00, 0111, 00, 00, 00,
    00, 0111, 0111, 00, 0111, 00, 00, 00, 00, 0111, 0111, 00, 0111, 00, 00, 00
};

// Hidden single in row, 1 indicate row is a hidden single, mode: 1 to 111
static const uint32_t rowHiddenSingleMaskTbl[512] = {
    0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
    2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
    6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7,
};

// Hidden single in row, 1 indicate row is not a hidden single, mode: 1 to 111
static const int rowHiddenSingleReverseMaskTbl[512] = {
    7, 6, 6, 6, 6, 6, 6, 6, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
    5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
};

// Rows where single found  000 to 111
static const unsigned int solvedRowsBitMaskTbl[8] = {
    00, 0777, 0777000, 0777777, 0777000000, 0777000777, 0777777000, 0777777777
};

// Rows where single found  000 to 111
static const unsigned int solvedRowsReverseBitMaskTbl[8] = {
    0777777777, 0777777000, 0777000777, 0777000000, 0777777, 0777000, 0777, 00
};

static const uint32_t bandUnsolvedMaskTbl32[81] = {
    0x37E3F001, 0x37E3F002, 0x37E3F004, 0x371F8E08, 0x371F8E10, 0x371F8E20, 0x30FC7E40, 0x30FC7E80,
    0x30FC7F00, 0x2FE003F8, 0x2FE005F8, 0x2FE009F8, 0x2F1C11C7, 0x2F1C21C7, 0x2F1C41C7, 0x28FC803F,
    0x28FD003F, 0x28FE003F, 0x1807F1F8, 0x180BF1F8, 0x1813F1F8, 0x18238FC7, 0x18438FC7, 0x18838FC7,
    0x19007E3F, 0x1A007E3F, 0x1C007E3F, 0x37E3F001, 0x37E3F002, 0x37E3F004, 0x371F8E08, 0x371F8E10,
    0x371F8E20, 0x30FC7E40, 0x30FC7E80, 0x30FC7F00, 0x2FE003F8, 0x2FE005F8, 0x2FE009F8, 0x2F1C11C7,
    0x2F1C21C7, 0x2F1C41C7, 0x28FC803F, 0x28FD003F, 0x28FE003F, 0x1807F1F8, 0x180BF1F8, 0x1813F1F8,
    0x18238FC7, 0x18438FC7, 0x18838FC7, 0x19007E3F, 0x1A007E3F, 0x1C007E3F, 0x37E3F001, 0x37E3F002,
    0x37E3F004, 0x371F8E08, 0x371F8E10, 0x371F8E20, 0x30FC7E40, 0x30FC7E80, 0x30FC7F00, 0x2FE003F8,
    0x2FE005F8, 0x2FE009F8, 0x2F1C11C7, 0x2F1C21C7, 0x2F1C41C7, 0x28FC803F, 0x28FD003F, 0x28FE003F,
    0x1807F1F8, 0x180BF1F8, 0x1813F1F8, 0x18238FC7, 0x18438FC7, 0x18838FC7, 0x19007E3F, 0x1A007E3F,
    0x1C007E3F
};

static const uint64_t bandUnsolvedMaskTbl64[81] = {
    0xFFFFFFFF37E3F001, 0xFFFFFFFF37E3F002, 0xFFFFFFFF37E3F004, 0xFFFFFFFF371F8E08,
    0xFFFFFFFF371F8E10, 0xFFFFFFFF371F8E20, 0xFFFFFFFF30FC7E40, 0xFFFFFFFF30FC7E80,

    0xFFFFFFFF30FC7F00, 0xFFFFFFFF2FE003F8, 0xFFFFFFFF2FE005F8, 0xFFFFFFFF2FE009F8,
    0xFFFFFFFF2F1C11C7, 0xFFFFFFFF2F1C21C7, 0xFFFFFFFF2F1C41C7, 0xFFFFFFFF28FC803F,

    0xFFFFFFFF28FD003F, 0xFFFFFFFF28FE003F, 0xFFFFFFFF1807F1F8, 0xFFFFFFFF180BF1F8,
    0xFFFFFFFF1813F1F8, 0xFFFFFFFF18238FC7, 0xFFFFFFFF18438FC7, 0xFFFFFFFF18838FC7,

    0xFFFFFFFF19007E3F, 0xFFFFFFFF1A007E3F, 0xFFFFFFFF1C007E3F, 0x37E3F001FFFFFFFF,  // 28 (27 + 1)
    0x37E3F002FFFFFFFF, 0x37E3F004FFFFFFFF, 0x371F8E08FFFFFFFF, 0x371F8E10FFFFFFFF,

    0x371F8E20FFFFFFFF, 0x30FC7E40FFFFFFFF, 0x30FC7E80FFFFFFFF, 0x30FC7F00FFFFFFFF,
    0x2FE003F8FFFFFFFF, 0x2FE005F8FFFFFFFF, 0x2FE009F8FFFFFFFF, 0x2F1C11C7FFFFFFFF,

    0x2F1C21C7FFFFFFFF, 0x2F1C41C7FFFFFFFF, 0x28FC803FFFFFFFFF, 0x28FD003FFFFFFFFF,
    0x28FE003FFFFFFFFF, 0x1807F1F8FFFFFFFF, 0x180BF1F8FFFFFFFF, 0x1813F1F8FFFFFFFF,

    0x18238FC7FFFFFFFF, 0x18438FC7FFFFFFFF, 0x18838FC7FFFFFFFF, 0x19007E3FFFFFFFFF,
    0x1A007E3FFFFFFFFF, 0x1C007E3FFFFFFFFF, 0xFFFFFFFF37E3F001, 0xFFFFFFFF37E3F002,  // 56 (54 + 2)

    0xFFFFFFFF37E3F004, 0xFFFFFFFF371F8E08, 0xFFFFFFFF371F8E10, 0xFFFFFFFF371F8E20,
    0xFFFFFFFF30FC7E40, 0xFFFFFFFF30FC7E80, 0xFFFFFFFF30FC7F00, 0xFFFFFFFF2FE003F8,

    0xFFFFFFFF2FE005F8, 0xFFFFFFFF2FE009F8, 0xFFFFFFFF2F1C11C7, 0xFFFFFFFF2F1C21C7,
    0xFFFFFFFF2F1C41C7, 0xFFFFFFFF28FC803F, 0xFFFFFFFF28FD003F, 0xFFFFFFFF28FE003F,

    0xFFFFFFFF1807F1F8, 0xFFFFFFFF180BF1F8, 0xFFFFFFFF1813F1F8, 0xFFFFFFFF18238FC7,
    0xFFFFFFFF18438FC7, 0xFFFFFFFF18838FC7, 0xFFFFFFFF19007E3F, 0xFFFFFFFF1A007E3F,

    0xFFFFFFFF1C007E3F
};

static const int8_t bandBitPosToPos64[2][64] = {
    // bit64[0]
    {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, -1, -1, -1, -1, -1,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, -1, -1, -1, -1, -1
    },
        
    // bit64[1]
    {
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    }
};

static const int8_t bandBitPosToPos32[4][32] = {
    // bit32[0]
    {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, -1, -1, -1, -1, -1
    },

    // bit32[1]
    {
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, -1, -1, -1, -1, -1
    },
        
    // bit32[2]
    {
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, -1, -1, -1, -1, -1
    },

    // bit32[3]
    {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    }
};

static const uint32_t boxesMaskTbl[4] = {
    0007007007, 0070070070, 0700700700, 0
};

static const uint32_t boxToBoxesMaskTbl[9] = {
    0007007007, 0070070070, 0700700700,
    0007007007, 0070070070, 0700700700,
    0007007007, 0070070070, 0700700700
};

class RustV1Solver : public BasicSolver {
public:
    typedef BasicSolver                         basic_solver;
    typedef RustV1Solver                          this_type;

    typedef typename Sudoku::BitMask            BitMask;
    typedef typename Sudoku::BitMaskTable       BitMaskTable;

    typedef typename SudokuTable::NeighborCells NeighborCells;
    typedef typename SudokuTable::CellInfo      CellInfo;
    typedef typename SudokuTable::BoxesInfo     BoxesInfo;

    static const size_t kAlignment = Sudoku::Alignment;
    static const size_t BoxCellsX = Sudoku::BoxCellsX;      // 3
    static const size_t BoxCellsY = Sudoku::BoxCellsY;      // 3
    static const size_t BoxCountX = Sudoku::BoxCountX;      // 3
    static const size_t BoxCountY = Sudoku::BoxCountY;      // 3
    static const size_t MinNumber = Sudoku::MinNumber;      // 1
    static const size_t MaxNumber = Sudoku::MaxNumber;      // 9

    static const size_t Rows = Sudoku::Rows;
    static const size_t Cols = Sudoku::Cols;
    static const size_t Boxes = Sudoku::Boxes;
    static const size_t BoxSize = Sudoku::BoxSize;
    static const size_t Numbers = Sudoku::Numbers;

    static const size_t BoardSize = Sudoku::BoardSize;
    static const size_t TotalSize = Sudoku::TotalSize;
    static const size_t Neighbors = Sudoku::Neighbors;

    static const size_t Rows16 = Sudoku::Rows16;
    static const size_t Cols16 = Sudoku::Cols16;
    static const size_t Numbers10 = Sudoku::Numbers10;
    static const size_t Numbers16 = Sudoku::Numbers16;
    static const size_t Boxes16 = Sudoku::Boxes16;
    static const size_t BoxSize16 = Sudoku::BoxSize16;
    static const size_t BoardSize16 = Sudoku::BoardSize16;

    static const size_t kAllRowBits = Sudoku::AllRowBits;
    static const size_t kAllColBits = Sudoku::AllColBits;
    static const size_t kAllBoxBits = Sudoku::AllBoxBits;
    static const size_t kAllBoxCellBits = Sudoku::AllBoxCellBits;
    static const size_t kAllNumberBits = Sudoku::AllNumberBits;

    static const bool kAllDimIsSame = Sudoku::AllDimIsSame;

    // all pencil marks set - 27 bits per band
    static const uint32_t kBitSet27          = 0x07FFFFFFUL;
    static const uint64_t kBitSet27_Single64 = 0x0000000007FFFFFFULL;
    static const uint64_t kBitSet27_Double64 = 0x07FFFFFF07FFFFFFULL;

    static const uint32_t kFullRowBits   = 0x01FFUL;
    static const uint32_t kFullRowBits_1 = 0x01FFUL << 9U;
    static const uint32_t kFullRowBits_2 = 0x01FFUL << 18U;

    static const uint32_t kBand0RowBits  = 0007;
    static const uint32_t kBand1RowBits  = 0070;
    static const uint32_t kBand2RowBits  = 0700;

private:
    enum LiteralType {
        NumRowCols,
        NumColRows,
        NumBoxCells,
        BoxCellNums,
        Unknown
    };

#pragma pack(push, 1)

    struct alignas(32) Counter {
        int16_t digits[16];
    };

    struct alignas(32) BoxCounter {
        int16_t boxes[16];
    };

    union alignas(16) BandBoard {
        uint32_t bands[4];
        uint64_t bands64[2];

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
        void clear() {
            this->bands64[0] = 0;
            this->bands64[1] = 0;
        }

        void full() {
            this->bands64[0] = kBitSet27_Double64;
            this->bands64[1] = kBitSet27_Single64;
        }

        bool operator == (const BandBoard & rhs) {
            return (this->bands64[0] == rhs.bands64[0] && this->bands64[1] == rhs.bands64[1]);
        }

        bool operator != (const BandBoard & rhs) {
            return (this->bands64[0] != rhs.bands64[0] || this->bands64[1] != rhs.bands64[1]);
        }
#else
        void clear() {
            this->bands[0] = 0;
            this->bands[1] = 0;
            this->bands[2] = 0;
            this->bands[3] = 0;
        }

        void full() {
            this->bands[0] = kBitSet27;
            this->bands[1] = kBitSet27;
            this->bands[2] = kBitSet27;
            this->bands[3] = 0;
        }

        bool operator == (const BandBoard & rhs) {
            return (this->bands[0] == rhs.bands[0] &&
                    this->bands[1] == rhs.bands[1] &&
                    this->bands[2] == rhs.bands[2]);
        }

        bool operator == (const BandBoard & rhs) {
            return (this->bands[0] != rhs.bands[0] ||
                    this->bands[1] != rhs.bands[1] ||
                    this->bands[2] != rhs.bands[2]);
        }
#endif // __amd64__
    };

    struct alignas(32) State {
        BandBoard candidates[Numbers10];
        BandBoard prevCandidates[Numbers10];
        BandBoard solvedCells;
        BandBoard solvedRows;
        BandBoard pairs;
        BandBoard reserve;

        State() {
            /* Do nothing !! */
        }

        State(const State & src) {
            this->copy(src);
        }

        ~State() {
            /* Do nothing !! */
        }

        void init() {
#if defined(__AVX2__)
            BitVec16x16_AVX bitset27(kBitSet27_Double64, kBitSet27_Single64, kBitSet27_Double64, kBitSet27_Single64);
            BitVec16x16_AVX bitset27_half(kBitSet27_Double64, kBitSet27_Single64, 0, 0);
            BitVec16x16_AVX zeros;
            zeros.setAllZeros();
            {
                bitset27.saveAligned((void *)&this->candidates[0]);
                bitset27.saveAligned((void *)&this->candidates[2]);
                bitset27.saveAligned((void *)&this->candidates[4]);
                bitset27.saveAligned((void *)&this->candidates[6]);
                bitset27_half.saveAligned((void *)&this->candidates[8]);

                zeros.saveAligned((void *)&this->prevCandidates[0]);                
                zeros.saveAligned((void *)&this->prevCandidates[2]);               
                zeros.saveAligned((void *)&this->prevCandidates[4]);                
                zeros.saveAligned((void *)&this->prevCandidates[6]);
                zeros.saveAligned((void *)&this->prevCandidates[8]);
            }
            zeros.saveAligned((void *)&this->solvedCells);
            zeros.saveAligned((void *)&this->pairs);
#elif defined(__SSE2__)
            BitVec08x16 bitset27(kBitSet27_Double64, kBitSet27_Single64);
            BitVec08x16 zeros;
            zeros.setAllZeros();
            {
                bitset27.saveAligned((void *)&this->candidates[0]);
                bitset27.saveAligned((void *)&this->candidates[1]);
                bitset27.saveAligned((void *)&this->candidates[2]);
                bitset27.saveAligned((void *)&this->candidates[3]);
                bitset27.saveAligned((void *)&this->candidates[4]);
                bitset27.saveAligned((void *)&this->candidates[5]);
                bitset27.saveAligned((void *)&this->candidates[6]);
                bitset27.saveAligned((void *)&this->candidates[7]);
                bitset27.saveAligned((void *)&this->candidates[8]);
                zeros.saveAligned((void *)&this->candidates[9]);

                zeros.saveAligned((void *)&this->prevCandidates[0]);
                zeros.saveAligned((void *)&this->prevCandidates[1]);
                zeros.saveAligned((void *)&this->prevCandidates[2]);
                zeros.saveAligned((void *)&this->prevCandidates[3]);
                zeros.saveAligned((void *)&this->prevCandidates[4]);
                zeros.saveAligned((void *)&this->prevCandidates[5]);
                zeros.saveAligned((void *)&this->prevCandidates[6]);
                zeros.saveAligned((void *)&this->prevCandidates[7]);
                zeros.saveAligned((void *)&this->prevCandidates[8]);
                zeros.saveAligned((void *)&this->prevCandidates[9]);
            }
            zeros.saveAligned((void *)&this->solvedCells);
            zeros.saveAligned((void *)&this->pairs);
#else
            for (size_t num = 0; num < Numbers10; num++) {
                this->candidates[num].full();
                this->prevCandidates[num].clear();
            }
            this->solvedCells.clear();
            this->solvedRows.clear();
            this->pairs.clear();
#endif
        }

        void copy(const State & other) {
#if defined(__AVX2__)
            BitVec16x16_AVX B1, B2, B3, B4;
            BitVec08x16 B5;
            {
                B1.loadAligned((void *)&other.candidates[0]);
                B2.loadAligned((void *)&other.candidates[2]);
                B3.loadAligned((void *)&other.candidates[4]);
                B4.loadAligned((void *)&other.candidates[6]);
                B5.loadAligned((void *)&other.candidates[8]);
                B1.saveAligned((void *)&this->candidates[0]);
                B2.saveAligned((void *)&this->candidates[2]);
                B3.saveAligned((void *)&this->candidates[4]);
                B4.saveAligned((void *)&this->candidates[6]);
                B5.saveAligned((void *)&this->candidates[8]);

                B1.loadAligned((void *)&other.prevCandidates[0]);
                B2.loadAligned((void *)&other.prevCandidates[2]);
                B3.loadAligned((void *)&other.prevCandidates[4]);
                B4.loadAligned((void *)&other.prevCandidates[6]);
                B5.loadAligned((void *)&other.prevCandidates[8]);
                B1.saveAligned((void *)&this->prevCandidates[0]);
                B2.saveAligned((void *)&this->prevCandidates[2]);
                B3.saveAligned((void *)&this->prevCandidates[4]);
                B4.saveAligned((void *)&this->prevCandidates[6]);
                B5.saveAligned((void *)&this->prevCandidates[8]);
            }
            B1.loadAligned((void *)&other.solvedCells);
            B5.loadAligned((void *)&other.pairs);
            B1.saveAligned((void *)&this->solvedCells);
            B5.saveAligned((void *)&this->pairs);
#elif defined(__SSE2__)
            BitVec08x16 B1, B2, B3, B4;
            {
                B1.loadAligned((void *)&other.candidates[0]);
                B2.loadAligned((void *)&other.candidates[1]);
                B3.loadAligned((void *)&other.candidates[2]);
                B4.loadAligned((void *)&other.candidates[3]);
                B1.saveAligned((void *)&this->candidates[0]);
                B2.saveAligned((void *)&this->candidates[1]);
                B3.saveAligned((void *)&this->candidates[2]);
                B4.saveAligned((void *)&this->candidates[3]);

                B1.loadAligned((void *)&other.candidates[4]);
                B2.loadAligned((void *)&other.candidates[5]);
                B3.loadAligned((void *)&other.candidates[6]);
                B4.loadAligned((void *)&other.candidates[7]);
                B1.saveAligned((void *)&this->candidates[4]);
                B2.saveAligned((void *)&this->candidates[5]);
                B3.saveAligned((void *)&this->candidates[6]);
                B4.saveAligned((void *)&this->candidates[7]);

                B1.loadAligned((void *)&other.candidates[8]);
                B1.saveAligned((void *)&this->candidates[8]);

                B1.loadAligned((void *)&other.prevCandidates[0]);
                B2.loadAligned((void *)&other.prevCandidates[1]);
                B3.loadAligned((void *)&other.prevCandidates[2]);
                B4.loadAligned((void *)&other.prevCandidates[3]);
                B1.saveAligned((void *)&this->prevCandidates[0]);
                B2.saveAligned((void *)&this->prevCandidates[1]);
                B3.saveAligned((void *)&this->prevCandidates[2]);
                B4.saveAligned((void *)&this->prevCandidates[3]);

                B1.loadAligned((void *)&other.prevCandidates[4]);
                B2.loadAligned((void *)&other.prevCandidates[5]);
                B3.loadAligned((void *)&other.prevCandidates[6]);
                B4.loadAligned((void *)&other.prevCandidates[7]);
                B1.saveAligned((void *)&this->prevCandidates[4]);
                B2.saveAligned((void *)&this->prevCandidates[5]);
                B3.saveAligned((void *)&this->prevCandidates[6]);
                B4.saveAligned((void *)&this->prevCandidates[7]);

                B1.loadAligned((void *)&other.prevCandidates[8]);
                B1.saveAligned((void *)&this->prevCandidates[8]);
            }
            B1.loadAligned((void *)&other.solvedCells);
            B2.loadAligned((void *)&other.solvedRows);
            B3.loadAligned((void *)&other.pairs);
            B1.saveAligned((void *)&this->solvedCells);
            B2.saveAligned((void *)&this->solvedRows);            
            B3.saveAligned((void *)&this->pairs);
#else
            std::memcpy((void *)this, (const void *)&other, sizeof(State));
#endif
        }
    };

    struct alignas(32) StaticData {
        BandBoard flip_mask[BoardSize + 1];
        BandBoard fill_mask[BoardSize + 1];

        PackedBitSet3D<BoardSize, Rows16, Cols16>   num_row_mask;
        PackedBitSet3D<BoardSize, Rows16, Cols16>   row_fill_mask;
        
        bool mask_is_inited;

        StaticData() : mask_is_inited(false) {
            if (!Static.mask_is_inited) {
                SudokuTable::initialize();
                this_type::init_mask();
                Static.mask_is_inited = true;
            }
        }
    };

#pragma pack(pop)

    int numSolutions_;
    int limitSolutions_;

    State state_;

    static StaticData Static;

public:
    RustV1Solver() : basic_solver(), numSolutions_(0), limitSolutions_(1) {
    }
    ~RustV1Solver() {}

private:
    static void make_flip_mask(size_t fill_pos, size_t row, size_t col) {
        PackedBitSet2D<Rows16, Cols16> & rows_mask = Static.num_row_mask[fill_pos];

        const CellInfo * pCellInfo = SudokuTable::cell_info;
        size_t box = pCellInfo[fill_pos].box;
        size_t cell = pCellInfo[fill_pos].cell;

        Static.row_fill_mask[fill_pos][row].set(col);

        size_t box_x = col / BoxCellsX;
        size_t box_y = row / BoxCellsY;
        size_t box_first_y = box_y * BoxCellsY;
        size_t box_first_x = box_x * BoxCellsX;

        // Literal::NumRowCols
        {
            size_t index = 0;
            // horizontal scanning
            for (size_t x = 0; x < Cols; x++) {
                rows_mask[row].set(x);
                index++;
            }
            // vertical scanning
            for (size_t y = 0; y < Rows; y++) {
                if (y != row) {
                    rows_mask[y].set(col);
                    index++;
                }
            }
            // scanning the current box
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                size_t row_y = box_first_y + cy;
                for (size_t cx = 0; cx < BoxCellsX; cx++) {
                    size_t col_x = box_first_x + cx;
                    rows_mask[row_y].set(col_x);
                    index++;
                }
            }
            assert(index == (Cols + Rows + (BoxCellsY * BoxCellsX) - 1));

            rows_mask[row].reset(col);
        }
    }

    static void transform_to_BandBoard(const PackedBitSet2D<Rows16, Cols16> & bit_mask,
                                       BandBoard & band_mask) {
        static const uint32_t kBoxCountY32 = (uint32_t)BoxCountY;
        static const uint32_t kBoxCellsY32 = (uint32_t)BoxCellsY;
        uint32_t band;
        for (band = 0; band < kBoxCountY32; band++) {
            uint32_t band_bits = 0;
            uint32_t row = band * kBoxCellsY32;
            for (uint32_t cellY = 0; cellY < kBoxCellsY32; cellY++) {
                uint32_t mask = bit_mask[row].value();
                row++;
                band_bits |= mask << (cellY * 9);
            }
            band_mask.bands[band] = band_bits;
        }
        band_mask.bands[band] = 0;
    }

    static void print_rowHiddenSingleMaskTbl() {
        printf("\n");
        printf("static const uint32_t rowHiddenSingleMaskTbl[512] = {\n");
        for (size_t i = 0; i < 512; i++) {
            if ((i % 32) == 0)
                printf("    ");
            uint32_t rowHiddenSingleMask = rowHiddenSingleReverseMaskTbl[i] ^ 0x07U;
            printf("%d", rowHiddenSingleMask);
            if ((i % 32) == 31)
                printf(",\n");
            else
                printf(", ");
        }
        printf("};\n");
        printf("\n");
    }

    static void init_flip_mask() {
        Static.num_row_mask.reset();
        Static.row_fill_mask.reset();

        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_flip_mask(fill_pos, row, col);
                transform_to_BandBoard(Static.num_row_mask[fill_pos], Static.flip_mask[fill_pos]);
                transform_to_BandBoard(Static.row_fill_mask[fill_pos], Static.fill_mask[fill_pos]);
                fill_pos++;
            }
        }
    }

    static void init_mask() {
        if (bPrintSudokuStaticInit) {
            printf("Rust::v1::Solver:StaticData::init_mask()\n");
        }

        init_flip_mask();

        //print_rowHiddenSingleMaskTbl();
    }

    JSTD_FORCED_INLINE
    void extract_solution(State & state, char * solution) {
#if 1
#if defined(_DEBUG)
        for (size_t pos = 0; pos < BoardSize; pos++) {
            solution[pos] = '.';
        }
#endif

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
        for (size_t num = 0; num < Numbers; num++) {
            // Band64: 0
            {
                uint64_t band_bits = state.candidates[num].bands64[0];
                while (band_bits != 0) {
                    uint32_t bit_pos = BitUtils::bsf64(band_bits);
                    uint64_t mask = BitUtils::ls1b64(band_bits);
                    band_bits ^= mask;

                    size_t pos = bandBitPosToPos64[0][bit_pos];
                    assert(pos != size_t(-1));

                    assert(solution[pos] == '.');
                    solution[pos] = (char)('1' + num);
                }
            }

            // Band64: 1
            {
                uint64_t band_bits = state.candidates[num].bands64[1];
                while (band_bits != 0) {
                    uint32_t bit_pos = BitUtils::bsf64(band_bits);
                    uint64_t mask = BitUtils::ls1b64(band_bits);
                    band_bits ^= mask;

                    size_t pos = bandBitPosToPos64[1][bit_pos];
                    assert(pos != size_t(-1));

                    assert(solution[pos] == '.');
                    solution[pos] = (char)('1' + num);
                }
            }
        }
#else
        for (size_t num = 0; num < Numbers; num++) {
            // Band 0
            {
                uint32_t band_bits = state.candidates[num].bands[0];
                while (band_bits != 0) {
                    uint32_t bit_pos = BitUtils::bsf32(band_bits);
                    uint32_t mask = BitUtils::ls1b32(band_bits);
                    band_bits ^= mask;

                    size_t pos = bandBitPosToPos32[0][bit_pos];
                    assert(pos != size_t(-1));

                    assert(solution[pos] == '.');
                    solution[pos] = (char)('1' + num);
                }
            }

            // Band 1
            {
                uint32_t band_bits = state.candidates[num].bands[1];
                while (band_bits != 0) {
                    uint32_t bit_pos = BitUtils::bsf32(band_bits);
                    uint32_t mask = BitUtils::ls1b32(band_bits);
                    band_bits ^= mask;

                    size_t pos = bandBitPosToPos32[1][bit_pos];
                    assert(pos != size_t(-1));

                    assert(solutions[pos] == '.');
                    solution[pos] = (char)('1' + num);
                }
            }

            // Band 2
            {
                uint32_t band_bits = state.candidates[num].bands[2];
                while (band_bits != 0) {
                    uint32_t bit_pos = BitUtils::bsf32(band_bits);
                    uint32_t mask = BitUtils::ls1b32(band_bits);
                    band_bits ^= mask;

                    size_t pos = bandBitPosToPos32[2][bit_pos];
                    assert(pos != size_t(-1));

                    assert(solution[pos] == '.');
                    solution[pos] = (char)('1' + num);
                }
            }
        }
#endif
#else
        for (size_t pos = 0; pos < BoardSize; pos++) {
            uint32_t mask = tables.posToMask[pos];
            uint32_t band = tables.div27[pos];
            for (size_t num = 0; num < Numbers; num++) {
                if ((state.candidates[num].bands[band] & mask) != 0) {
                    solution[pos] = (char)('1' + num);
                    break;
                }
            }
        }
#endif
    }

    JSTD_FORCED_INLINE
    int init_sudoku(State & state, const char * puzzle) {
        state.init();

        BitVec08x16 solved_cells;
        solved_cells.setAllZeros();

        int candidates = 0;
        for (size_t pos = 0; pos < BoardSize; pos++) {
            unsigned char val = puzzle[pos];
            if (val != '.') {
                size_t num = val - '1';
                assert(num >= (Sudoku::MinNumber - 1) && num <= (Sudoku::MaxNumber - 1));
                int validity = this->update_peer_cells(state, solved_cells, pos, num);
                if (validity == Status::Invalid)
                    return -1;
                candidates++;
            }
        }

        void * pCells16 = (void *)&state.solvedCells;
        solved_cells.saveAligned(pCells16);

        return candidates;
    }

    inline int update_peer_cells(State & state, BitVec08x16 & solved_cells, size_t fill_pos, size_t fill_num) {
        assert(fill_pos < Sudoku::BoardSize);
        assert(fill_num >= (Sudoku::MinNumber - 1) && fill_num <= (Sudoku::MaxNumber - 1));

        BitVec08x16 cells16, mask16;
        void * pCells16, * pMask16;

        BitVec08x16 candidates;
        pCells16 = (void *)&state.candidates[fill_num];
        candidates.loadAligned(pCells16);

        BitVec08x16 fill_mask;
        pMask16 = (void *)&Static.fill_mask[fill_pos];
        fill_mask.loadAligned(pMask16);

        BitVec08x16 verify_bit = candidates & fill_mask;
        if (verify_bit.isAllZeros())
            return Status::Invalid;

        size_t rowBit = fill_num * Rows + tables.div9[fill_pos];
        uint32_t band = tables.div27[rowBit];
        uint32_t shift = tables.mod27[rowBit];
        state.solvedRows.bands[band] |= 1U << shift;

        solved_cells |= fill_mask;

        for (size_t num = 0; num < Numbers; num++) {
            pCells16 = (void *)&state.candidates[num];
            cells16.loadAligned(pCells16);
            cells16.andnot_equal(fill_mask);
            cells16.saveAligned(pCells16);
        }

        pMask16 = (void *)&Static.flip_mask[fill_pos];
        mask16.loadAligned(pMask16);
        candidates.andnot_equal(mask16);
        candidates.or_equal(fill_mask);
        candidates.saveAligned((void *)&state.candidates[fill_num]);

        return Status::Success;
    }

    inline void update_peer_cells(State & state, size_t fill_pos, size_t fill_num) {
        assert(fill_pos < Sudoku::BoardSize);
        assert(fill_num >= (Sudoku::MinNumber - 1) && fill_num <= (Sudoku::MaxNumber - 1));

        size_t rowBit = fill_num * Rows + tables.div9[fill_pos];
        uint32_t band = tables.div27[rowBit];
        uint32_t shift = tables.mod27[rowBit];
        state.solvedRows.bands[band] |= 1U << shift;

        BitVec08x16 cells16, mask16;
        void * pCells16, * pMask16;

        BitVec08x16 fill_mask, solved_cells;
        pCells16 = (void *)&state.solvedCells;
        pMask16 = (void *)&Static.fill_mask[fill_pos];
        solved_cells.loadAligned(pCells16);
        fill_mask.loadAligned(pMask16);
        solved_cells |= fill_mask;
        solved_cells.saveAligned(pCells16);

        for (size_t num = 0; num < Numbers; num++) {
            pCells16 = (void *)&state.candidates[num];
            cells16.loadAligned(pCells16);
            cells16.andnot_equal(fill_mask);
            cells16.saveAligned(pCells16);
        }

        pCells16 = (void *)&state.candidates[fill_num];
        pMask16 = (void *)&Static.flip_mask[fill_pos];
        cells16.loadAligned(pCells16);
        mask16.loadAligned(pMask16);
        cells16.andnot_equal(mask16);
        cells16.or_equal(fill_mask);
        cells16.saveAligned(pCells16);
    }

    JSTD_FORCED_INLINE
    void update_band_solved_mask32(State & state, size_t band, size_t pos, size_t num) {
        state.candidates[num].bands[band] &= bandUnsolvedMaskTbl32[pos];
    }

    JSTD_FORCED_INLINE
    void update_band_solved_mask64(State & state, size_t band, size_t pos, size_t num) {
        state.candidates[num].bands64[band] &= bandUnsolvedMaskTbl64[pos];
    }

    JSTD_FORCED_INLINE
    void update_band_solved_mask32(State & state, size_t band, size_t pos, size_t num, uint32_t mask) {
        if ((state.candidates[num].bands[band] & mask) != 0) {
            state.candidates[num].bands[band] &= bandUnsolvedMaskTbl32[pos];
        }
    }

    JSTD_FORCED_INLINE
    void update_band_solved_mask64(State & state, size_t band, size_t pos, size_t num, uint64_t mask) {
        if ((state.candidates[num].bands64[band] & mask) != 0) {
            state.candidates[num].bands64[band] &= bandUnsolvedMaskTbl64[pos];
        }
    }

    template <uint32_t digit, uint32_t self, uint32_t peer1, uint32_t peer2, bool fast_mode>
    JSTD_FORCED_INLINE
    uint32_t update_up_down_cells(State & state, uint32_t & band) {
        uint32_t rowTriadsMask = rowTriadsMaskTbl[band & kFullRowBits] |
                                (rowTriadsMaskTbl[(band >> 9U) & kFullRowBits] << 3U) |
                                (rowTriadsMaskTbl[(band >> 18U) & kFullRowBits] << 6U);
        uint32_t lockedCandidates = keepLockedCandidatesTbl[rowTriadsMask];
        uint32_t newBand = band & lockedCandidates;
        if (fast_mode || newBand != 0) {
            assert(newBand != 0);
            state.candidates[digit].bands[self] = newBand;
            state.prevCandidates[digit].bands[self] = newBand;
            band = newBand;

            uint32_t possible_cols = (newBand | (newBand >> 9U) | (newBand >> 18U)) & kFullRowBits;
            uint32_t nonconflicting_other = nonconflicting_cells_neighbor_bands_by_locked_candidates[possible_cols];
            state.candidates[digit].bands[peer1] &= nonconflicting_other;
            state.candidates[digit].bands[peer2] &= nonconflicting_other;
            uint32_t locked_candidates_intersection = rowTriadsSingleMaskTbl[rowTriadsMask] & combColumnSingleMaskTbl[possible_cols];
            uint32_t newSolvedRows = rowHiddenSingleMaskTbl[locked_candidates_intersection];
            return newSolvedRows;
        }
        else {
            return Status::Unsolvable;
        }
    }

    template <uint32_t digit, uint32_t self>
    JSTD_FORCED_INLINE
    void update_solved_rows(State & state, uint32_t band, uint32_t bandSolvedRows) {
        uint32_t solvedCells = band & solvedRowsBitMaskTbl[bandSolvedRows];
        //assert(solvedCells != 0);
        state.solvedCells.bands[self] |= solvedCells;
        uint32_t nonconflictingCells = ~solvedCells;
        if (digit != 0)
            state.candidates[0].bands[self] &= nonconflictingCells;
        if (digit != 1)
            state.candidates[1].bands[self] &= nonconflictingCells;
        if (digit != 2)
            state.candidates[2].bands[self] &= nonconflictingCells;
        if (digit != 3)
            state.candidates[3].bands[self] &= nonconflictingCells;
        if (digit != 4)
            state.candidates[4].bands[self] &= nonconflictingCells;
        if (digit != 5)
            state.candidates[5].bands[self] &= nonconflictingCells;
        if (digit != 6)
            state.candidates[6].bands[self] &= nonconflictingCells;
        if (digit != 7)
            state.candidates[7].bands[self] &= nonconflictingCells;
        if (digit != 8)
            state.candidates[8].bands[self] &= nonconflictingCells;
    }

    template <uint32_t digit, uint32_t self, uint32_t peer1, uint32_t peer2, uint32_t shift, bool fast_mode>
    JSTD_FORCED_INLINE
    uint32_t find_and_update_band(State & state, uint32_t band, uint32_t & solvedRows) {
        uint32_t rowTriadsMask = rowTriadsMaskTbl[band & kFullRowBits] |
                                (rowTriadsMaskTbl[(band >> 9U) & kFullRowBits] << 3U) |
                                (rowTriadsMaskTbl[(band >> 18U) & kFullRowBits] << 6U);
        uint32_t lockedCandidates = keepLockedCandidatesTbl[rowTriadsMask];
        uint32_t newBand = band & lockedCandidates;
        if (fast_mode || newBand != 0) {
            assert(newBand != 0);
            state.candidates[digit].bands[self] = newBand;
            state.prevCandidates[digit].bands[self] = newBand;

            uint32_t possible_cols = (newBand | (newBand >> 9U) | (newBand >> 18U)) & kFullRowBits;
            uint32_t nonconflicting_other = nonconflicting_cells_neighbor_bands_by_locked_candidates[possible_cols];
            state.candidates[digit].bands[peer1] &= nonconflicting_other;
            state.candidates[digit].bands[peer2] &= nonconflicting_other;

            uint32_t locked_candidates_intersection = rowTriadsSingleMaskTbl[rowTriadsMask] &
                                                      combColumnSingleMaskTbl[possible_cols];
            uint32_t bandSolvedRows = rowHiddenSingleMaskTbl[locked_candidates_intersection];
            uint32_t newSolvedRows = bandSolvedRows << shift;
            //if ((solvedRows & (0x007U << shift)) != newSolvedRows) {
                solvedRows |= newSolvedRows;
                this->update_solved_rows<digit, self>(state, newBand, bandSolvedRows);
            //}
            return Status::Success;
        }
        else {
            return Status::Unsolvable;
        }
    }

    template <bool fast_mode = false>
    JSTD_NO_INLINE
    int find_locked_candidates_and_update(State & state) {
        bool found_nothing;

        do {
            register uint32_t solvedRows;
            found_nothing = true;

            /********* Number 1-3 Start *********/
        
            // Number 1
            solvedRows = state.solvedRows.bands[0];        
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits) != kFullRowBits) {
                static const uint32_t digit = 0;

                // Number 1 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 0, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 1 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 3, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 1 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 6, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 2
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_1) != kFullRowBits_1) {
                static const uint32_t digit = 1;

                // Number 2 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 9, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 2 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 12, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 2 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 15, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 3
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_2) != kFullRowBits_2) {
                static const uint32_t digit = 2;

                // Number 3 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 18, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 3 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 21, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 3 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 24, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            state.solvedRows.bands[0] = solvedRows;

            /********* Number 1-3 End *********/

            /********* Number 4-6 Start *********/
        
            // Number 4
            solvedRows = state.solvedRows.bands[1];        
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits) != kFullRowBits) {
                static const uint32_t digit = 3;

                // Number 4 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 0, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 4 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 3, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 4 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 6, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 5
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_1) != kFullRowBits_1) {
                static const uint32_t digit = 4;

                // Number 5 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 9, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 5 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 12, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 5 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 15, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 6
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_2) != kFullRowBits_2) {
                static const uint32_t digit = 5;

                // Number 6 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 18, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 6 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 21, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 6 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 24, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            state.solvedRows.bands[1] = solvedRows;

            /********* Number 4-6 End *********/

            /********* Number 7-9 Start *********/
        
            // Number 7
            solvedRows = state.solvedRows.bands[2];        
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits) != kFullRowBits) {
                static const uint32_t digit = 6;

                // Number 7 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 0, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 7 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 3, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 7 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 6, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 8
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_1) != kFullRowBits_1) {
                static const uint32_t digit = 7;

                // Number 8 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 9, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 8 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 12, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 8 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 15, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            // Number 9
            if (!kCheckSolvedRows || (solvedRows & kFullRowBits_2) != kFullRowBits_2) {
                static const uint32_t digit = 8;

                // Number 9 - band 0
                register uint32_t band = state.candidates[digit].bands[0];
                if (band != state.prevCandidates[digit].bands[0]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 0, 1, 2, 18, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 9 - band 1
                band = state.candidates[digit].bands[1];
                if (band != state.prevCandidates[digit].bands[1]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 1, 0, 2, 21, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }

                // Number 9 - band 2
                band = state.candidates[digit].bands[2];
                if (band != state.prevCandidates[digit].bands[2]) {
                    found_nothing = false;
                    int status = this->find_and_update_band<digit, 2, 0, 1, 24, fast_mode>(state, band, solvedRows);
                    if (status == Status::Unsolvable)
                        return Status::Unsolvable;
                }
            }

            state.solvedRows.bands[2] = solvedRows;

            /********* Number 7-9 End *********/
        } while (!found_nothing);

        return Status::Success;
    }

    JSTD_NO_INLINE
    int fast_find_naked_singles(State & state) {
#if 1
        BitVec08x16 R1, R2;
        BitVec08x16 band_bits;

        // 0
        R1.loadAligned((void *)&state.candidates[0]);
        //R2.setAllZeros();

        // 1
        band_bits.loadAligned((void *)&state.candidates[1]);
        R2 = R1 & band_bits;
        R1 |= band_bits;

        // 2
        band_bits.loadAligned((void *)&state.candidates[2]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 3
        band_bits.loadAligned((void *)&state.candidates[3]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 4
        band_bits.loadAligned((void *)&state.candidates[4]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 5
        band_bits.loadAligned((void *)&state.candidates[5]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 6
        band_bits.loadAligned((void *)&state.candidates[6]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 7
        band_bits.loadAligned((void *)&state.candidates[7]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 8
        band_bits.loadAligned((void *)&state.candidates[8]);
        R2 |= R1 & band_bits;
        R1 |= band_bits;
#else
        BitVec08x16 R1, R2;

        void * pCells16 = (void *)&state.candidates[0];
        R1.loadAligned(pCells16);
        R2.setAllZeros();

        for (size_t num = 1; num < Numbers; num++) {
            BitVec08x16 band_bits;
            pCells16 = (void *)&state.candidates[num];
            band_bits.loadAligned(pCells16);

            R2 |= R1 & band_bits;
            R1 |= band_bits;
        }
#endif
        BitVec08x16 full_mask(kBitSet27, kBitSet27, kBitSet27, 0);
        bool is_legal = R1.isEqual(full_mask);
        assert(is_legal);
        //if (!is_legal) return Status::Unsolvable;

        BitVec08x16 solved_bits;
        solved_bits.loadAligned((void *)&state.solvedCells);

        R1.andnot_equal(R2);
        R1.andnot_equal(solved_bits);

        if (R1.isAllZeros()) {
            return 0;
        }
        else {
            int cell_count = 0;

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
            register uint64_t bits64 = R1.extractU64<0>();
            if (bits64 != 0) {
                do {
                    uint32_t bit_pos = BitUtils::bsf64(bits64);
                    uint64_t bit = BitUtils::ls1b64(bits64);
                    bits64 ^= bit;

                    uint32_t pos = bandBitPosToPos64[0][bit_pos];
                    assert(pos != uint32_t(int8_t(-1)));

                    for (size_t num = 0; num < Numbers; num++) {
                        uint64_t band_bits = state.candidates[num].bands64[0];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask64(state, 0, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                } while (bits64 != 0);

                bits64 = R1.extractU64<1>();
                if (bits64 != 0)
                    goto Band64_01_Loop;
            }
            else {
                bits64 = R1.extractU64<1>();
Band64_01_Loop:
                do {
                    uint32_t bit_pos = BitUtils::bsf64(bits64);
                    uint64_t bit = BitUtils::ls1b64(bits64);
                    bits64 ^= bit;

                    uint32_t pos = bandBitPosToPos64[1][bit_pos];
                    assert(pos != uint32_t(int8_t(-1)));

                    for (size_t num = 0; num < Numbers; num++) {
                        uint64_t band_bits = state.candidates[num].bands64[1];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask64(state, 1, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                } while (bits64 != 0);
            }
#else // !__amd64__
            BandBoard R1_bits;
            R1.saveAligned((void *)&R1_bits);

            for (size_t band = 0; band < 3; band++) {
                register uint32_t bits32 = R1_bits.bands[band];
                while (bits32 != 0) {
                    uint32_t bit_pos = BitUtils::bsf32(bits32);
                    uint32_t bit = BitUtils::ls1b32(bits32);
                    bits32 ^= bit;

                    uint32_t pos = bandBitPosToPos32[band][bit_pos];
                    assert(pos != uint32_t(-1));

                    for (size_t num = 0; num < Numbers; num++) {
                        uint32_t band_bits = state.candidates[num].bands[band];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask32(state, band, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                }
            }
#endif // __amd64__

            return cell_count;
        }
    }

    JSTD_NO_INLINE
    int normal_find_naked_singles(State & state) {
#if 1
        BitVec08x16 R1, R2, R3;
        BitVec08x16 band_bits;

        // 0
        R1.loadAligned((void *)&state.candidates[0]);
        //R2.setAllZeros();
        //R3.setAllZeros();

        // 1
        band_bits.loadAligned((void *)&state.candidates[1]);
        R2 = R1 & band_bits;
        R1 |= band_bits;

        // 2
        band_bits.loadAligned((void *)&state.candidates[2]);
        R3 = R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 3
        band_bits.loadAligned((void *)&state.candidates[3]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 4
        band_bits.loadAligned((void *)&state.candidates[4]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 5
        band_bits.loadAligned((void *)&state.candidates[5]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 6
        band_bits.loadAligned((void *)&state.candidates[6]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 7
        band_bits.loadAligned((void *)&state.candidates[7]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;

        // 8
        band_bits.loadAligned((void *)&state.candidates[8]);
        R3 |= R2 & band_bits;
        R2 |= R1 & band_bits;
        R1 |= band_bits;
#else
        BitVec08x16 R1, R2, R3;

        R1.loadAligned((void *)&state.candidates[0]);
        R2.setAllZeros();
        R3.setAllZeros();

        for (size_t num = 1; num < Numbers; num++) {
            BitVec08x16 band_bits;
            band_bits.loadAligned((void *)&state.candidates[num]);

            R3 |= R2 & band_bits;
            R2 |= R1 & band_bits;
            R1 |= band_bits;
        }
#endif
        BitVec08x16 full_mask(kBitSet27, kBitSet27, kBitSet27, 0);
        bool is_legal = R1.isEqual(full_mask);
        if (!is_legal)
            return Status::Unsolvable;

        BitVec08x16 solved_bits;
        solved_bits.loadAligned((void *)&state.solvedCells);

        R1.andnot_equal(R2);
        R1.andnot_equal(solved_bits);

        if (R1.isAllZeros()) {
            R2.andnot_equal(R3);
            R2.saveAligned((void *)&state.pairs);
            return 0;
        }
        else {
            int cell_count = 0;

#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
            register uint64_t bits64 = R1.extractU64<0>();
            if (bits64 != 0) {
                do {
                    uint32_t bit_pos = BitUtils::bsf64(bits64);
                    uint64_t bit = BitUtils::ls1b64(bits64);
                    bits64 ^= bit;

                    uint32_t pos = bandBitPosToPos64[0][bit_pos];
                    assert(pos != uint32_t(int8_t(-1)));

                    size_t num;
                    for (num = 0; num < Numbers; num++) {
                        uint64_t band_bits = state.candidates[num].bands64[0];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask64(state, 0, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                    if (num == Numbers)
                        return Status::Unsolvable;
                } while (bits64 != 0);

                bits64 = R1.extractU64<1>();
                if (bits64 != 0)
                    goto Band64_01_Loop;
            }
            else {
                bits64 = R1.extractU64<1>();
Band64_01_Loop:
                do {
                    size_t bit_pos = BitUtils::bsf64(bits64);
                    uint64_t bit = BitUtils::ls1b64(bits64);
                    bits64 ^= bit;

                    size_t pos = bandBitPosToPos64[1][bit_pos];
                    assert(pos != uint32_t(int8_t(-1)));

                    size_t num;
                    for (num = 0; num < Numbers; num++) {
                        uint64_t band_bits = state.candidates[num].bands64[1];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask64(state, 1, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                    if (num == Numbers)
                        return Status::Unsolvable;
                } while (bits64 != 0);
            }
#else // !__amd64__
            BandBoard R1_bits;
            R1.saveAligned((void *)&R1_bits);

            for (size_t band = 0; band < 3; band++) {
                register uint32_t bits32 = R1_bits.bands[band];
                while (bits32 != 0) {
                    size_t bit_pos = BitUtils::bsf32(bits32);
                    uint32_t bit = BitUtils::ls1b32(bits32);
                    bits32 ^= bit;

                    size_t pos = bandBitPosToPos32[band][bit_pos];
                    assert(pos != size_t(-1));

                    size_t num;
                    for (num = 0; num < Numbers; num++) {
                        uint32_t band_bits = state.candidates[num].bands[band];
                        if ((band_bits & bit) != 0) {
                            this->update_band_solved_mask32(state, band, pos, num);
                            cell_count++;
                            break;
                        }
                    }
                    if (num == Numbers)
                        return Status::Unsolvable;
                }
            }
#endif // __amd64__

            return cell_count;
        }
    }

    JSTD_NO_INLINE
    int guess_bivalue_cells(State & state, char * solution) {
#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
        for (size_t band = 0; band < 2; band++) {
            uint64_t pairs = state.pairs.bands64[band];
            if (pairs != 0) {
                uint32_t bit_pos = BitUtils::bsf64(pairs);
                uint64_t mask = BitUtils::ls1b64(pairs);

                uint32_t pos = bandBitPosToPos64[band][bit_pos];
                assert(pos != uint32_t(-1));

                int tries = 2;
                for (size_t num = 0; num < Numbers; num++) {
                    if ((state.candidates[num].bands64[band] & mask) != 0) {
                        if (--tries) {
                            // The first of pair
                            State next_state(state);
                            state.candidates[num].bands64[band] ^= mask;
                            basic_solver::num_guesses++;

                            this->update_band_solved_mask64(next_state, band, pos, num);

                            if (this->search<false>(next_state) != Status::Unsolvable) {
                                this->guess_next_cell(next_state, solution);

                                if (kReachSolutionsLimitToExit && this->numSolutions_ >= this->limitSolutions_)
                                    return Status::Success;
                            }
                        }
                        else {
                            // The second of pair
                            this->update_band_solved_mask64(state, band, pos, num);

                            if (this->search<false>(state) != Status::Unsolvable) {
                                this->guess_next_cell(state, solution);
                            }
                            return Status::Success;
                        }
                    }
                }
            }
        }
#else // !__amd64__
        for (size_t band = 0; band < 3; band++) {
            uint32_t pairs = state.pairs.bands[band];
            if (pairs != 0) {
                uint32_t bit_pos = BitUtils::bsf32(pairs);
                uint32_t mask = BitUtils::ls1b32(pairs);

                uint32_t pos = bandBitPosToPos32[band][bit_pos];
                assert(pos != uint32_t(-1));

                int tries = 2;
                for (size_t num = 0; num < Numbers; num++) {
                    if ((state.candidates[num].bands[band] & mask) != 0) {
                        if (--tries) {
                            // The first of pair
                            State next_state(state);
                            state.candidates[num].bands[band] ^= mask;
                            basic_solver::num_guesses++;

                            this->update_band_solved_mask32(next_state, band, pos, num);

                            if (this->search<false>(next_state) != Status::Unsolvable) {
                                this->guess_next_cell(next_state, solution);

                                if (kReachSolutionsLimitToExit && this->numSolutions_ >= this->limitSolutions_)
                                    return Status::Success;
                            }
                        }
                        else {
                            // The second of pair
                            this->update_band_solved_mask32(state, band, pos, num);

                            if (this->search<false>(state) != Status::Unsolvable) {
                                this->guess_next_cell(state, solution);
                            }
                            return Status::Success;
                        }
                    }
                }
            }
        }
#endif // __amd64__
        return Status::Failed;
    }

    JSTD_NO_INLINE
    int guess_first_cell(State & state, char * solution) {
#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
        // Band64: 0
        uint64_t unsolvedCells = state.solvedCells.bands64[0] ^ kBitSet27_Double64;
        if (unsolvedCells != 0) {
            uint32_t bit_pos = BitUtils::bsf64(unsolvedCells);
            uint64_t mask = BitUtils::ls1b64(unsolvedCells);

            uint32_t pos = bandBitPosToPos64[0][bit_pos];
            assert(pos != uint32_t(-1));

            for (size_t num = 0; num < Numbers; num++) {
                if ((state.candidates[num].bands64[0] & mask) != 0) {
                    State next_state(state);
                    state.candidates[num].bands64[0] ^= mask;
                    basic_solver::num_guesses++;

                    this->update_band_solved_mask64(next_state, 0, pos, num);

                    if (this->search<false>(next_state) != Status::Unsolvable) {
                        this->guess_next_cell(next_state, solution);

                        if (kReachSolutionsLimitToExit && this->numSolutions_ >= this->limitSolutions_)
                            return Status::Success;
                    }
                }
            }

            return Status::Success;
        }

        unsolvedCells = state.solvedCells.bands64[1] ^ kBitSet27_Single64;
        if (unsolvedCells != 0) {
            uint32_t bit_pos = BitUtils::bsf64(unsolvedCells);
            uint64_t mask = BitUtils::ls1b64(unsolvedCells);

            uint32_t pos = bandBitPosToPos64[1][bit_pos];
            assert(pos != uint32_t(-1));

            for (size_t num = 0; num < Numbers; num++) {
                if ((state.candidates[num].bands64[1] & mask) != 0) {
                    State next_state(state);
                    state.candidates[num].bands64[1] ^= mask;
                    basic_solver::num_guesses++;

                    this->update_band_solved_mask64(next_state, 1, pos, num);

                    if (this->search<false>(next_state) != Status::Unsolvable) {
                        this->guess_next_cell(next_state, solution);

                        if (kReachSolutionsLimitToExit && this->numSolutions_ >= this->limitSolutions_)
                            return Status::Success;
                    }
                }
            }

            return Status::Success;
        }
#else
        for (size_t band = 0; band < 3; band++) {
            uint32_t unsolvedCells = state.solvedCells.bands[band] ^ kBitSet27;
            if (unsolvedCells == 0)
                continue;

            uint32_t bit_pos = BitUtils::bsf32(unsolvedCells);
            uint32_t mask = BitUtils::ls1b32(unsolvedCells);

            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(-1));

            for (size_t num = 0; num < Numbers; num++) {
                if ((state.candidates[num].bands[band] & mask) != 0) {
                    State next_state(state);
                    state.candidates[num].bands[band] ^= mask;
                    basic_solver::num_guesses++;

                    this->update_band_solved_mask32(next_state, band, pos, num);

                    if (this->search<false>(next_state) != Status::Unsolvable) {
                        this->guess_next_cell(next_state, solution);

                        if (kReachSolutionsLimitToExit && this->numSolutions_ >= this->limitSolutions_)
                            return Status::Success;
                    }
                }
            }

            return Status::Success;
        }
#endif
        return Status::Failed;
    }

    JSTD_FORCED_INLINE
    int guess_three_candidates_cell(State & state, char * solution, uint32_t band, uint32_t pos, uint32_t mask) {
        int tries = 3;
        for (size_t num = 0; num < Numbers; num++) {
            if ((state.candidates[num].bands[band] & mask) != 0) {
                if (--tries) {
                    // The first and second digit
                    State next_state(state);
                    state.candidates[num].bands[band] ^= mask;
                    basic_solver::num_guesses++;

                    this->update_band_solved_mask32(next_state, band, pos, num);

                    if (this->search<false>(next_state) != Status::Unsolvable) {
                        this->guess_next_cell(next_state, solution);

                        if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                            return Status::Success;
                    }
                }
                else {
                    // The last digit
                    this->update_band_solved_mask32(state, band, pos, num);

                    if (this->search<false>(state) != Status::Unsolvable) {
                        this->guess_next_cell(state, solution);
                    }
                    return Status::Success;
                }
            }
        }

        return Status::Failed;
    }

    JSTD_FORCED_INLINE
    int guess_more_than_4_cell(State & state, char * solution, uint32_t band, uint32_t pos, int candidates) {
        int tries = candidates;
        uint32_t mask = tables.posToMask[pos];
        for (size_t num = 0; num < Numbers; num++) {
            if ((state.candidates[num].bands[band] & mask) != 0) {
                if (--tries) {
                    // The front of digits
                    State next_state(state);
                    state.candidates[num].bands[band] ^= mask;
                    basic_solver::num_guesses++;

                    this->update_band_solved_mask32(next_state, band, pos, num);

                    if (this->search<false>(next_state) != Status::Unsolvable) {
                        this->guess_next_cell(next_state, solution);

                        if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                            return Status::Success;
                    }
                }
                else {
                    // The last digit
                    this->update_band_solved_mask32(state, band, pos, num);

                    if (this->search<false>(state) != Status::Unsolvable) {
                        return this->guess_next_cell(state, solution);
                    }
                    return Status::Success;
                }
            }
        }

        return Status::Failed;
    }

    template <uint32_t v1, uint32_t v2, uint32_t v3>
    inline
    uint32_t make_band_order() {
#if GZ_SUDOKU_ENDIAN == GZ_LITTLE_ENDIAN
        uint32_t order = v1 | (v2 << 8U) | (v3 << 16U);
#else
        uint32_t order = (v1 << 24U) | (v2 << 16U) | (v3 << 8U);
#endif
        return order;
    }

    JSTD_NO_INLINE
    int guess_some_cell(State & state, char * solution) {
#if 0
        uint8_t band_order[4];
        uint32_t band_solved_0 = BitUtils::popcnt32(state.solvedCells.bands[0]);
        uint32_t band_solved_1 = BitUtils::popcnt32(state.solvedCells.bands[1]);
        uint32_t band_solved_2 = BitUtils::popcnt32(state.solvedCells.bands[2]);
        uint32_t * order = (uint32_t *)&band_order[0];
        if (band_solved_0 >= band_solved_1) {
            if (band_solved_0 >= band_solved_2) {
                *order = make_band_order<0, 1, 2>();
                /*
                if (band_solved_1 >= band_solved_2)
                    *order = make_band_order<0, 1, 2>();
                else
                    *order = make_band_order<0, 2, 1>();
                //*/
            }
            else {
                // band_solved_0 < band_solved_2
                *order = make_band_order<2, 0, 1>();
            }
        }
        else {
            // band_solved_0 < band_solved_1
            if (band_solved_0 >= band_solved_2) {
                *order = make_band_order<1, 0, 2>();
            }
            else {
                // band_solved_0 < band_solved_2
                *order = make_band_order<1, 2, 0>();
                /*
                if (band_solved_1 >= band_solved_2)
                    *order = make_band_order<1, 2, 0>();
                else
                    *order = make_band_order<2, 1, 0>();
                //*/
            }
        }
#endif
        uint32_t min_unsolved_cnt = (uint32_t)-1;
        uint32_t min_unsolved_pos;
        uint32_t min_unsolved_band;

        UNUSED_VARIABLE(min_unsolved_pos);
        UNUSED_VARIABLE(min_unsolved_band);

        for (uint32_t i = 0; i < 3; i++) {
            //uint32_t band = band_order[i];
            uint32_t band = i;
            uint32_t unsolvedCells = state.solvedCells.bands[band] ^ kBitSet27;
            if (unsolvedCells == 0)
                continue;

            uint32_t bit_pos = BitUtils::bsf32(unsolvedCells);
            uint32_t mask = BitUtils::ls1b32(unsolvedCells);

            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(-1));

            uint32_t unsolved_cnt = 0;
            for (size_t num = 0; num < Numbers; num++) {
                if ((state.candidates[num].bands[band] & mask) != 0) {
                    unsolved_cnt++;
                }
            }

            assert(unsolved_cnt >= 3);
            if (unsolved_cnt == 3) {
                return this->guess_three_candidates_cell(state, solution, band, pos, mask);
            }
            else if (unsolved_cnt < min_unsolved_cnt) {
                min_unsolved_cnt = unsolved_cnt;
                min_unsolved_pos = pos;
                min_unsolved_band = band;
            }
        }

        if (min_unsolved_cnt != (uint32_t)-1) {
            return this->guess_more_than_4_cell(state, solution, min_unsolved_band, min_unsolved_pos, min_unsolved_cnt);
        }
        else {
            return Status::Failed;
        }
    }

    JSTD_FORCED_INLINE
    int guess_hidden_box_pair(State & state, char * solution, uint32_t digit, uint32_t box) {
        assert(digit >= 0 && digit < Numbers);
        assert(box >= 0 && box < Boxes);
        uint32_t band = tables.div3[box];
        uint32_t box_bits = state.candidates[digit].bands[band] & boxToBoxesMaskTbl[box];
#if 0
        int tries = 2;
        while (box_bits != 0) {
            size_t bit_pos = BitUtils::bsf32(box_bits);
            uint32_t mask = BitUtils::ls1b32(box_bits);
            box_bits ^= mask;

            size_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != size_t(-1));
            if (--tries) {
                // The first of pair
                State next_state(state);
                state.candidates[digit].bands[band] ^= mask;
                basic_solver::num_guesses++;

                this->update_band_solved_mask32(next_state, band, pos, digit);

                if (this->search<false>(next_state) != Status::Unsolvable) {
                    this->guess_next_cell(next_state, solution);

                    if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                        return Status::Success;
                }
            }
            else {
                // The second of pair
                this->update_band_solved_mask32(state, band, pos, digit);

                if (this->search<false>(state) != Status::Unsolvable) {
                    this->guess_next_cell(state, solution);
                }
                return Status::Success;
            }
        }

        return Status::Failed;
#else
        {
            // The first of row bi-value
            assert (box_bits != 0);
            uint32_t bit_pos = BitUtils::bsf32(box_bits);
            uint32_t mask = BitUtils::ls1b32(box_bits);
            box_bits ^= mask;

            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(int8_t(-1)));

            State next_state(state);
            state.candidates[digit].bands[band] ^= mask;
            basic_solver::num_guesses++;

            this->update_band_solved_mask32(next_state, band, pos, digit);

            if (this->search<false>(next_state) != Status::Unsolvable) {
                this->guess_next_cell(next_state, solution);

                if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                    return Status::Success;
            }
        }

        {
            // The second of row bi-value
            assert (box_bits != 0);
            uint32_t bit_pos = BitUtils::bsf32(box_bits);
            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(int8_t(-1)));

            this->update_band_solved_mask32(state, band, pos, digit);

            if (this->search<false>(state) != Status::Unsolvable) {
                this->guess_next_cell(state, solution);
            }
            return Status::Success;
        }
#endif
    }

    JSTD_FORCED_INLINE
    int guess_box_cell_more_than_2(State & state, char * solution, uint32_t digit, uint32_t box) {
        assert(digit >= 0 && digit < (uint32_t)Numbers);
        assert(box >= 0 && box < (uint32_t)Boxes);
        uint32_t band = tables.div3[box];
        uint32_t box_bits = state.candidates[digit].bands[band] & boxToBoxesMaskTbl[box];
        while (box_bits != 0) {
            size_t bit_pos = BitUtils::bsf32(box_bits);
            uint32_t mask = BitUtils::ls1b32(box_bits);
            box_bits ^= mask;

            size_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != size_t(-1));

            State next_state(state);
            state.candidates[digit].bands[band] ^= mask;
            basic_solver::num_guesses++;

            this->update_band_solved_mask32(next_state, band, pos, digit);

            if (this->search<false>(next_state) != Status::Unsolvable) {
                this->guess_next_cell(next_state, solution);

                if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                    return Status::Success;
            }
        }

        return Status::Success;
    }

    JSTD_NO_INLINE
    int guess_hidden_box_bivalue_v1(State & state, char * solution) {
        // Count the total number of candidates under each digit.
        Counter counter;
#if 0
        for (size_t num = 0; num < Numbers; num++) {
            counter.digits[num] = BitUtils::popcnt32(state.candidates[num].bands[0]) +
                                  BitUtils::popcnt32(state.candidates[num].bands[1]) +
                                  BitUtils::popcnt32(state.candidates[num].bands[2]);
        }
        counter.digits[Numbers - 1] -= 10;
#else
        counter.digits[0] = BitUtils::popcnt32(state.candidates[0].bands[0]) +
                            BitUtils::popcnt32(state.candidates[0].bands[1]) +
                            BitUtils::popcnt32(state.candidates[0].bands[2]);

        counter.digits[1] = BitUtils::popcnt32(state.candidates[1].bands[0]) +
                            BitUtils::popcnt32(state.candidates[1].bands[1]) +
                            BitUtils::popcnt32(state.candidates[1].bands[2]);

        counter.digits[2] = BitUtils::popcnt32(state.candidates[2].bands[0]) +
                            BitUtils::popcnt32(state.candidates[2].bands[1]) +
                            BitUtils::popcnt32(state.candidates[2].bands[2]);

        counter.digits[3] = BitUtils::popcnt32(state.candidates[3].bands[0]) +
                            BitUtils::popcnt32(state.candidates[3].bands[1]) +
                            BitUtils::popcnt32(state.candidates[3].bands[2]);

        counter.digits[4] = BitUtils::popcnt32(state.candidates[4].bands[0]) +
                            BitUtils::popcnt32(state.candidates[4].bands[1]) +
                            BitUtils::popcnt32(state.candidates[4].bands[2]);

        counter.digits[5] = BitUtils::popcnt32(state.candidates[5].bands[0]) +
                            BitUtils::popcnt32(state.candidates[5].bands[1]) +
                            BitUtils::popcnt32(state.candidates[5].bands[2]);

        counter.digits[6] = BitUtils::popcnt32(state.candidates[6].bands[0]) +
                            BitUtils::popcnt32(state.candidates[6].bands[1]) +
                            BitUtils::popcnt32(state.candidates[6].bands[2]);

        counter.digits[7] = BitUtils::popcnt32(state.candidates[7].bands[0]) +
                            BitUtils::popcnt32(state.candidates[7].bands[1]) +
                            BitUtils::popcnt32(state.candidates[7].bands[2]);

        counter.digits[8] = BitUtils::popcnt32(state.candidates[8].bands[0]) +
                            BitUtils::popcnt32(state.candidates[8].bands[1]) +
                            BitUtils::popcnt32(state.candidates[8].bands[2]) - 10;
#endif
        Counter total_min, total_box;
        BoxCounter box_cnt;

        // Find the least number of candidates among all the digits
        BitVec08x16 counter8, minpos16;
        counter8.loadAligned((void *)&counter.digits[0]);
        // Exclude 0-9 candidates in a digit
        counter8 = _mm_sub_epi16(counter8.m128, _mm_set1_epi16(10));
        minpos16 = _mm_minpos_epu16(counter8.m128);
        uint32_t min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
        uint32_t min_candidates = min_and_index & 0xFFFFU;
        uint32_t min_digit;
        if (min_candidates <= (uint32_t)counter.digits[8]) {
            min_digit = min_and_index >> 16U;
        }
        else {
            min_digit = 8;
            min_candidates = (uint32_t)counter.digits[8];
        }

        if (min_candidates >= (81 - 10)) {
            return Status::Failed;
        }

        int cnt = Numbers;
        do {          
            // Count the total number of candidates under each box in one digit.
            uint32_t band_bits = state.candidates[min_digit].bands[0];
            box_cnt.boxes[0] = BitUtils::popcnt32(band_bits & 0007007007);
            box_cnt.boxes[1] = BitUtils::popcnt32(band_bits & 0070070070);
            box_cnt.boxes[2] = BitUtils::popcnt32(band_bits & 0700700700);

            band_bits = state.candidates[min_digit].bands[1];
            box_cnt.boxes[3] = BitUtils::popcnt32(band_bits & 0007007007);
            box_cnt.boxes[4] = BitUtils::popcnt32(band_bits & 0070070070);
            box_cnt.boxes[5] = BitUtils::popcnt32(band_bits & 0700700700);

            band_bits = state.candidates[min_digit].bands[2];
            box_cnt.boxes[6] = BitUtils::popcnt32(band_bits & 0007007007);
            box_cnt.boxes[7] = BitUtils::popcnt32(band_bits & 0070070070);
            box_cnt.boxes[8] = BitUtils::popcnt32(band_bits & 0700700700) - 2;

            // Find the cell of the first two candidates in a box
            counter8.loadAligned((void *)&box_cnt.boxes[0]);
            // Exclude 0 or 1 candidates in a box
            counter8 = _mm_sub_epi16(counter8.m128, _mm_set1_epi16(2));
            minpos16 = _mm_minpos_epu16(counter8.m128);
            min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
            min_candidates = min_and_index & 0xFFFFU;
            uint32_t min_box;
            if (min_candidates <= (uint32_t)box_cnt.boxes[8]) {
                min_box = min_and_index >> 16U;
            }
            else {
                min_box = 8;
                min_candidates = (uint32_t)box_cnt.boxes[8];
            }

            if (min_candidates == 0) {
                return this->guess_hidden_box_pair(state, solution, min_digit, min_box);
            }

            total_min.digits[min_digit] = min_candidates;
            total_box.digits[min_digit] = min_box;

            cnt--;
            if (cnt == 0)
                break;

            counter.digits[min_digit] = 99;

            // Find the least number of candidates among all the digits
            counter8.loadAligned((void *)&counter.digits[0]);
            // Exclude 0-9 candidates in a digit
            counter8 = _mm_sub_epi16(counter8.m128, _mm_set1_epi16(10));
            minpos16 = _mm_minpos_epu16(counter8.m128);
            min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
            min_candidates = min_and_index & 0xFFFFU;
            if (min_candidates <= (uint32_t)counter.digits[8]) {
                min_digit = min_and_index >> 16U;
            }
            else {
                min_digit = 8;
                //min_candidates = (uint32_t)counter.digits[8];
            }
            assert(int16_t(min_candidates) != int16_t(-1));
        } while (1);

        // Find the least number of candidates among all the boxes each digits
        {
            BitVec08x16 counter8, minpos16;
            counter8.loadAligned((void *)&total_min.digits[0]);
            minpos16 = _mm_minpos_epu16(counter8.m128);
            uint32_t min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
            uint32_t min_candidates = min_and_index & 0xFFFFU;
            uint32_t min_digit;
            if (min_candidates <= (uint32_t)total_min.digits[8]) {
                min_digit = min_and_index >> 16U;
            }
            else {
                min_digit = 8;
                min_candidates = (uint32_t)total_min.digits[8];
            }

            if (min_candidates <= (9 - 2)) {
                assert(min_digit >= 0 && min_digit < Numbers);
                uint32_t min_box = total_box.digits[min_digit];
                assert(min_box != uint32_t(-1));
                return this->guess_box_cell_more_than_2(state, solution, min_digit, min_box);
            }
            else {
                return Status::Failed;
            }
        }
    }

    JSTD_FORCED_INLINE
    int guess_hidden_box_bivalue(State & state, char * solution) {
        Counter total_min, total_box;
        BoxCounter box_cnt;
        BitVec08x16 counter8, minpos16;

        uint32_t min_and_index, min_candidates;

        uint32_t digit = 0;
        do {          
            // Count the total number of candidates under each box in one digit.
            uint32_t band_bits = state.candidates[digit].bands[0];
            box_cnt.boxes[0] = BitUtils::popcnt32(band_bits & 0007007007);
            box_cnt.boxes[1] = BitUtils::popcnt32(band_bits & 0070070070);
            box_cnt.boxes[2] = BitUtils::popcnt32(band_bits & 0700700700);

            uint32_t band_bits2 = state.candidates[digit].bands[1];
            box_cnt.boxes[3] = BitUtils::popcnt32(band_bits2 & 0007007007);
            box_cnt.boxes[4] = BitUtils::popcnt32(band_bits2 & 0070070070);
            box_cnt.boxes[5] = BitUtils::popcnt32(band_bits2 & 0700700700);

            band_bits = state.candidates[digit].bands[2];
            box_cnt.boxes[6] = BitUtils::popcnt32(band_bits & 0007007007);
            box_cnt.boxes[7] = BitUtils::popcnt32(band_bits & 0070070070);
            box_cnt.boxes[8] = BitUtils::popcnt32(band_bits & 0700700700) - 2;

            // Find the cell of the first two candidates in a box
            counter8.loadAligned((void *)&box_cnt.boxes[0]);
            // Exclude 0 or 1 candidates in a box
            counter8 = _mm_sub_epi16(counter8.m128, _mm_set1_epi16(2));
            minpos16 = _mm_minpos_epu16(counter8.m128);
            min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
            min_candidates = min_and_index & 0xFFFFU;
            uint32_t min_box;
            if (min_candidates <= (uint32_t)box_cnt.boxes[8]) {
                min_box = min_and_index >> 16U;
            }
            else {
                min_box = 8;
                min_candidates = (uint32_t)box_cnt.boxes[8];
            }

            if (min_candidates == 0) {
                return this->guess_hidden_box_pair(state, solution, digit, min_box);
            }

            total_min.digits[digit] = min_candidates;
            total_box.digits[digit] = min_box;

            digit++;
        } while (digit < (uint32_t)Numbers);

        // Find the least number of candidates among all the boxes each digits
        {
            counter8.loadAligned((void *)&total_min.digits[0]);
            minpos16 = _mm_minpos_epu16(counter8.m128);
            uint32_t min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos16.m128);
            uint32_t min_candidates = min_and_index & 0xFFFFU;
            uint32_t min_digit;
            if (min_candidates <= (uint32_t)total_min.digits[8]) {
                min_digit = min_and_index >> 16U;
            }
            else {
                min_digit = 8;
                min_candidates = (uint32_t)total_min.digits[8];
            }

            if (min_candidates <= (uint32_t)(Numbers - 2)) {
                assert(min_digit >= 0 && min_digit < Numbers);
                uint32_t min_box = total_box.digits[min_digit];
                assert(min_box != uint32_t(-1));
                return this->guess_box_cell_more_than_2(state, solution, min_digit, min_box);
            }
            else {
                return Status::Failed;
            }
        }
    }

    JSTD_FORCED_INLINE
    int guess_hidden_row_bivalue(State & state, char * solution) {
        uint32_t hidden_bits;
        uint32_t row_cnt, row_cnt2;
        uint32_t digit, band;
        for (digit = 0; digit < (uint32_t)Numbers; digit++) {
            uint32_t row_id = tables.div3[digit];
            uint32_t row_shift = tables.mod3[digit] * 9;
            uint32_t solvedRows = state.solvedRows.bands[row_id] >> row_shift;
            if ((solvedRows & kBand0RowBits) != kBand0RowBits) {
                // Band 0
                uint32_t band_bits = state.candidates[digit].bands[0];
                hidden_bits = band_bits & 0007007007;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                band = 0;
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0070070070;
                row_cnt2 = BitUtils::popcnt32(hidden_bits);                
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0700700700;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
            }
            if ((solvedRows & kBand1RowBits) != kBand1RowBits) {
                // Band 1
                uint32_t band_bits = state.candidates[digit].bands[1];
                hidden_bits = band_bits & 0007007007;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                band = 1;
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0070070070;
                row_cnt2 = BitUtils::popcnt32(hidden_bits);
                if (unlikely(row_cnt2 == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0700700700;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
            }
            if ((solvedRows & kBand2RowBits) != kBand2RowBits) {
                // Band 2
                uint32_t band_bits = state.candidates[digit].bands[2];
                hidden_bits = band_bits & 0007007007;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                band = 2;
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0070070070;
                row_cnt2 = BitUtils::popcnt32(hidden_bits);                
                if (unlikely(row_cnt2 == 2))
                    goto Row_BiValue_Find;
                hidden_bits = band_bits & 0700700700;
                row_cnt = BitUtils::popcnt32(hidden_bits);
                if (unlikely(row_cnt == 2))
                    goto Row_BiValue_Find;
            }
        }

        return Status::Failed;

Row_BiValue_Find:
        {
            // The first of row bi-value
            assert (hidden_bits != 0);
            uint32_t bit_pos = BitUtils::bsf32(hidden_bits);
            uint32_t mask = BitUtils::ls1b32(hidden_bits);
            hidden_bits ^= mask;

            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(int8_t(-1)));

            State next_state(state);
            state.candidates[digit].bands[band] ^= mask;
            basic_solver::num_guesses++;

            this->update_band_solved_mask32(next_state, band, pos, digit);

            if (this->search<false>(next_state) != Status::Unsolvable) {
                this->guess_next_cell(next_state, solution);

                if (kReachSolutionsLimitToExit2 && this->numSolutions_ >= this->limitSolutions_)
                    return Status::Success;
            }
        }

        {
            // The second of row bi-value
            assert (hidden_bits != 0);
            uint32_t bit_pos = BitUtils::bsf32(hidden_bits);
            uint32_t pos = bandBitPosToPos32[band][bit_pos];
            assert(pos != uint32_t(int8_t(-1)));

            this->update_band_solved_mask32(state, band, pos, digit);

            if (this->search<false>(state) != Status::Unsolvable) {
                this->guess_next_cell(state, solution);
            }
            return Status::Success;
        }
    }

    int guess_hidden_bivalue_cells(State & state, char * solution) {
        int status = this->guess_hidden_row_bivalue(state, solution);
        if (status == Status::Failed) {
            status = this->guess_hidden_box_bivalue(state, solution);
        }
        return status;
    }

    JSTD_FORCED_INLINE
    int guess_next_cell(State & state, char * solution) {
        if (this->is_solved(state)) {
            if (kSearchMode > SearchMode::OneSolution) {
                if (this->numSolutions_ == 0)
                    this->extract_solution(state, solution);
                this->numSolutions_++;
                if (kSearchMode == SearchMode::MoreThanOneSolution) {
                    if (this->numSolutions_ > this->limitSolutions_)
                        return Status::Solved;
                }
                return Status::Solved;
            }
            else {
                if (this->numSolutions_ == 0)
                    this->extract_solution(state, solution);
                this->numSolutions_++;
                return Status::Solved;
            }
        }

        if (this->guess_bivalue_cells(state, solution) == Status::Failed) {
            this->guess_hidden_bivalue_cells(state, solution);
        }

        return Status::Success;
    }

    template <bool fast_mode>
    int find_naked_singles(State & state) {
        int naked_singles;
        if (fast_mode)
            naked_singles = this->fast_find_naked_singles(state);
        else
            naked_singles = this->normal_find_naked_singles(state);
        return naked_singles;
    }

    template <bool fast_mode>
    JSTD_FORCED_INLINE
    int find_all_single_literals(State & state) {
        if (!fast_mode && (this->numSolutions_ >= this->limitSolutions_))
            return Status::Unsolvable;

        do {
            int status = this->find_locked_candidates_and_update<fast_mode>(state);
            if (!fast_mode && (status == Status::Unsolvable))
                return status;

            if (this->is_solved(state)) {
                return Status::Solved;
            }

            // If naked singles has found, go again
            int naked_singles = this->find_naked_singles<fast_mode>(state);
            if (naked_singles == 0)
                break;
            else if (!fast_mode && (naked_singles < 0))
                return Status::Unsolvable;
        } while (1);

        return Status::Success;
    }

    template <bool fast_mode>
    JSTD_FORCED_INLINE
    int search(State & state) {
        if (this->numSolutions_ >= this->limitSolutions_)
            return Status::Unsolvable;

        do {
            int status = this->find_locked_candidates_and_update<false>(state);
            if (status == Status::Unsolvable)
                return status;

            if (this->is_solved(state)) {
                return Status::Solved;
            }

            // If naked singles has found, go again
            int naked_singles = this->find_naked_singles<false>(state);
            if (naked_singles > 0)
                continue;
            else if (!fast_mode && (naked_singles < 0))
                return Status::Unsolvable;
            else
                break;
        } while (1);

        return Status::Success;
    }

    JSTD_FORCED_INLINE
    bool is_solved(State & state) {
#if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
 || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
        return ((state.solvedCells.bands64[0] == kBitSet27_Double64) &&
                (state.solvedCells.bands64[1] == kBitSet27_Single64));
#else
        return ((state.solvedCells.bands[0] == kBitSet27) &&
                (state.solvedCells.bands[1] == kBitSet27) &&
                (state.solvedCells.bands[2] == kBitSet27));
#endif
    }

public:
    JSTD_NO_INLINE
    int solve(const char * puzzle, char * solution, int limitSolutions = 1) {
        this->numSolutions_ = 0;
        this->limitSolutions_ = limitSolutions;
        basic_solver::num_guesses = 0;

        State & state = this->state_;
        int candidates = this->init_sudoku(state, puzzle);
        if (candidates < (int)Sudoku::MinInitCandidates)
            return -1;

        int naked_singles = this->find_naked_singles<kUseFastMode>(state);
        if (!kUseFastMode && (naked_singles < 0)) {
            return 0;
        }

#if (RUST_V1_ONLY_NO_GUESS == 0)
        int status = this->search<kUseFastMode>(state);
        if (status != Status::Unsolvable) {
            status = this->guess_next_cell(state, solution);
        }
        return this->numSolutions_;
#else
        return 0;
#endif
    }

    int solve(const Board & puzzle, Board & solution, int limitSolutions = 1) {
        const char * puzzle_ptr = (const char *)&puzzle.cells[0];
        char * solution_ptr = (char *)&solution.cells[0];
        return this->solve(puzzle_ptr, solution_ptr, limitSolutions);
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        basic_solver::display_result<kSearchMode>(board, elapsed_time, print_answer, print_all_answers);
    }
};

#if defined(GZ_SUDOKU)
RustV1Solver rust_v1_solver;
#endif

} // namespace

#if defined(GZ_SUDOKU)

#ifdef __cplusplus
extern "C"
#endif
int GzSudoku(const char * input, char * output, int limit)
{
    int solutions = rust_v1_solver.solve(input, output, limit);
    return solutions;
}

#endif // GZ_SUDOKU

#endif // GZ_SUDOKU_RUST_V1_H
