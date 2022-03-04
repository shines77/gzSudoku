
#ifndef GZ_SUDOKU_SOLVER_V4B_H
#define GZ_SUDOKU_SOLVER_V4B_H

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
#include "StopWatch.h"
#include "BitUtils.h"
#include "BitSet.h"
#include "PackedBitSet.h"
#include "BitArray.h"
#include "BitVec.h"

#define V4B_SAVE_COUNT_SIZE     1

#define V4B_USE_SIMD_INIT       1

namespace gzSudoku {
namespace v4b {

static const size_t kSearchMode = SearchMode::OneSolution;

static const size_t TableLs2b[16] = {
    0xFFFFFFFC, 0xFFFFFFF0, 0xFFFFFFC0, 0xFFFFFF00,
    0xFFFFFC00, 0xFFFFF000, 0xFFFFC000, 0xFFFF0000,
    0xFFFC0000, 0xFFF00000, 0xFFC00000, 0xFF000000,
    0xFC000000, 0xF0000000, 0xC0000000, 0x00000000
};

class Solver : public BasicSolverEx {
public:
    typedef BasicSolverEx                       basic_solver;
    typedef Solver                              this_type;

    typedef typename Sudoku::NeighborCells      NeighborCells;
    typedef typename Sudoku::CellInfo           CellInfo;
    typedef typename Sudoku::BoxesInfo          BoxesInfo;

    typedef typename Sudoku::BitMask            BitMask;
    typedef typename Sudoku::BitMaskTable       BitMaskTable;

    static const size_t kAlignment = Sudoku::kAlignment;
    static const size_t BoxCellsX = Sudoku::kBoxCellsX;      // 3
    static const size_t BoxCellsY = Sudoku::kBoxCellsY;      // 3
    static const size_t BoxCountX = Sudoku::kBoxCountX;      // 3
    static const size_t BoxCountY = Sudoku::kBoxCountY;      // 3
    static const size_t MinNumber = Sudoku::kMinNumber;      // 1
    static const size_t MaxNumber = Sudoku::kMaxNumber;      // 9

    static const size_t Rows = Sudoku::kRows;
    static const size_t Cols = Sudoku::kCols;
    static const size_t Boxes = Sudoku::kBoxes;
    static const size_t BoxSize = Sudoku::kBoxSize;
    static const size_t Numbers = Sudoku::kNumbers;

    static const size_t BoardSize = Sudoku::kBoardSize;
    static const size_t TotalSize = Sudoku::kTotalSize;
    static const size_t Neighbors = Sudoku::kNeighbors;

    static const size_t Rows16 = Sudoku::kRows16;
    static const size_t Cols16 = Sudoku::kCols16;
    static const size_t Numbers10 = Sudoku::kNumbers10;
    static const size_t Numbers16 = Sudoku::kNumbers16;
    static const size_t Boxes16 = Sudoku::kBoxes16;
    static const size_t BoxSize16 = Sudoku::kBoxSize16;
    static const size_t BoardSize16 = Sudoku::kBoardSize16;

    static const size_t Rows32 = Rows16 * 2;
    static const size_t Cols32 = Cols16 * 2;
    static const size_t Numbers32 = Numbers16 * 2;
    static const size_t Boxes32 = Boxes16 * 2;
    static const size_t BoxSize32 = BoxSize16 * 2;
    static const size_t BoardSize32 = Boxes32 * BoxSize32;

    static const size_t MaxConfig = 6;
    static const size_t Config8 = AlignedTo<MaxConfig, 8>::value;

    static const size_t TotalCellLiterals = Boxes16 * BoxSize16;
    static const size_t TotalRowLiterals = Rows16 * Numbers16;
    static const size_t TotalColLiterals = Cols16 * Numbers16;
    static const size_t TotalBoxLiterals = Boxes16 * Numbers16;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t RowLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t BoxLiteralFirst  = ColLiteralFirst + TotalColLiterals;
    static const size_t LiteralLast      = BoxLiteralFirst + TotalBoxLiterals;

    static const size_t CellLiteralLast  = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = LiteralLast;

    static const size_t kAllRowBits = Sudoku::kAllRowBits;
    static const size_t kAllColBits = Sudoku::kAllColBits;
    static const size_t kAllBoxBits = Sudoku::kAllBoxBits;
    static const size_t kAllBoxCellBits = Sudoku::kAllBoxCellBits;
    static const size_t kAllNumberBits = Sudoku::kAllNumberBits;

    static const size_t kDisableNumberMask = 0x0200U;

    static const bool kAllDimIsSame = Sudoku::kAllDimIsSame;

    static const int kLiteralCntThreshold = 0;
    static const uint32_t kLiteralCntThreshold2 = 0;

private:
    enum LiteralType {
        NumRowCols,
        NumColRows,
        NumBoxCells,
        BoxCellNums,
        Unknown
    };

private:

#pragma pack(push, 1)

    struct alignas(32) BandConfigure {
        PackedBitSet2D<Config8, Numbers16> config;          // Band[config][num]
        PackedBitSet2D<Config8, Numbers16> exclude;         // Band[config][num]
    };

    union LiteralChanged {
        struct {
            uint16_t literal[4];
        };

        uint64_t value;

        LiteralChanged(uint64_t _value = 0) : value(_value) {}
        ~LiteralChanged() {}
    };

    struct alignas(32) InitState {
        PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
        PackedBitSet3D<Numbers, Rows16, Cols16>       num_row_cols;     // [num][row][col]
        PackedBitSet3D<Numbers, Cols16, Rows16>       num_col_rows;     // [num][col][row]
        PackedBitSet3D<Numbers, Boxes16, BoxSize16>   num_box_cells;    // [num][box][cell]

        LiteralChanged changed;

        InitState() {}
        ~InitState() {}
    };

    struct alignas(32) State {
        PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]

        BandConfigure h_band[BoxCountX];
        BandConfigure v_band[BoxCountY];        

        State() {}
        ~State() {}
    };

    struct alignas(32) Count {
        struct Sizes {
            uint16_t box_cells[Boxes][BoxSize16];
            uint16_t num_boxes[Numbers][Boxes16];
            uint16_t num_rows[Numbers][Rows16];
            uint16_t num_cols[Numbers][Cols16];
        } sizes;

        struct Counts {
            uint16_t box_cells[Boxes16];
            uint16_t num_boxes[Numbers16];
            uint16_t num_rows[Numbers16];
            uint16_t num_cols[Numbers16];
        } counts;

        struct Index {
            uint8_t box_cells[Boxes16];
            uint8_t num_boxes[Numbers16];
            uint8_t num_rows[Numbers16];
            uint8_t num_cols[Numbers16];
        } indexs;

        struct Total {
            uint16_t min_literal_size[16];
            uint16_t min_literal_index[16];
        } total;

        Count() {}
        ~Count() {}
    };

    union literal_count_t {
        struct {
            uint8_t count:7;
            uint8_t enable:1;
        };
        uint8_t value;
    };

    union LiteralInfoEx {
        struct {
            uint32_t literal_size;
            uint16_t literal_type;
            uint16_t literal_index;
        };

        uint64_t value;

        LiteralInfoEx(uint64_t _value = 0) : value(_value) {}
        LiteralInfoEx(size_t size, size_t type, size_t index)
            : literal_size((uint32_t)size), literal_type((uint16_t)type), literal_index((uint16_t)index) {}
        LiteralInfoEx(const LiteralInfoEx & src) : value(src.value) {
        }

        LiteralInfoEx & operator = (const LiteralInfoEx & rhs) {
            this->value = rhs.value;
            return *this;
        }

        bool isValid() const {
            return (this->value != uint64_t(-1));
        }
    };

    union LiteralInfo {
        struct {
            uint32_t literal_type;
            uint32_t literal_index;
        };

        uint64_t value;

        LiteralInfo(uint64_t _value = 0) : value(_value) {}
        LiteralInfo(size_t type, size_t index)
            : literal_type((uint32_t)type), literal_index((uint32_t)index) {}
        LiteralInfo(const LiteralInfo & src) : value(src.value) {
        }

        LiteralInfo & operator = (const LiteralInfo & rhs) {
            this->value = rhs.value;
            return *this;
        }

        bool isValid() const {
            return (this->value != uint64_t(-1));
        }

        LiteralInfoEx toLiteralInfoEx(uint32_t literal_size) {
            return LiteralInfoEx(literal_size, literal_type, literal_index);
        }
    };

    union LiteralMask {
        struct {
            uint16_t literal_type;
            uint16_t literal_first;
            uint32_t literal_mask;
        };

        struct {
            uint32_t literal_info_0;
            uint32_t literal_mask_0;
        };

        uint64_t value;

        LiteralMask(uint64_t _value = 0) : value(_value) {}
        LiteralMask(size_t info, size_t mask)
            : literal_info_0((uint32_t)info), literal_mask_0((uint32_t)mask) {}
        LiteralMask(size_t type, size_t first, size_t mask)
            : literal_type((uint16_t)type), literal_first((uint16_t)first), literal_mask((uint32_t)mask) {}
        LiteralMask(const LiteralMask & src) : value(src.value) {
        }

        LiteralMask & operator = (const LiteralMask & rhs) {
            this->value = rhs.value;
            return *this;
        }

        bool isValid() const {
            return (this->value != 0);
        }
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct PeerBoxes {
        static const uint32_t kBoxesCount = (uint32_t)((nBoxCountX - 1) + (nBoxCountY - 1));

        uint32_t boxes_count() const { return kBoxesCount; }

        int boxes[kBoxesCount];
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct HVPeerBoxes {
        static const uint32_t kHorizontalBoxes = uint32_t(nBoxCountX - 1);
        static const uint32_t kVerticalBoxes = uint32_t(nBoxCountX - 1);

        uint32_t h_boxes_count() const { return kHorizontalBoxes; }
        uint32_t v_boxes_count() const { return kVerticalBoxes; }

        int h_boxes[kHorizontalBoxes];
        int v_boxes[kVerticalBoxes];
    };

    typedef PeerBoxes<BoxCountX, BoxCountY>     peer_boxes_t;
    typedef HVPeerBoxes<BoxCountX, BoxCountY>   hv_peer_boxes_t;

    struct alignas(32) StaticData {
        PackedBitSet3D<4, BoxSize16, Numbers16>         flip_mask[BoxSize][Numbers];

        PackedBitSet3D<BoardSize, Rows16, Cols16>       num_row_mask;
        PackedBitSet3D<BoardSize, Cols16, Rows16>       num_col_mask;
        PackedBitSet3D<BoardSize, Boxes16, BoxSize16>   num_box_mask;

        peer_boxes_t    peer_boxes[Boxes];
        hv_peer_boxes_t hv_peer_boxes[Boxes];

        uint16_t        box_changed_bits[8][Boxes16];
        bool            mask_is_inited;

        StaticData() : mask_is_inited(false) {
            if (!Static.mask_is_inited) {
                Sudoku::initialize();
                this_type::init_mask();
                Static.mask_is_inited = true;
            }
        }
    };

#pragma pack(pop)

    InitState       init_state_;
    State           state_;
    Count           count_;
    LiteralInfoEx   last_literal_;

    alignas(32) literal_count_t literal_info_[TotalLiterals];

    static StaticData Static;

public:
    Solver() : basic_solver() {
    }
    ~Solver() {}

private:
    static void init_peer_boxes() {
        for (size_t box_y = 0; box_y < BoxCellsY; box_y++) {
            size_t box_y_base = box_y * BoxCellsX;
            for (size_t box_x = 0; box_x < BoxCellsX; box_x++) {
                uint32_t box = uint32_t(box_y_base + box_x);
                size_t index = 0;
                peer_boxes_t peerBoxes;
                //peerBoxes.boxes[index++] = box;
                uint32_t h_changed_bits = (1U << box);
                uint32_t v_changed_bits = (1U << box);
                for (size_t box_i = 0; box_i < BoxCellsX; box_i++) {
                    if (box_i != box_x) {
                        uint32_t peer_box = uint32_t(box_y * BoxCellsX + box_i);
                        peerBoxes.boxes[index++] = peer_box;
                        h_changed_bits |= (1U << peer_box);
                    }
                }
                for (size_t box_j = 0; box_j < BoxCellsY; box_j++) {
                    if (box_j != box_y) {
                        uint32_t peer_box = uint32_t(box_j * BoxCellsX + box_x);
                        peerBoxes.boxes[index++] = peer_box;
                        v_changed_bits |= (1U << peer_box);
                    }
                }
                assert(index == peerBoxes.boxes_count());

                //std::sort(&peerBoxes.boxes[1], &peerBoxes.boxes[peerBoxes.boxes_count()]);
                Static.peer_boxes[box] = peerBoxes;

                for (size_t type = 0; type < 8; type++) {
                    Static.box_changed_bits[type][box] = (uint16_t)(h_changed_bits | v_changed_bits);
                }
                Static.box_changed_bits[LiteralType::NumRowCols][box] = (uint16_t)(v_changed_bits);
                Static.box_changed_bits[LiteralType::NumColRows][box] = (uint16_t)(h_changed_bits);
            }
        }
    }

    static void make_num_flip_mask(size_t fill_pos, size_t row, size_t col) {
        PackedBitSet2D<Rows16, Cols16> & rows_mask      = Static.num_row_mask[fill_pos];
        PackedBitSet2D<Cols16, Rows16> & cols_mask      = Static.num_col_mask[fill_pos];
        PackedBitSet2D<Boxes16, BoxSize16> & boxes_mask = Static.num_box_mask[fill_pos];

        const CellInfo * pCellInfo = Sudoku::cell_info;
        size_t box_x = col / BoxCellsX;
        size_t box_y = row / BoxCellsY;
        size_t box_first_y = box_y * BoxCellsY;
        size_t box_first_x = box_x * BoxCellsX;
        size_t box, cell;

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
        }

        // Literal::NumColRows
        {
            size_t index = 0;
            // horizontal scanning
            for (size_t x = 0; x < Cols; x++) {
                if (x != col) {
                    cols_mask[x].set(row);
                    index++;
                }
            }
            // vertical scanning
            for (size_t y = 0; y < Rows; y++) {
                cols_mask[col].set(y);
                index++;
            }
            // scanning the current box
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                size_t row_y = box_first_y + cy;
                for (size_t cx = 0; cx < BoxCellsX; cx++) {
                    size_t col_x = box_first_x + cx;
                    cols_mask[col_x].set(row_y);
                    index++;
                }
            }
            assert(index == (Cols + Rows + (BoxCellsY * BoxCellsX) - 1));
        }

        // Literal::NumBoxCells
        {
            size_t cell_x = col % BoxCellsX;
            size_t cell_y = row % BoxCellsY;
            size_t box_id_first = box_y * BoxCellsX + box_x;

            size_t index = 0;

            // horizontal scanning
            size_t pos_start = row * Cols;
            for (size_t x = 0; x < Cols; x++) {
                size_t pos = pos_start + x;
                box = pCellInfo[pos].box;
                cell = pCellInfo[pos].cell;
                boxes_mask[box].set(cell);
                index++;
            }
            // vertical scanning
            for (size_t y = 0; y < Rows; y++) {
                if (y != row) {
                    size_t pos = y * Cols + col;
                    box = pCellInfo[pos].box;
                    cell = pCellInfo[pos].cell;
                    boxes_mask[box].set(cell);
                    index++;
                }
            }
            // scanning the current box
            box = pCellInfo[fill_pos].box;
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                for (size_t cx = 0; cx < BoxCellsX; cx++) {
                    size_t cell_idx = cy * BoxCellsX + cx;
                    boxes_mask[box].set(cell_idx);
                    index++;
                }
            }
            assert(index == (Cols + Rows + (BoxCellsY * BoxCellsX) - 1));
        }
    }

    static void make_flip_mask(size_t fill_cell, size_t box_cell_y, size_t box_cell_x) {
        // Literal::BoxCellNums
        {
            size_t index = 0;

            // horizontal scanning
            for (size_t cx = 0; cx < BoxCellsX; cx++) {
                size_t cell = box_cell_y * BoxCellsX + cx;

                for (size_t num = 0; num < Numbers; num++) {
                    PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
                                = Static.flip_mask[fill_cell][num];
                    flip_mask[0][cell].set(num);
                }
                index++;
            }

            // vertical scanning
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                size_t cell = cy * BoxCellsX + box_cell_x;

                for (size_t num = 0; num < Numbers; num++) {
                    PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
                                = Static.flip_mask[fill_cell][num];
                    flip_mask[1][cell].set(num);
                }
                index++;
            }

            // scanning the current box
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                for (size_t cx = 0; cx < BoxCellsX; cx++) {
                    size_t cell = cy * BoxCellsX + cx;

                    for (size_t num = 0; num < Numbers; num++) {
                        PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
                                    = Static.flip_mask[fill_cell][num];
                        flip_mask[2][cell].set(num);
                    }
                    index++;
                }
            }

            assert(index == (BoxCellsX + BoxCellsY + (BoxCellsY * BoxCellsX)));

            // Current fill pos, set  all number bits
            for (size_t num = 0; num < Numbers; num++) {
                PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
                    = Static.flip_mask[fill_cell][num];
                flip_mask[2][fill_cell].fill(kAllNumberBits);
            }
        }
    }

    static void init_flip_mask() {
        for (size_t cell = 0; cell < BoxSize; cell++) {
            for (size_t num = 0; num < Numbers; num++) {
                Static.flip_mask[cell][num].reset();
            }
        }

        size_t fill_cell = 0;
        for (size_t cy = 0; cy < BoxCellsY; cy++) {
            for (size_t cx = 0; cx < BoxCellsX; cx++) {
                make_flip_mask(fill_cell, cy, cx);
                fill_cell++;
            }
        }

        Static.num_row_mask.reset();
        Static.num_col_mask.reset();
        Static.num_box_mask.reset();

        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_num_flip_mask(fill_pos, row, col);
                fill_pos++;
            }
        }
    }

    static void init_mask() {
        printf("v4b::Solver::StaticData::init_mask()\n");

        init_peer_boxes();
        init_flip_mask();
    }

    static const uint16_t kEnableLiteral16 = 0x0000;
    static const uint16_t kDisableLiteral16 = 0xFFFF;

    void init_literal_info() {
#if V4B_USE_SIMD_INIT
        // init_literal_size()
        BitVec16x16_AVX full_mask;
        full_mask.fill_u16(Numbers);
        for (size_t box = 0; box < Boxes; box++) {
            full_mask.saveAligned((void *)&this->count_.sizes.box_cells[box]);
        }

        for (size_t num = 0; num < Numbers; num++) {
            full_mask.saveAligned((void *)&this->count_.sizes.num_rows[num]);
            full_mask.saveAligned((void *)&this->count_.sizes.num_cols[num]);
            full_mask.saveAligned((void *)&this->count_.sizes.num_boxes[num]);
        }

        // init_literal_count()
        BitVec16x16_AVX all_ones_avx;
        all_ones_avx.setAllOnes();
        all_ones_avx.saveAligned((void *)&this->count_.counts.box_cells[0]);
        all_ones_avx.saveAligned((void *)&this->count_.counts.num_rows[0]);
        all_ones_avx.saveAligned((void *)&this->count_.counts.num_cols[0]);
        all_ones_avx.saveAligned((void *)&this->count_.counts.num_boxes[0]);

        // init_literal_total()
        all_ones_avx.saveAligned((void *)&this->count_.total.min_literal_size[0]);
        all_ones_avx.saveAligned((void *)&this->count_.total.min_literal_index[0]);

        // init_literal_index()
        BitVec08x16 all_ones;
        all_ones.setAllOnes();
        all_ones.saveAligned((void *)&this->count_.indexs.box_cells[0]);
        all_ones.saveAligned((void *)&this->count_.indexs.num_rows[0]);
        all_ones.saveAligned((void *)&this->count_.indexs.num_cols[0]);
        all_ones.saveAligned((void *)&this->count_.indexs.num_boxes[0]);
#else
        init_literal_size();
        init_literal_count();
        init_literal_index();
        init_literal_total();
#endif
    }

    void init_literal_size() {
        for (size_t box = 0; box < Boxes; box++) {
            for (size_t cell = 0; cell < BoxSize16; cell++) {
                this->count_.sizes.box_cells[box][cell] = Numbers;
            }
        }

        for (size_t num = 0; num < Numbers; num++) {
            for (size_t row = 0; row < Rows16; row++) {
                this->count_.sizes.num_rows[num][row] = Cols;
            }
            for (size_t col = 0; col < Cols16; col++) {
                this->count_.sizes.num_cols[num][col] = Rows;
            }
            for (size_t box = 0; box < Boxes16; box++) {
                this->count_.sizes.num_boxes[num][box] = BoxSize;
            }
        }
    }

    void init_literal_count() {
        for (size_t i = 0; i < Boxes16; i++) {
            this->count_.counts.box_cells[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.num_rows[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.num_cols[i] = 255;
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.counts.num_boxes[i] = 255;
        }
    }

    void init_literal_index() {
        for (size_t i = 0; i < Boxes16; i++) {
            this->count_.indexs.box_cells[i] = uint8_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.num_rows[i] = uint8_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.num_cols[i] = uint8_t(-1);
        }

        for (size_t i = 0; i < Numbers16; i++) {
            this->count_.indexs.num_boxes[i] = uint8_t(-1);
        }
    }

    void init_literal_total() {
        for (size_t i = 0; i < 16; i++) {
            this->count_.total.min_literal_size[i] = 32767;
            this->count_.total.min_literal_index[i] = uint16_t(-1);
        }
    }

    void init_board(const Board & board) {
        init_literal_info();

#if V4B_USE_SIMD_INIT
        BitVec16x16_AVX full_mask;
        full_mask.fill_u16(kAllNumberBits);
        for (size_t box = 0; box < Boxes; box++) {
            full_mask.saveAligned((void *)&this->init_state_.box_cell_nums[box]);
        }

        for (size_t num = 0; num < Numbers; num++) {
            full_mask.saveAligned((void *)&this->init_state_.num_row_cols[num]);
            full_mask.saveAligned((void *)&this->init_state_.num_col_rows[num]);
            full_mask.saveAligned((void *)&this->init_state_.num_box_cells[num]);
        }
#else
        this->init_state_.box_cell_nums.fill(kAllNumberBits);
        this->init_state_.num_row_cols.fill(kAllColBits);
        this->init_state_.num_col_rows.fill(kAllRowBits);
        this->init_state_.num_box_cells.fill(kAllBoxCellBits);
#endif

#if 0
        this->init_state_.changed.value = (kAllNumberBits << 48) | (kAllBoxCellBits << 32) |
                                          (kAllColBits    << 16) | kAllRowBits;
#else
        this->init_state_.changed.literal[LiteralType::NumRowCols] = kAllColBits;
        this->init_state_.changed.literal[LiteralType::NumColRows] = kAllRowBits;
        this->init_state_.changed.literal[LiteralType::NumBoxCells] = kAllBoxCellBits;
        this->init_state_.changed.literal[LiteralType::BoxCellNums] = kAllNumberBits;
#endif

        if (kSearchMode > SearchMode::OneSolution) {
            this->answers_.clear();
        }

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            size_t cell_y = (row % BoxCellsY) * BoxCellsX;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board.cells[pos];
                if (val != '.') {
                    size_t num = val - '1';
                    size_t box_x = col / BoxCellsX;
                    size_t cell_x = col % BoxCellsX;
                    size_t box = box_y + box_x;
                    size_t cell = cell_y + cell_x;

                    this->fill_num_init(this->init_state_, row, col, box, cell, num);
                    this->update_peer_cells(this->init_state_, pos, box, cell, num);
                }
                pos++;
            }
        }
        assert(pos == BoardSize);
    }

    inline void fill_num_init(InitState & init_state, size_t row, size_t col,
                              size_t box, size_t cell, size_t num) {
        assert(init_state.box_cell_nums[box][cell].test(num));
        assert(init_state.num_row_cols[num][row].test(col));
        assert(init_state.num_col_rows[num][col].test(row));
        assert(init_state.num_box_cells[num][box].test(cell));

        size_t num_bits = init_state.box_cell_nums[box][cell].value();
        //init_state.box_cell_nums[box][cell].reset();

        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        while (num_bits != 0) {
            size_t cur_num = BitUtils::bsf(num_bits);
            size_t num_bit = BitUtils::ls1b(num_bits);
            num_bits ^= num_bit;

            assert(init_state.num_row_cols[cur_num][row].test(col));
            assert(init_state.num_col_rows[cur_num][col].test(row));
            assert(init_state.num_box_cells[cur_num][box].test(cell));

            init_state.num_row_cols[cur_num][row].reset(col);
            init_state.num_col_rows[cur_num][col].reset(row);
            init_state.num_box_cells[cur_num][box].reset(cell);
        }
    }

    inline size_t fill_num(InitState & init_state, size_t row, size_t col,
                           size_t box, size_t cell, size_t num) {
        assert(init_state.box_cell_nums[box][cell].test(num));
        assert(init_state.num_row_cols[num][row].test(col));
        assert(init_state.num_col_rows[num][col].test(row));
        assert(init_state.num_box_cells[num][box].test(cell));

        size_t num_bits = init_state.box_cell_nums[box][cell].value();
        //init_state.box_cell_nums[box][cell].reset();

        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        size_t other_num_bits = num_bits;
        while (num_bits != 0) {
            size_t cur_num = BitUtils::bsf(num_bits);
            size_t num_bit = BitUtils::ls1b(num_bits);
            num_bits ^= num_bit;

            assert(init_state.num_row_cols[cur_num][row].test(col));
            assert(init_state.num_col_rows[cur_num][col].test(row));
            assert(init_state.num_box_cells[cur_num][box].test(cell));

            init_state.num_row_cols[cur_num][row].reset(col);
            init_state.num_col_rows[cur_num][col].reset(row);
            init_state.num_box_cells[cur_num][box].reset(cell);
        }

        return other_num_bits;
    }

    template <size_t nLiteralType = LiteralType::Unknown>
    inline void update_peer_cells(InitState & init_state, size_t fill_pos, size_t fill_box, size_t cell, size_t num) {
        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;
        const peer_boxes_t & peerBoxes = Static.peer_boxes[fill_box];
        const PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
            = Static.flip_mask[cell][num];

        // LiteralType::BoxCellNums
        size_t box = fill_box;
        //init_state.box_cell_nums[box] &= flip_mask[box];
        pCells16 = (void *)&init_state.box_cell_nums[box];
        pMask16 = (void *)&flip_mask[2];
        cells16.loadAligned(pCells16);
        mask16.loadAligned(pMask16);
        cells16.and_not(mask16);
        cells16.saveAligned(pCells16);

        if (nLiteralType != LiteralType::NumRowCols) {
            box = peerBoxes.boxes[0];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            pMask16 = (void *)&flip_mask[0];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box = peerBoxes.boxes[1];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        if (nLiteralType != LiteralType::NumColRows) {
            box = peerBoxes.boxes[2];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            pMask16 = (void *)&flip_mask[1];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box = peerBoxes.boxes[3];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        //init_state.num_row_cols[num] &= Static.num_row_mask[fill_pos];
        //init_state.num_col_rows[num] &= Static.num_col_mask[fill_pos];
        //init_state.num_box_cells[num] &= Static.num_box_mask[fill_pos];

        // LiteralType::NumRowCols
        {
            pCells16 = (void *)&init_state.num_row_cols[num];
            pMask16 = (void *)&Static.num_row_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumColRows
        {
            pCells16 = (void *)&init_state.num_col_rows[num];
            pMask16 = (void *)&Static.num_col_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumBoxCells
        {
            pCells16 = (void *)&init_state.num_box_cells[num];
            pMask16 = (void *)&Static.num_box_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }
    }

    template <size_t nLiteralType = LiteralType::Unknown>
    inline void update_peer_cells_delta(InitState & init_state, size_t fill_pos, size_t row, size_t col,
                                        size_t fill_box, size_t cell, size_t num) {
        assert(init_state.box_cell_nums[fill_box][cell].test(num));
        assert(init_state.num_row_cols[num][row].test(col));
        assert(init_state.num_col_rows[num][col].test(row));
        assert(init_state.num_box_cells[num][fill_box].test(cell));

        static const size_t fixedLiteralType = (nLiteralType > LiteralType::Unknown) ? LiteralType::Unknown : nLiteralType;
        uint64_t box_changed_bits = Static.box_changed_bits[fixedLiteralType][fill_box];

        size_t num_bits = init_state.box_cell_nums[fill_box][cell].value();
        assert(num_bits != 0);
#if 0
        init_state.changed.literal[LiteralType::NumRowCols] |= (uint16_t)num_bits;
        init_state.changed.literal[LiteralType::NumColRows] |= (uint16_t)num_bits;
        init_state.changed.literal[LiteralType::NumBoxCells] |= (uint16_t)num_bits;
        init_state.changed.literal[LiteralType::BoxCellNums] |= (uint16_t)box_changed_bits;
#else
        uint64_t changed = (uint64_t)num_bits * 0x0000000100010001ULL | (box_changed_bits << 48);
        assert(changed == ((box_changed_bits << 48) | (num_bits << 32) | (num_bits << 16) | num_bits));
        init_state.changed.value |= changed;
#endif
        if (nLiteralType != LiteralType::BoxCellNums) {
            // Exclude the current number, because it will be process later.
            num_bits ^= (size_t(1) << num);
            while (num_bits != 0) {
                size_t cur_num = BitUtils::bsf(num_bits);
                size_t num_bit = BitUtils::ls1b(num_bits);
                num_bits ^= num_bit;

                assert(init_state.num_row_cols[cur_num][row].test(col));
                assert(init_state.num_col_rows[cur_num][col].test(row));
                assert(init_state.num_box_cells[cur_num][fill_box].test(cell));

                init_state.num_row_cols[cur_num][row].reset(col);
                init_state.num_col_rows[cur_num][col].reset(row);
                init_state.num_box_cells[cur_num][fill_box].reset(cell);
            }
        }

        const PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask = Static.flip_mask[cell][num];
        const peer_boxes_t & peerBoxes = Static.peer_boxes[fill_box];

        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;

        // LiteralType::BoxCellNums
        size_t box = fill_box;
        //init_state.box_cell_nums[box] &= flip_mask[box];
        pCells16 = (void *)&init_state.box_cell_nums[box];
        pMask16 = (void *)&flip_mask[2];
        cells16.loadAligned(pCells16);
        mask16.loadAligned(pMask16);
        cells16.and_not(mask16);
        cells16.saveAligned(pCells16);

        if (nLiteralType != LiteralType::NumRowCols) {
            box = peerBoxes.boxes[0];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            pMask16 = (void *)&flip_mask[0];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box = peerBoxes.boxes[1];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        if (nLiteralType != LiteralType::NumColRows) {
            box = peerBoxes.boxes[2];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            pMask16 = (void *)&flip_mask[1];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box = peerBoxes.boxes[3];
            //init_state.box_cell_nums[box] &= flip_mask[box];
            pCells16 = (void *)&init_state.box_cell_nums[box];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        //init_state.num_row_cols[num] &= Static.num_row_mask[fill_pos];
        //init_state.num_col_rows[num] &= Static.num_col_mask[fill_pos];
        //init_state.num_box_cells[num] &= Static.num_box_mask[fill_pos];

        // LiteralType::NumRowCols
        {
            pCells16 = (void *)&init_state.num_row_cols[num];
            pMask16 = (void *)&Static.num_row_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumColRows
        {
            pCells16 = (void *)&init_state.num_col_rows[num];
            pMask16 = (void *)&Static.num_col_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumBoxCells
        {
            pCells16 = (void *)&init_state.num_box_cells[num];
            pMask16 = (void *)&Static.num_box_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }
    }

    LiteralMask find_single_literal_full() {
        BitVec16x16_AVX unique_mask;
        unique_mask.fill_u16(1);

        // Row literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumRowCols, uint32_t(num), single_mask);
            }
        }

        // Col literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumColRows, uint32_t(num), single_mask);
            }
        }

        // Box-Cell literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumBoxCells, uint32_t(num), single_mask);
            }
        }

        // Position (Box-Cell) literal
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void *)&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = box_bits.popcount16<BoxSize, Numbers>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::BoxCellNums, uint32_t(box), single_mask);
            }
        }

        return LiteralMask(0);
    }

    LiteralMask find_single_literal_full(size_t fill_num) {
        BitVec16x16_AVX unique_mask;
        unique_mask.fill_u16(1);

        // Row literal
        {
            void * pCells16 = (void *)&this->init_state_.num_row_cols[fill_num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumRowCols, uint32_t(fill_num), single_mask);
            }
        }

        // Col literal
        {
            void * pCells16 = (void *)&this->init_state_.num_col_rows[fill_num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumColRows, uint32_t(fill_num), single_mask);
            }
        }

        // Box-Cell literal
        {
            void * pCells16 = (void *)&this->init_state_.num_box_cells[fill_num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumBoxCells, uint32_t(fill_num), single_mask);
            }
        }

        // Row literal
        for (size_t num = 0; num < Numbers; num++) {
            if (num == fill_num) continue;
            void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumRowCols, uint32_t(num), single_mask);
            }
        }

        // Col literal
        for (size_t num = 0; num < Numbers; num++) {
            if (num == fill_num) continue;
            void * pCells16 = (void *)&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumColRows, uint32_t(num), single_mask);
            }
        }

        // Box-Cell literal
        for (size_t num = 0; num < Numbers; num++) {
            if (num == fill_num) continue;
            void * pCells16 = (void *)&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::NumBoxCells, uint32_t(num), single_mask);
            }
        }

        // Position (Box-Cell) literal
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void *)&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = box_bits.popcount16<BoxSize, Numbers>();

            uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
            if (single_mask != 0) {
                assert(single_mask < (1UL << 19));
                return LiteralMask(LiteralType::BoxCellNums, uint32_t(box), single_mask);
            }
        }

        return LiteralMask(0);
    }

    LiteralMask find_single_literal_delta() {
        BitVec16x16_AVX unique_mask;
        unique_mask.fill_u16(1);

        // Row literal
        uint16_t num_bits = this->init_state_.changed.literal[LiteralType::NumRowCols];
        for (size_t num = 0; num < Numbers; num++) {
            if ((num_bits & 1) != 0) {
                void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
                BitVec16x16_AVX num_row_bits;
                num_row_bits.loadAligned(pCells16);

                BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

                uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
                if (single_mask != 0) {
                    num_bits = (num_bits & 0xFFFEU) << num;
                    this->init_state_.changed.literal[LiteralType::NumRowCols] = num_bits;
                    assert(single_mask < (1UL << 19));
                    return LiteralMask(LiteralType::NumRowCols, uint32_t(num), single_mask);
                }
            }
            num_bits >>= 1;
        }

        this->init_state_.changed.literal[LiteralType::NumRowCols] = 0;

        // Col literal
        num_bits = this->init_state_.changed.literal[LiteralType::NumColRows];
        for (size_t num = 0; num < Numbers; num++) {
            if ((num_bits & 1) != 0) {
                void * pCells16 = (void *)&this->init_state_.num_col_rows[num];
                BitVec16x16_AVX num_col_bits;
                num_col_bits.loadAligned(pCells16);

                BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

                uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
                if (single_mask != 0) {
                    num_bits = (num_bits & 0xFFFEU) << num;
                    this->init_state_.changed.literal[LiteralType::NumColRows] = num_bits;
                    assert(single_mask < (1UL << 19));
                    return LiteralMask(LiteralType::NumColRows, uint32_t(num), single_mask);
                }
            }
            num_bits >>= 1;
        }

        this->init_state_.changed.literal[LiteralType::NumColRows] = 0;

        // Box-Cell literal
        num_bits = this->init_state_.changed.literal[LiteralType::NumBoxCells];
        for (size_t num = 0; num < Numbers; num++) {
            if ((num_bits & 1) != 0) {
                void * pCells16 = (void *)&this->init_state_.num_box_cells[num];
                BitVec16x16_AVX num_box_bits;
                num_box_bits.loadAligned(pCells16);

                BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();

                uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
                if (single_mask != 0) {
                    num_bits = (num_bits & 0xFFFEU) << num;
                    this->init_state_.changed.literal[LiteralType::NumBoxCells] = num_bits;
                    assert(single_mask < (1UL << 19));
                    return LiteralMask(LiteralType::NumBoxCells, uint32_t(num), single_mask);
                }
            }
            num_bits >>= 1;
        }

        this->init_state_.changed.literal[LiteralType::NumBoxCells] = 0;

        // Position (Box-Cell) literal
        uint16_t boxes_bits = this->init_state_.changed.literal[LiteralType::BoxCellNums];
        for (size_t box = 0; box < Boxes; box++) {
            if ((boxes_bits & 1) != 0) {
                void * pCells16 = (void *)&this->init_state_.box_cell_nums[box];
                BitVec16x16_AVX box_bits;
                box_bits.loadAligned(pCells16);

                BitVec16x16_AVX popcnt16 = box_bits.popcount16<BoxSize, Numbers>();

                uint32_t single_mask = popcnt16.maskOfIsEqual16<true>(unique_mask);
                if (single_mask != 0) {
                    boxes_bits = (boxes_bits & 0xFFFEU) << box;
                    this->init_state_.changed.literal[LiteralType::BoxCellNums] = boxes_bits;
                    assert(single_mask < (1UL << 19));
                    return LiteralMask(LiteralType::BoxCellNums, uint32_t(box), single_mask);
                }
            }
            boxes_bits >>= 1;
        }

        this->init_state_.changed.literal[LiteralType::BoxCellNums] = 0;

        return LiteralMask(0);
    }

    size_t find_all_single_literal(std::vector<LiteralInfo> & single_literal_list) {
        BitVec16x16_AVX unique_mask;
        unique_mask.fill_u16(1);

        // Row literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t row_index = (uint32_t)(num * Rows16 + min_index);
                single_literal_list.push_back(LiteralInfo(LiteralType::NumRowCols, row_index));
            }
        }

        // Col literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t col_index = (uint32_t)(num * Cols16 + min_index);
                single_literal_list.push_back(LiteralInfo(LiteralType::NumColRows, col_index));
            }
        }

        // Box-Cell literal
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void *)&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();
            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t box_index = (uint32_t)(num * Boxes16 + min_index);
                single_literal_list.push_back(LiteralInfo(LiteralType::NumBoxCells, box_index));
            }
        }

        // Position (Box-Cell) literal
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void *)&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            BitVec16x16_AVX popcnt16 = box_bits.popcount16<BoxSize, Numbers>();
            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t box_index = (uint32_t)(box * BoxSize16 + min_index);
                single_literal_list.push_back(LiteralInfo(LiteralType::BoxCellNums, box_index));
            }
        }

        return single_literal_list.size();
    }

    void do_single_literal_delta(InitState & init_state, Board & board, LiteralInfo literalInfo) {
        size_t pos, row, col, box, cell, num;

        switch (literalInfo.literal_type) {
            case LiteralType::NumRowCols:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Rows16);

                num = literal / Rows16;
                row = literal % Rows16;

                size_t col_bits = init_state.num_row_cols[num][row].to_ulong();
                assert(col_bits != 0);
                assert((col_bits & (col_bits - 1)) == 0);
                col = BitUtils::bsf(col_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells_delta<LiteralType::NumRowCols>(init_state, pos, row, col, box, cell, num);

                break;
            }

            case LiteralType::NumColRows:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Cols16);

                num = literal / Cols16;
                col = literal % Cols16;

                size_t row_bits = init_state.num_col_rows[num][col].to_ulong();
                assert(row_bits != 0);
                assert((row_bits & (row_bits - 1)) == 0);
                row = BitUtils::bsf(row_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells_delta<LiteralType::NumColRows>(init_state, pos, row, col, box, cell, num);

                break;
            }

            case LiteralType::NumBoxCells:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Boxes16);

                num = literal / Boxes16;
                box = literal % Boxes16;

                size_t cell_bits = init_state.num_box_cells[num][box].to_ulong();
                assert(cell_bits != 0);
                assert((cell_bits & (cell_bits - 1)) == 0);
                cell = BitUtils::bsf(cell_bits);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box * BoxSize16 + cell];
                row = boxesInfo.row;
                col = boxesInfo.col;
                pos = boxesInfo.pos;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells_delta<LiteralType::NumBoxCells>(init_state, pos, row, col, box, cell, num);

                break;
            }

            case LiteralType::BoxCellNums:
            {
                size_t box_pos = literalInfo.literal_index;
                assert(box_pos < Boxes * BoxSize16);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                row = boxesInfo.row;
                col = boxesInfo.col;
                box = boxesInfo.box;
                cell = boxesInfo.cell;
                pos = boxesInfo.pos;

                size_t num_bits = init_state.box_cell_nums[box][cell].to_ulong();
                assert(num_bits != 0);
                assert((num_bits & (num_bits - 1)) == 0);
                num = BitUtils::bsf(num_bits);

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells_delta<LiteralType::BoxCellNums>(init_state, pos, row, col, box, cell, num);

                break;
            }

            default:
                assert(false);
                break;
        }
    }

    bool check_and_do_single_literal_delta(InitState & init_state, Board & board, LiteralInfo literalInfo) {
        size_t pos, row, col, box, cell, num;

        switch (literalInfo.literal_type) {
            case LiteralType::BoxCellNums:
            {
                size_t box_pos = literalInfo.literal_index;
                assert(box_pos < Boxes * BoxSize16);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                row = boxesInfo.row;
                col = boxesInfo.col;
                box = boxesInfo.box;
                cell = boxesInfo.cell;
                pos = boxesInfo.pos;

                size_t num_bits = init_state.box_cell_nums[box][cell].to_ulong();
                if (num_bits != 0) {
                    assert(num_bits != 0);
                    assert((num_bits & (num_bits - 1)) == 0);
                    num = BitUtils::bsf(num_bits);

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells_delta<LiteralType::BoxCellNums>(init_state, pos, row, col, box, cell, num);
                    return true;
                }

                break;
            }

            case LiteralType::NumRowCols:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Rows16);

                num = literal / Rows16;
                row = literal % Rows16;

                size_t col_bits = init_state.num_row_cols[num][row].to_ulong();
                if (col_bits != 0) {
                    assert(col_bits != 0);
                    assert((col_bits & (col_bits - 1)) == 0);
                    col = BitUtils::bsf(col_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells_delta<LiteralType::NumRowCols>(init_state, pos, row, col, box, cell, num);
                    return true;
                }

                break;
            }

            case LiteralType::NumColRows:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Cols16);

                num = literal / Cols16;
                col = literal % Cols16;

                size_t row_bits = init_state.num_col_rows[num][col].to_ulong();
                if (row_bits != 0) {
                    assert(row_bits != 0);
                    assert((row_bits & (row_bits - 1)) == 0);
                    row = BitUtils::bsf(row_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells_delta<LiteralType::NumColRows>(init_state, pos, row, col, box, cell, num);
                    return true;
                }

                break;
            }

            case LiteralType::NumBoxCells:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Boxes16);

                num = literal / Boxes16;
                box = literal % Boxes16;

                size_t cell_bits = init_state.num_box_cells[num][box].to_ulong();
                if (cell_bits != 0) {
                    assert(cell_bits != 0);
                    assert((cell_bits & (cell_bits - 1)) == 0);
                    cell = BitUtils::bsf(cell_bits);

                    const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box * BoxSize16 + cell];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    pos = boxesInfo.pos;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells_delta<LiteralType::NumBoxCells>(init_state, pos, row, col, box, cell, num);
                    return true;
                }

                break;
            }

            default:
                assert(false);
                break;
        }

        return false;
    }

    size_t do_single_literal_mask(InitState & init_state, Board & board, LiteralMask literalMask) {
        size_t count = 0;
        size_t pos, row, col, box, cell, num;

        switch (literalMask.literal_type) {
            case LiteralType::NumRowCols:
            {
                num = literalMask.literal_first;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    row = index;                    

                    size_t col_bits = init_state.num_row_cols[num][row].to_ulong();
                    assert(col_bits != 0);
                    assert((col_bits & (col_bits - 1)) == 0);
                    col = BitUtils::bsf(col_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumRowCols>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::NumColRows:
            {
                num = literalMask.literal_first;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    col = index;                    

                    size_t row_bits = init_state.num_col_rows[num][col].to_ulong();
                    assert(row_bits != 0);
                    assert((row_bits & (row_bits - 1)) == 0);
                    row = BitUtils::bsf(row_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumColRows>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::NumBoxCells:
            {
                num = literalMask.literal_first;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    box = index;

                    size_t cell_bits = init_state.num_box_cells[num][box].to_ulong();
                    assert(cell_bits != 0);
                    assert((cell_bits & (cell_bits - 1)) == 0);
                    cell = BitUtils::bsf(cell_bits);

                    size_t box_pos = box * BoxSize16 + cell;
                    const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    pos = boxesInfo.pos;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumBoxCells>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::BoxCellNums:
            {
                box = literalMask.literal_first;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    cell = index;
                    size_t box_pos = box * BoxSize16 + cell;

                    const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    pos = boxesInfo.pos;

                    size_t num_bits = init_state.box_cell_nums[box][cell].to_ulong();
                    assert(num_bits != 0);
                    assert((num_bits & (num_bits - 1)) == 0);
                    num = BitUtils::bsf(num_bits);

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    //this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::BoxCellNums>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            default:
                assert(false);
                break;
        }

        return count;
    }

    size_t do_single_literal_mask(InitState & init_state, Board & board, LiteralMask literalMask, size_t & fill_num) {
        size_t count = 0;
        size_t pos, row, col, box, cell, num;

        switch (literalMask.literal_type) {
            case LiteralType::NumRowCols:
            {
                num = literalMask.literal_first;
                fill_num = num;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    row = index;                    

                    size_t col_bits = init_state.num_row_cols[num][row].to_ulong();
                    assert(col_bits != 0);
                    assert((col_bits & (col_bits - 1)) == 0);
                    col = BitUtils::bsf(col_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumRowCols>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::NumColRows:
            {
                num = literalMask.literal_first;
                fill_num = num;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    col = index;                    

                    size_t row_bits = init_state.num_col_rows[num][col].to_ulong();
                    assert(row_bits != 0);
                    assert((row_bits & (row_bits - 1)) == 0);
                    row = BitUtils::bsf(row_bits);
                    pos = row * Cols + col;

                    const CellInfo & cellInfo = Sudoku::cell_info[pos];
                    box = cellInfo.box;
                    cell = cellInfo.cell;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumColRows>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::NumBoxCells:
            {
                num = literalMask.literal_first;
                fill_num = num;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    box = index;

                    size_t cell_bits = init_state.num_box_cells[num][box].to_ulong();
                    assert(cell_bits != 0);
                    assert((cell_bits & (cell_bits - 1)) == 0);
                    cell = BitUtils::bsf(cell_bits);

                    size_t box_pos = box * BoxSize16 + cell;
                    const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    pos = boxesInfo.pos;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::NumBoxCells>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            case LiteralType::BoxCellNums:
            {
                box = literalMask.literal_first;

                size_t single_mask = literalMask.literal_mask;
                do {
                    uint32_t index = BitUtils::bsf(single_mask);
                    assert((index & 1) == 0);
                    assert(index < 9 * 2);
                    index >>= 1;
                    assert(index < 9);
                    single_mask &= TableLs2b[index];

                    cell = index;
                    size_t box_pos = box * BoxSize16 + cell;

                    const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box_pos];
                    row = boxesInfo.row;
                    col = boxesInfo.col;
                    pos = boxesInfo.pos;

                    size_t num_bits = init_state.box_cell_nums[box][cell].to_ulong();
                    assert(num_bits != 0);
                    assert((num_bits & (num_bits - 1)) == 0);
                    num = BitUtils::bsf(num_bits);
                    fill_num = num;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    //this->fill_num_init(init_state, row, col, box, cell, num);
                    this->update_peer_cells<LiteralType::BoxCellNums>(init_state, pos, box, cell, num);

                    count++;
                } while (single_mask != 0);

                break;
            }

            default:
                assert(false);
                break;
        }

        return count;
    }

public:
    bool search(Board & board, ptrdiff_t empties, const LiteralInfoEx & literalInfo) {
        if (empties <= 0) {
            if (kSearchMode > SearchMode::OneSolution) {
                this->answers_.push_back(board);
                if (kSearchMode == SearchMode::MoreThanOneSolution) {
                    if (this->answers_.size() > 1)
                        return true;
                }
            }
            else {
                return true;
            }
        }

        return false;
    }

    int solve(const Board & board, Board & solution, int limitSolutions = 1) {
        this->init_board(board);

        ptrdiff_t empties = this->calc_empties(board);
        assert(empties <= (Sudoku::kBoardSize - Sudoku::kMinInitCandidates));

        solution = board;
#if 0
        LiteralInfo literalInfo;
        LiteralMask literalMask = this->find_single_literal_full();

        while (literalMask.isValid()) {
            uint32_t literal_type    = literalMask.literal_type;
            uint32_t literal_first   = literalMask.literal_first;
            literalInfo.literal_type = literal_type;
            size_t single_mask       = literalMask.literal_mask;
            assert(single_mask != 0);
            ptrdiff_t count = 0;
            while (single_mask != 0) {
                uint32_t index = BitUtils::bsf(single_mask);
                assert((index & 1) == 0);
                assert(index < 9 * 2);
                index >>= 1;
                assert(index < 9);
                single_mask &= TableLs2b[index];
                literalInfo.literal_index = (literal_first * 16) | index;
                this->do_single_literal_delta(this->init_state_, solution, literalInfo);
                count++;
            }
            empties -= count;
            if (empties <= 0) {
                break;
            }
            literalMask = this->find_single_literal_delta();
        }
#elif 1
        LiteralMask literalMask = this->find_single_literal_full();

        while (literalMask.isValid()) {
            size_t single_mask = literalMask.literal_mask;
            assert(single_mask != 0);
            //size_t fill_num;
            ptrdiff_t count = this->do_single_literal_mask(this->init_state_, solution, literalMask);
            assert(count != 0);
            empties -= count;
            if (empties <= 0) {
                break;
            }
            literalMask = this->find_single_literal_full();
        }
#else
        LiteralMask literalMask = this->find_single_literal_full();

        while (literalMask.isValid()) {
            size_t single_mask = literalMask.literal_mask;
            assert(single_mask != 0);
            size_t fill_num;
            ptrdiff_t count = this->do_single_literal_mask(this->init_state_, solution, literalMask, fill_num);
            assert(count != 0);
            empties -= count;
            if (empties <= 0) {
                break;
            }
            literalMask = this->find_single_literal_full(fill_num);
        }
#endif
        bool success = this->search(solution, empties, this->last_literal_);
        return (success) ? Status::Solved : Status::Invalid;
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        basic_solver::display_result<kSearchMode>(board, elapsed_time, print_answer, print_all_answers);
    }
};

Solver::StaticData Solver::Static;

} // namespace v4b
} // namespace gzSudoku

#endif // GZ_SUDOKU_SOLVER_V4B_H
