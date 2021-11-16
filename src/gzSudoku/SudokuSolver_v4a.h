
#ifndef GZ_SUDOKU_SOLVER_V4A_H
#define GZ_SUDOKU_SOLVER_V4A_H

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

#if defined(_MSC_VER)
#include <emmintrin.h>      // For SSE 2
#include <tmmintrin.h>      // For SSE 3
#include <smmintrin.h>      // For SSE 4.1
#else
#include <x86intrin.h>      // For SSE 4.1
#endif // _MSC_VER

#include "BasicSolver.h"
#include "Sudoku.h"
#include "StopWatch.h"
#include "BitUtils.h"
#include "BitSet.h"
#include "PackedBitSet.h"
#include "BitArray.h"
#include "BitVec.h"

#define V4A_LITERAL_ORDER_MODE   0
#define V4A_SAVE_COUNT_SIZE      1

namespace gzSudoku {
namespace v4a {

static const size_t kSearchMode = SearchMode::OneAnswer;

class Solver : public BasicSolver {
public:
    typedef BasicSolver                         basic_solver_t;
    typedef Solver                              solver_type;

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
    static const size_t MaxConfig = 6;

    static const size_t Config8 = AlignedTo<MaxConfig, 8>::value;
    static const size_t Rows16 = AlignedTo<Rows, 16>::value;
    static const size_t Cols16 = AlignedTo<Cols, 16>::value;
    static const size_t Numbers16 = AlignedTo<Numbers, 16>::value;
    static const size_t Boxes16 = AlignedTo<Boxes, 16>::value;
    static const size_t BoxSize16 = AlignedTo<BoxSize, 16>::value;
    static const size_t BoardSize16 = Boxes16 * BoxSize16;

    static const size_t Rows32 = Rows16 * 2;
    static const size_t Cols32 = Cols16 * 2;
    static const size_t Numbers32 = Numbers16 * 2;
    static const size_t Boxes32 = Boxes16 * 2;
    static const size_t BoxSize32 = BoxSize16 * 2;
    static const size_t BoardSize32 = Boxes32 * BoxSize32;

    static const size_t TotalCellLiterals = Boxes16 * BoxSize16;
    static const size_t TotalRowLiterals = Rows16 * Numbers16;
    static const size_t TotalColLiterals = Cols16 * Numbers16;
    static const size_t TotalBoxLiterals = Boxes16 * Numbers16;

    static const size_t TotalLiterals =
        TotalCellLiterals + TotalRowLiterals + TotalColLiterals + TotalBoxLiterals;

#if (V4A_LITERAL_ORDER_MODE == 0)
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
#else
    static const size_t LiteralFirst     = 0;
    static const size_t CellLiteralFirst = LiteralFirst;
    static const size_t BoxLiteralFirst  = CellLiteralFirst + TotalCellLiterals;
    static const size_t RowLiteralFirst  = BoxLiteralFirst + TotalBoxLiterals;
    static const size_t ColLiteralFirst  = RowLiteralFirst + TotalRowLiterals;
    static const size_t LiteralLast      = ColLiteralFirst + TotalColLiterals;

    static const size_t CellLiteralLast  = BoxLiteralFirst;
    static const size_t BoxLiteralLast   = RowLiteralFirst;
    static const size_t RowLiteralLast   = ColLiteralFirst;
    static const size_t ColLiteralLast   = LiteralLast;
#endif // (V4A_LITERAL_ORDER_MODE == 0)

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
#if (V4A_LITERAL_ORDER_MODE == 0)
    enum LiteralType {
        BoxCellNums,
        NumRowCols,
        NumColRows,
        NumBoxCells,
        Unknown
    };
#else
    enum LiteralType {
        BoxCellNums,
        NumBoxCells,
        NumRowCols,
        NumColRows,
        Unknown
    };
#endif // (V4A_LITERAL_ORDER_MODE == 0)

private:

#pragma pack(push, 1)

    struct BandConfigure {
        alignas(32) PackedBitSet2D<Config8, Numbers16> config;          // Band[config][num]
        alignas(16) PackedBitSet2D<Config8, Numbers16> exclude;         // Band[config][num]
    };

    struct Box {
        alignas(32) PackedBitSet2D<BoxSize16, Numbers16> nums;          // Boxes[cell][num]
    };

    struct Row {
        alignas(32) PackedBitSet2D<Rows16, Cols16> cols;                // Number[row][col]
    };

    struct Col {
        alignas(32) PackedBitSet2D<Rows16, Cols16> rows;                // Number[col][row]
    };

    struct BoxCell {
        alignas(32) PackedBitSet2D<Boxes16, BoxSize16> cells;           // Number[box][cell]
    };

    struct InitState {
        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
        alignas(32) PackedBitSet3D<Numbers, Rows16, Cols16>       num_row_cols;     // [num][row][col]
        alignas(32) PackedBitSet3D<Numbers, Cols16, Rows16>       num_col_rows;     // [num][col][row]
        alignas(32) PackedBitSet3D<Numbers, Boxes16, BoxSize16>   num_box_cells;    // [num][box][cell]
    };

    struct State {
        BandConfigure h_band[BoxCountX];
        BandConfigure v_band[BoxCountY];

        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
    };

    struct Count {
        struct Sizes {
            alignas(32) uint16_t box_cells[Boxes][BoxSize16];
            alignas(32) uint16_t num_boxes[Numbers][Boxes16];
            alignas(32) uint16_t num_rows[Numbers][Rows16];
            alignas(32) uint16_t num_cols[Numbers][Cols16];
        } sizes;

        struct Counts {
            alignas(32) uint16_t box_cells[Boxes16];
            alignas(32) uint16_t num_boxes[Numbers16];
            alignas(32) uint16_t num_rows[Numbers16];
            alignas(32) uint16_t num_cols[Numbers16];
        } counts;

        struct Index {
            alignas(16) uint8_t box_cells[Boxes16];
            alignas(16) uint8_t num_boxes[Numbers16];
            alignas(16) uint8_t num_rows[Numbers16];
            alignas(16) uint8_t num_cols[Numbers16];
        } indexs;

        struct Total {
            alignas(32) uint16_t min_literal_size[16];
            alignas(32) uint16_t min_literal_index[16];
        } total;
    };

    union literal_count_t {
        struct {
            uint8_t count:7;
            uint8_t enable:1;
        };
        uint8_t value;
    };

    struct LiteralInfo {
        uint32_t literal_size;
        uint16_t literal_type;
        uint16_t literal_index;

        LiteralInfo() : literal_size(0), literal_type(0), literal_index(0) {}
        LiteralInfo(uint32_t size, uint32_t type, uint32_t index)
            : literal_size(size), literal_type((uint16_t)type), literal_index((uint16_t)index) {}
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct NeighborBoxes {
        static const uint32_t kBoxesCount = (uint32_t)((nBoxCountX - 1) + (nBoxCountY - 1));

        uint32_t boxes_count() const { return kBoxesCount; }

        int boxes[kBoxesCount];
    };

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct HVNeighborBoxes {
        static const uint32_t kHorizontalBoxes = uint32_t(nBoxCountX - 1);
        static const uint32_t kVerticalBoxes = uint32_t(nBoxCountX - 1);

        uint32_t h_boxes_count() const { return kHorizontalBoxes; }
        uint32_t v_boxes_count() const { return kVerticalBoxes; }

        int h_boxes[kHorizontalBoxes];
        int v_boxes[kVerticalBoxes];
    };

    typedef NeighborBoxes<BoxCountX, BoxCountY>     neighbor_boxes_t;
    typedef HVNeighborBoxes<BoxCountX, BoxCountY>   hv_neighbor_boxes_t;

    struct StaticData {
        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>     box_flip_mask[BoardSize][Numbers];

        alignas(32) PackedBitSet3D<BoardSize, Rows16, Cols16>       num_row_mask;
        alignas(32) PackedBitSet3D<BoardSize, Cols16, Rows16>       num_col_mask;
        alignas(32) PackedBitSet3D<BoardSize, Boxes16, BoxSize16>   num_box_mask;

        alignas(64) PackedBitSet3D<4, BoxSize16, Numbers16>         flip_mask[BoxSize][Numbers];

        bool                    mask_is_inited;
        neighbor_boxes_t        neighbor_boxes[Boxes];
        hv_neighbor_boxes_t     hv_neighbor_boxes[Boxes];

        StaticData() : mask_is_inited(false) {
            if (!Static.mask_is_inited) {
                Sudoku::initialize();
                solver_type::init_mask();
                Static.mask_is_inited = true;
            }
        }
    };

#pragma pack(pop)

    InitState       init_state_;
    State           state_;
    Count           count_;
    LiteralInfo     min_info_;

    alignas(32) literal_count_t literal_info_[TotalLiterals];

    static StaticData Static;

public:
    Solver() : basic_solver_t() {
    }
    ~Solver() {}

private:
    static void init_neighbor_boxes() {
        for (size_t box_y = 0; box_y < BoxCellsY; box_y++) {
            size_t box_y_base = box_y * BoxCellsX;
            for (size_t box_x = 0; box_x < BoxCellsX; box_x++) {
                uint32_t box = uint32_t(box_y_base + box_x);
                size_t index = 0;
                neighbor_boxes_t neighborBoxes;
                //neighborBoxes.boxes[index++] = box;
                for (size_t box_i = 0; box_i < BoxCellsX; box_i++) {
                    if (box_i != box_x) {
                        neighborBoxes.boxes[index++] = uint32_t(box_y * BoxCellsX + box_i);
                    }
                }
                for (size_t box_j = 0; box_j < BoxCellsY; box_j++) {
                    if (box_j != box_y) {
                        neighborBoxes.boxes[index++] = uint32_t(box_j * BoxCellsX + box_x);
                    }
                }
                assert(index == neighborBoxes.boxes_count());

                //std::sort(&neighborBoxes.boxes[1], &neighborBoxes.boxes[neighborBoxes.boxes_count()]);
                Static.neighbor_boxes[box] = neighborBoxes;
            }
        }
    }

    static void make_box_flip_mask(size_t fill_pos, size_t row, size_t col) {
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

        // Literal::BoxCellNums
        {
            size_t index = 0;

            // horizontal scanning
            for (size_t x = 0; x < Cols; x++) {
                size_t pos = row * Cols + x;

                box = pCellInfo[pos].box;
                cell = pCellInfo[pos].cell;

                for (size_t num = 0; num < Numbers; num++) {
                    PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_flip_mask
                                = Static.box_flip_mask[fill_pos][num];
                    box_flip_mask[box][cell].set(num);
                }
                index++;
            }

            // vertical scanning
            for (size_t y = 0; y < Rows; y++) {
                if (y != row) {
                    size_t pos = y * Cols + col;

                    box = pCellInfo[pos].box;
                    cell = pCellInfo[pos].cell;

                    for (size_t num = 0; num < Numbers; num++) {
                        PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_flip_mask
                                    = Static.box_flip_mask[fill_pos][num];
                        box_flip_mask[box][cell].set(num);
                    }
                    index++;
                }
            }

            // scanning the current box
            size_t box_first = (box_y * BoxCellsY) * Cols + box_x * BoxCellsX;
            for (size_t cy = 0; cy < BoxCellsY; cy++) {
                for (size_t cx = 0; cx < BoxCellsX; cx++) {
                    size_t pos = box_first + cy * Cols + cx;
                    size_t cell_idx = cy * BoxCellsX + cx;

                    box = pCellInfo[pos].box;
                    cell = pCellInfo[pos].cell;
                    assert(cell == cell_idx);

                    for (size_t num = 0; num < Numbers; num++) {
                        PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_flip_mask
                                    = Static.box_flip_mask[fill_pos][num];
                        box_flip_mask[box][cell].set(num);
                    }
                    index++;
                }
            }

            assert(index == (Cols + Rows + (BoxCellsY * BoxCellsX) - 1));

            box = pCellInfo[fill_pos].box;
            cell = pCellInfo[fill_pos].cell;

            // Current fill pos, set all number bits
            for (size_t num = 0; num < Numbers; num++) {
                PackedBitSet3D<Boxes, BoxSize16, Numbers16> & box_flip_mask
                    = Static.box_flip_mask[fill_pos][num];
                box_flip_mask[box][cell].fill(kAllNumberBits);
            }
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

            // Current fill pos, set all number bits
            for (size_t num = 0; num < Numbers; num++) {
                PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
                    = Static.flip_mask[fill_cell][num];
                flip_mask[2][fill_cell].fill(kAllNumberBits);
            }
        }
    }

    static void init_flip_mask() {
        for (size_t pos = 0; pos < BoardSize; pos++) {
            for (size_t num = 0; num < Numbers; num++) {
                Static.box_flip_mask[pos][num].reset();
            }
        }
        Static.num_row_mask.reset();
        Static.num_col_mask.reset();
        Static.num_box_mask.reset();

        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_box_flip_mask(fill_pos, row, col);
                fill_pos++;
            }
        }

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
    }

    static void init_mask() {
        printf("v4a::StaticData::init_mask()\n");

        init_neighbor_boxes();
        init_flip_mask();
    }

    static const uint16_t kEnableLiteral16 = 0x0000;
    static const uint16_t kDisableLiteral16 = 0xFFFF;

    void init_literal_info() {
        init_literal_size();
        init_literal_count();
        init_literal_index();
        init_literal_total();
    }

    void init_literal_size() {
#if 1
        BitVec16x16_AVX mask_1, mask_2, mask_3, mask_4;
        mask_1.fill_u16(Numbers);
        for (size_t box = 0; box < Boxes; box++) {
            mask_1.saveAligned((void *)&this->count_.sizes.box_cells[box]);
        }

        mask_2.fill_u16(Cols);
        mask_3.fill_u16(Rows);
        mask_4.fill_u16(BoxSize);
        for (size_t num = 0; num < Numbers; num++) {
            mask_2.saveAligned((void *)&this->count_.sizes.num_rows[num]);
            mask_3.saveAligned((void *)&this->count_.sizes.num_cols[num]);
            mask_4.saveAligned((void *)&this->count_.sizes.num_boxes[num]);
        }
#else
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
#endif
    }

    void init_literal_count() {
#if 1
        BitVec16x16_AVX bitVec;
        bitVec.setAllOnes();
        bitVec.saveAligned((void *)&this->count_.counts.box_cells[0]);
        bitVec.saveAligned((void *)&this->count_.counts.num_rows[0]);
        bitVec.saveAligned((void *)&this->count_.counts.num_cols[0]);
        bitVec.saveAligned((void *)&this->count_.counts.num_boxes[0]);
#else
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
#endif
    }

    void init_literal_index() {
#if 1
        BitVec08x16 bitVec;
        bitVec.setAllOnes();
        bitVec.saveAligned((void *)&this->count_.indexs.box_cells[0]);
        bitVec.saveAligned((void *)&this->count_.indexs.num_rows[0]);
        bitVec.saveAligned((void *)&this->count_.indexs.num_cols[0]);
        bitVec.saveAligned((void *)&this->count_.indexs.num_boxes[0]);
#else
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
#endif
    }

    void init_literal_total() {
#if 1
        BitVec16x16_AVX bitVec;
        bitVec.setAllOnes();
        bitVec.saveAligned((void *)&this->count_.total.min_literal_size[0]);
        bitVec.saveAligned((void *)&this->count_.total.min_literal_index[0]);
#else
        for (size_t i = 0; i < 16; i++) {
            this->count_.total.min_literal_size[i] = 32767;
            this->count_.total.min_literal_index[i] = uint16_t(-1);
        }
#endif
    }

    inline size_t fillNum(InitState & init_state, size_t row, size_t col,
                          size_t box, size_t cell, size_t num) {
        assert(init_state.box_cell_nums[box][cell].test(num));
        //assert(init_state.num_row_cols[num][row].test(col));
        //assert(init_state.num_col_rows[num][col].test(row));
        //assert(init_state.num_box_cells[num][box].test(cell));

        PackedBitSet<Numbers16> cell_num_bits = init_state.box_cell_nums[box][cell];
        //state.box_cell_nums[box][cell].reset();

        /*
        size_t box_pos = box * BoxSize16 + cell;
        size_t row_idx = num * Rows16 + row;
        size_t col_idx = num * Cols16 + col;
        size_t box_idx = num * Boxes16 + box;

        disable_cell_literal(box_pos);
        disable_row_literal(row_idx);
        disable_col_literal(col_idx);
        disable_box_literal(box_idx);
        //*/

        size_t num_bits = cell_num_bits.to_ulong();
        // Exclude the current number, because it will be process later.
        num_bits ^= (size_t(1) << num);
        size_t save_num_bits = num_bits;
        while (num_bits != 0) {
            size_t num_bit = BitUtils::ls1b(num_bits);
            size_t cur_num = BitUtils::bsf(num_bit);
            num_bits ^= num_bit;

            assert(init_state.num_row_cols[cur_num][row].test(col));
            assert(init_state.num_col_rows[cur_num][col].test(row));
            assert(init_state.num_box_cells[cur_num][box].test(cell));

            init_state.num_row_cols[cur_num][row].reset(col);
            init_state.num_col_rows[cur_num][col].reset(row);
            init_state.num_box_cells[cur_num][box].reset(cell);
        }

        return save_num_bits;
    }

    template <size_t nLiteralType = LiteralType::Unknown>
    inline void updateNeighborCells(InitState & init_state, size_t fill_pos, size_t box, size_t cell, size_t num) {
        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;
        const neighbor_boxes_t & neighborBoxes = Static.neighbor_boxes[box];
        const PackedBitSet3D<4, BoxSize16, Numbers16> & flip_mask
            = Static.flip_mask[cell][num];

        size_t box_idx;
        //init_state.box_cell_nums[box_idx] &= flip_mask[box_idx];
        pCells16 = (void *)&init_state.box_cell_nums[box];
        pMask16 = (void *)&flip_mask[2];
        cells16.loadAligned(pCells16);
        mask16.loadAligned(pMask16);
        cells16.and_not(mask16);
        cells16.saveAligned(pCells16);

        if (nLiteralType != LiteralType::NumRowCols) {
            box_idx = neighborBoxes.boxes[0];
            //init_state.box_cell_nums[box_idx] &= flip_mask[box_idx];
            pCells16 = (void *)&init_state.box_cell_nums[box_idx];
            pMask16 = (void *)&flip_mask[0];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box_idx = neighborBoxes.boxes[1];
            //init_state.box_cell_nums[box_idx] &= flip_mask[box_idx];
            pCells16 = (void *)&init_state.box_cell_nums[box_idx];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        if (nLiteralType != LiteralType::NumColRows) {
            box_idx = neighborBoxes.boxes[2];
            //init_state.box_cell_nums[box_idx] &= flip_mask[box_idx];
            pCells16 = (void *)&init_state.box_cell_nums[box_idx];
            pMask16 = (void *)&flip_mask[1];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);

            box_idx = neighborBoxes.boxes[3];
            //init_state.box_cell_nums[box_idx] &= flip_mask[box_idx];
            pCells16 = (void *)&init_state.box_cell_nums[box_idx];
            cells16.loadAligned(pCells16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        //init_state.num_row_cols[num] &= Static.num_row_mask[fill_pos];
        //init_state.num_col_rows[num] &= Static.num_col_mask[fill_pos];
        //init_state.num_box_cells[num] &= Static.num_box_mask[fill_pos];

        //if (nLiteralType != LiteralType::NumRowCols)
        {
            pCells16 = (void *)&init_state.num_row_cols[num];
            pMask16 = (void *)&Static.num_row_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        //if (nLiteralType != LiteralType::NumColRows)
        {
            pCells16 = (void *)&init_state.num_col_rows[num];
            pMask16 = (void *)&Static.num_col_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }

        //if (nLiteralType != LiteralType::NumBoxCells)
        {
            pCells16 = (void *)&init_state.num_box_cells[num];
            pMask16 = (void *)&Static.num_box_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16.saveAligned(pCells16);
        }
    }

    LiteralInfo count_literal_size_init_v1() {
        BitVec16x16_AVX disable_mask;
        BitVec16x16_AVX numbits_mask;
        numbits_mask.fill_u16(kAllNumberBits);

        BitVec08x16 first_u16_mask(0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
        BitVec08x16 min_index_plus(0x70, 0x60, 0x50, 0x40, 0x30, 0x20, 0x10, 0);
        BitVec08x16 minpos;

        BitVec08x16 min_nums_8;
        BitVec08x16 min_indexs_8;

        min_nums_8.setAllZeros();
        min_indexs_8.setAllZeros();

        // Position (Box-Cell) literal
        uint32_t min_and_index;
        uint32_t min_cell_size;
        uint32_t min_cell_index;
        size_t box;
        for (box = 0; box < Boxes - 1; box += 2) {
            void * pCells16_1 = (void * )&this->init_state_.box_cell_nums[box];
            void * pCells16_2 = (void * )&this->init_state_.box_cell_nums[box + 1];
            BitVec16x16_AVX box_bits_1, box_bits_2;
            box_bits_1.loadAligned(pCells16_1);
            box_bits_2.loadAligned(pCells16_2);

            BitVec16x16_AVX disable_mask1 = box_bits_1.whichIsZeros();
            disable_mask1._and(numbits_mask);
            box_bits_1._or(disable_mask1);

            BitVec16x16_AVX disable_mask2 = box_bits_2.whichIsZeros();
            disable_mask2._and(numbits_mask);
            box_bits_2._or(disable_mask2);

            BitVec16x16 box_bits_sse_1, box_bits_sse_2;
            box_bits_1.castTo(box_bits_sse_1);
            box_bits_2.castTo(box_bits_sse_2);
            BitVec16x16 popcnt16_1 = box_bits_sse_1.popcount16<BoxSize, Numbers>();
            BitVec16x16 popcnt16_2 = box_bits_sse_2.popcount16<BoxSize, Numbers>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16_1.saveAligned(&this->count_.sizes.box_cells[box][0]);
            popcnt16_2.saveAligned(&this->count_.sizes.box_cells[box + 1][0]);
#endif
            BitVec08x16 minpos_1, minpos_2;
            popcnt16_1.minpos16<Numbers>(minpos_1);
            popcnt16_2.minpos16<Numbers>(minpos_2);

            min_nums_8 = _mm_slli_si128(min_nums_8.m128, 2);
            min_indexs_8 = _mm_slli_si128(min_indexs_8.m128, 2);

            BitVec08x16 min_num = minpos_1 & first_u16_mask;
            BitVec08x16 min_index = _mm_srli_epi32(minpos_1.m128, 16);

            min_nums_8 |= min_num;
            min_indexs_8 |= min_index;

            min_nums_8 = _mm_slli_si128(min_nums_8.m128, 2);
            min_indexs_8 = _mm_slli_si128(min_indexs_8.m128, 2);

            min_num = minpos_2 & first_u16_mask;
            min_index = _mm_srli_epi32(minpos_2.m128, 16);

            min_nums_8 |= min_num;
            min_indexs_8 |= min_index;
        }

        min_indexs_8 = _mm_add_epi16(min_indexs_8.m128, min_index_plus.m128);
        
        min_nums_8.minpos16<Numbers>(minpos);
        min_and_index = _mm_cvtsi128_si32(minpos.m128);
        min_cell_size = min_and_index & 0xFFFFUL;
        min_cell_index = min_and_index >> 16U;
        min_cell_index = min_indexs_8.extract(min_cell_index);

        if (min_cell_size == 1) {
            return LiteralInfo(1, 0, min_cell_index);
        }

        uint32_t min_cell_size_9;
        uint32_t min_cell_index_9;
        // box = 8
        {
            void * pCells16 = (void * )&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            disable_mask = box_bits.whichIsZeros();
            disable_mask._and(numbits_mask);
            box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            box_bits.castTo(box_bits_sse);
            BitVec16x16 popcnt16 = box_bits_sse.popcount16<BoxSize, Numbers>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box][0]);
#endif
            popcnt16.minpos16<Numbers>(minpos);

            min_and_index = _mm_cvtsi128_si32(minpos.m128);
            min_cell_size_9 = min_and_index & 0xFFFFUL;
            if (min_cell_size_9 < min_cell_size) {
                min_cell_size = min_cell_size_9;
                min_cell_index_9 = min_and_index >> 16U;
                min_cell_index = 8 * BoxSize16 + min_cell_index_9;
                if (min_cell_size_9 == 1) {
                    return LiteralInfo(1, 0, min_cell_index);
                }
            }
        }

        this->count_.total.min_literal_size[0] = (uint16_t)min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        BitVec16x16_AVX num_rows_mask;
        num_rows_mask.fill_u16(kAllColBits);

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            disable_mask = num_row_bits.whichIsZeros();
            disable_mask._and(num_rows_mask);

            num_row_bits._or(disable_mask);

            BitVec16x16 row_bits_sse;
            num_row_bits.castTo(row_bits_sse);

            BitVec16x16 popcnt16 = row_bits_sse.popcount16<Rows, Cols>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_rows[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.num_rows[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_rows[num] = (uint8_t)min_index;
            }
            else {
                size_t row_index = num * Rows16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 1, (uint32_t)row_index);
                }
                this->count_.indexs.num_rows[num] = (uint8_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
        }

        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        BitVec16x16_AVX num_cols_mask;
        num_cols_mask.fill_u16(kAllRowBits);

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            disable_mask = num_col_bits.whichIsZeros();
            disable_mask._and(num_cols_mask);

            num_col_bits._or(disable_mask);

            BitVec16x16 col_bits_sse;
            num_col_bits.castTo(col_bits_sse);

            BitVec16x16 popcnt16 = col_bits_sse.popcount16<Cols, Rows>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_cols[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.num_cols[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_cols[num] = (uint8_t)min_index;
            }
            else {
                size_t col_index = num * Cols16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 2, (uint32_t)col_index);
                }
                this->count_.indexs.num_cols[num] = (uint8_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
        }

        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        BitVec16x16_AVX num_box_mask;
        num_box_mask.fill_u16(kAllBoxCellBits);

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            disable_mask = num_box_bits.whichIsZeros();
            disable_mask._and(num_box_mask);

            num_box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            num_box_bits.castTo(box_bits_sse);

            BitVec16x16 popcnt16 = box_bits_sse.popcount16<Boxes, BoxSize>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_boxes[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.num_boxes[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_boxes[num] = (uint8_t)min_index;
            }
            else {
                size_t box_index = num * Boxes16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 3, (uint32_t)box_index);
                }
                this->count_.indexs.num_boxes[num] = (uint8_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
        }

        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        BitVec08x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        int min_literal_type;
        uint32_t min_literal_size = min_literal.minpos16<4>(min_literal_type);
        uint32_t min_literal_index = this->count_.total.min_literal_index[min_literal_type];

        LiteralInfo literalInfo;
        literalInfo.literal_size = min_literal_size;
        literalInfo.literal_type = (uint16_t)min_literal_type;
        literalInfo.literal_index = (uint16_t)min_literal_index;
        return literalInfo;
    }

    LiteralInfo count_literal_size_init() {
        BitVec16x16_AVX disable_mask;
        BitVec16x16_AVX numbits_mask;
        numbits_mask.fill_u16(kAllNumberBits);

        BitVec08x16 minpos;

        // Position (Box-Cell) literal
        uint32_t min_cell_size = 255;
        uint32_t min_cell_index = uint32_t(-1);
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void * )&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            BitVec16x16_AVX disable_mask = box_bits.whichIsZeros();
            disable_mask._and(numbits_mask);
            box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            box_bits.castTo(box_bits_sse);
            BitVec16x16 popcnt16 = box_bits_sse.popcount16<BoxSize, Numbers>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box][0]);
#endif
            popcnt16.minpos16<Numbers>(minpos);

            uint32_t min_and_index = _mm_cvtsi128_si32(minpos.m128);
            uint32_t min_size = min_and_index & 0xFFFFUL;
            assert(min_size > 0);

            if (min_size < min_cell_size) {
                uint32_t min_index = min_and_index >> 16U;
                min_index = (uint32_t)(box * BoxSize16 + min_index);
                if (min_size == 1) {
                    return LiteralInfo(1, 0, min_index);
                }
                min_cell_size = min_size;
                min_cell_index = min_index;
            }
        }

        this->count_.total.min_literal_size[0] = (uint16_t)min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        BitVec16x16_AVX num_rows_mask;
        num_rows_mask.fill_u16(kAllColBits);

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            disable_mask = num_row_bits.whichIsZeros();
            disable_mask._and(num_rows_mask);

            num_row_bits._or(disable_mask);

            BitVec16x16 row_bits_sse;
            num_row_bits.castTo(row_bits_sse);

            BitVec16x16 popcnt16 = row_bits_sse.popcount16<Rows, Cols>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_rows[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.num_rows[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_rows[num] = (uint8_t)min_index;
            }
            else {
                size_t row_index = num * Rows16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 1, (uint32_t)row_index);
                }
                this->count_.indexs.num_rows[num] = (uint8_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
        }

        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        BitVec16x16_AVX num_cols_mask;
        num_cols_mask.fill_u16(kAllRowBits);

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            disable_mask = num_col_bits.whichIsZeros();
            disable_mask._and(num_cols_mask);

            num_col_bits._or(disable_mask);

            BitVec16x16 col_bits_sse;
            num_col_bits.castTo(col_bits_sse);

            BitVec16x16 popcnt16 = col_bits_sse.popcount16<Cols, Rows>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_cols[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.num_cols[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_cols[num] = (uint8_t)min_index;
            }
            else {
                size_t col_index = num * Cols16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 2, (uint32_t)col_index);
                }
                this->count_.indexs.num_cols[num] = (uint8_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
        }

        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        BitVec16x16_AVX num_box_mask;
        num_box_mask.fill_u16(kAllBoxCellBits);

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            disable_mask = num_box_bits.whichIsZeros();
            disable_mask._and(num_box_mask);

            num_box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            num_box_bits.castTo(box_bits_sse);

            BitVec16x16 popcnt16 = box_bits_sse.popcount16<Boxes, BoxSize>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_boxes[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.num_boxes[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_boxes[num] = (uint8_t)min_index;
            }
            else {
                size_t box_index = num * Boxes16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 3, (uint32_t)box_index);
                }
                this->count_.indexs.num_boxes[num] = (uint8_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
        }

        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        BitVec08x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        int min_literal_type;
        uint32_t min_literal_size = min_literal.minpos16<4>(min_literal_type);
        uint32_t min_literal_index = this->count_.total.min_literal_index[min_literal_type];

        LiteralInfo literalInfo;
        literalInfo.literal_size = min_literal_size;
        literalInfo.literal_type = (uint16_t)min_literal_type;
        literalInfo.literal_index = (uint16_t)min_literal_index;
        return literalInfo;
    }

    template <size_t nLiteralType = LiteralType::Unknown>
    LiteralInfo count_literal_size_init(size_t in_box, size_t num, size_t num_bits) {
        BitVec16x16_AVX disable_mask;
        BitVec16x16_AVX numbits_mask;
        numbits_mask.fill_u16(kAllNumberBits);

        BitVec08x16 minpos;

        // Position (Box-Cell) literal
        uint32_t min_cell_size = 255;
        uint32_t min_cell_index = uint32_t(-1);
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void * )&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            BitVec16x16_AVX disable_mask = box_bits.whichIsZeros();
            disable_mask._and(numbits_mask);
            box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            box_bits.castTo(box_bits_sse);
            BitVec16x16 popcnt16 = box_bits_sse.popcount16<BoxSize, Numbers>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box][0]);
#endif
            popcnt16.minpos16<Numbers>(minpos);

            uint32_t min_and_index = _mm_cvtsi128_si32(minpos.m128);
            uint32_t min_size = min_and_index & 0xFFFFUL;
            assert(min_size > 0);

            if (min_size < min_cell_size) {
                uint32_t min_index = min_and_index >> 16U;
                min_index = (uint32_t)(box * BoxSize16 + min_index);
                if (min_size == 1) {
                    return LiteralInfo(1, 0, min_index);
                }
                min_cell_size = min_size;
                min_cell_index = min_index;
            }
        }

        this->count_.total.min_literal_size[0] = (uint16_t)min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        BitVec16x16_AVX num_rows_mask;
        num_rows_mask.fill_u16(kAllColBits);

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            disable_mask = num_row_bits.whichIsZeros();
            disable_mask._and(num_rows_mask);

            num_row_bits._or(disable_mask);

            BitVec16x16 row_bits_sse;
            num_row_bits.castTo(row_bits_sse);

            BitVec16x16 popcnt16 = row_bits_sse.popcount16<Rows, Cols>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_rows[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            assert(min_size > 0);
            this->count_.counts.num_rows[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_rows[num] = (uint8_t)min_index;
            }
            else {
                size_t row_index = num * Rows16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 1, (uint32_t)row_index);
                }
                this->count_.indexs.num_rows[num] = (uint8_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
        }

        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        BitVec16x16_AVX num_cols_mask;
        num_cols_mask.fill_u16(kAllRowBits);

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            disable_mask = num_col_bits.whichIsZeros();
            disable_mask._and(num_cols_mask);

            num_col_bits._or(disable_mask);

            BitVec16x16 col_bits_sse;
            num_col_bits.castTo(col_bits_sse);

            BitVec16x16 popcnt16 = col_bits_sse.popcount16<Cols, Rows>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_cols[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            assert(min_size > 0);
            this->count_.counts.num_cols[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_cols[num] = (uint8_t)min_index;
            }
            else {
                size_t col_index = num * Cols16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 2, (uint32_t)col_index);
                }
                this->count_.indexs.num_cols[num] = (uint8_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
        }

        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        BitVec16x16_AVX num_box_mask;
        num_box_mask.fill_u16(kAllBoxCellBits);

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            disable_mask = num_box_bits.whichIsZeros();
            disable_mask._and(num_box_mask);

            num_box_bits._or(disable_mask);

            BitVec16x16 box_bits_sse;
            num_box_bits.castTo(box_bits_sse);

            BitVec16x16 popcnt16 = box_bits_sse.popcount16<Boxes, BoxSize>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_boxes[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            assert(min_size > 0);
            this->count_.counts.num_boxes[num] = (uint16_t)min_size;
            if (likely(min_index == -1)) {
                this->count_.indexs.num_boxes[num] = (uint8_t)min_index;
            }
            else {
                size_t box_index = num * Boxes16 + min_index;
                if (min_size == 1) {
                    return LiteralInfo(1, 3, (uint32_t)box_index);
                }
                this->count_.indexs.num_boxes[num] = (uint8_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
        }

        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        BitVec08x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        int min_literal_type;
        uint32_t min_literal_size = min_literal.minpos16<4>(min_literal_type);
        uint32_t min_literal_index = this->count_.total.min_literal_index[min_literal_type];

        LiteralInfo literalInfo;
        literalInfo.literal_size = min_literal_size;
        literalInfo.literal_type = (uint16_t)min_literal_type;
        literalInfo.literal_index = (uint16_t)min_literal_index;
        return literalInfo;
    }

    LiteralInfo count_literal_size_init_old() {
        BitVec16x16_AVX disable_mask;
        BitVec16x16_AVX numbits_mask;
        numbits_mask.fill_u16(kAllNumberBits);

        // Position (Box-Cell) literal
        uint32_t min_cell_size = 255;
        uint32_t min_cell_index = uint32_t(-1);
        for (size_t box = 0; box < Boxes; box++) {
            void * pCells16 = (void * )&this->init_state_.box_cell_nums[box];
            BitVec16x16_AVX box_bits;
            box_bits.loadAligned(pCells16);

            disable_mask = box_bits.whichIsZeros();
            disable_mask._and(numbits_mask);

            box_bits._or(disable_mask);

            BitVec16x16_AVX popcnt16 = box_bits.popcount16<BoxSize, Numbers>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.box_cells[box][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Numbers>(min_cell_size, min_index);
            this->count_.counts.box_cells[box] = (uint8_t)min_size;
            if (likely(min_index != -1)) {
                size_t cell_index = box * BoxSize16 + min_index;
                this->count_.indexs.box_cells[box] = (uint8_t)cell_index;
                min_cell_index = (uint32_t)cell_index;
            }
            else {
                this->count_.indexs.box_cells[box] = (uint8_t)min_index;
            }
        }
        this->count_.total.min_literal_size[0] = (uint16_t)min_cell_size;
        this->count_.total.min_literal_index[0] = (uint16_t)min_cell_index;

        BitVec16x16_AVX num_rows_mask;
        num_rows_mask.fill_u16(kAllColBits);

        // Row literal
        uint32_t min_row_size = 255;
        uint32_t min_row_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_row_cols[num];
            BitVec16x16_AVX num_row_bits;
            num_row_bits.loadAligned(pCells16);

            disable_mask = num_row_bits.whichIsZeros();
            disable_mask._and(num_rows_mask);

            num_row_bits._or(disable_mask);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_rows[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Cols>(min_row_size, min_index);
            this->count_.counts.num_rows[num] = (uint8_t)min_size;
            if (likely(min_index != -1)) {
                size_t row_index = num * Rows16 + min_index;
                this->count_.indexs.num_rows[num] = (uint8_t)row_index;
                min_row_index = (uint32_t)row_index;
            }
            else {
                this->count_.indexs.num_rows[num] = (uint8_t)min_index;
            }
        }
        this->count_.total.min_literal_size[1] = (uint16_t)min_row_size;
        this->count_.total.min_literal_index[1] = (uint16_t)min_row_index;

        BitVec16x16_AVX num_cols_mask;
        num_cols_mask.fill_u16(kAllRowBits);

        // Col literal
        uint32_t min_col_size = 255;
        uint32_t min_col_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_col_rows[num];
            BitVec16x16_AVX num_col_bits;
            num_col_bits.loadAligned(pCells16);

            disable_mask = num_col_bits.whichIsZeros();
            disable_mask._and(num_cols_mask);

            num_col_bits._or(disable_mask);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_cols[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<Rows>(min_col_size, min_index);
            this->count_.counts.num_cols[num] = (uint8_t)min_size;
            if (likely(min_index != -1)) {
                size_t col_index = num * Cols16 + min_index;
                this->count_.indexs.num_cols[num] = (uint8_t)col_index;
                min_col_index = (uint32_t)col_index;
            }
            else {
                this->count_.indexs.num_cols[num] = (uint8_t)min_index;
            }
        }
        this->count_.total.min_literal_size[2] = (uint16_t)min_col_size;
        this->count_.total.min_literal_index[2] = (uint16_t)min_col_index;

        BitVec16x16_AVX num_box_mask;
        num_box_mask.fill_u16(kAllBoxCellBits);

        // Box-Cell literal
        uint32_t min_box_size = 255;
        uint32_t min_box_index = uint32_t(-1);
        for (size_t num = 0; num < Numbers; num++) {
            void * pCells16 = (void * )&this->init_state_.num_box_cells[num];
            BitVec16x16_AVX num_box_bits;
            num_box_bits.loadAligned(pCells16);

            disable_mask = num_box_bits.whichIsZeros();
            disable_mask._and(num_box_mask);

            num_box_bits._or(disable_mask);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();
#if V4A_SAVE_COUNT_SIZE
            popcnt16.saveAligned(&this->count_.sizes.num_boxes[num][0]);
#endif
            int min_index = -1;
            uint32_t min_size = popcnt16.minpos16<BoxSize>(min_box_size, min_index);
            this->count_.counts.num_boxes[num] = (uint8_t)min_size;
            if (likely(min_index != -1)) {
                size_t box_index = num * Boxes16 + min_index;
                this->count_.indexs.num_boxes[num] = (uint8_t)box_index;
                min_box_index = (uint32_t)box_index;
            }
            else {
                this->count_.indexs.num_boxes[num] = (uint8_t)min_index;
            }
        }
        this->count_.total.min_literal_size[3] = (uint16_t)min_box_size;
        this->count_.total.min_literal_index[3] = (uint16_t)min_box_index;

        BitVec08x16 min_literal;
        min_literal.loadAligned(&this->count_.total.min_literal_size[0]);
        int min_literal_type;
        uint32_t min_literal_size = min_literal.minpos16<4>(min_literal_type);
        uint32_t min_literal_index = this->count_.total.min_literal_index[min_literal_type];

        LiteralInfo literalInfo;
        literalInfo.literal_size = min_literal_size;
        literalInfo.literal_type = (uint16_t)min_literal_type;
        literalInfo.literal_index = (uint16_t)min_literal_index;
        return literalInfo;
    }

    void init_board(Board & board) {
        init_literal_info();
#if 1
        BitVec16x16_AVX mask_1, mask_2, mask_3, mask_4;
        mask_1.fill_u16(kAllNumberBits);
        for (size_t box = 0; box < Boxes; box++) {
            mask_1.saveAligned((void *)&this->init_state_.box_cell_nums[box]);
        }

        mask_2.fill_u16(kAllColBits);
        mask_3.fill_u16(kAllRowBits);
        mask_4.fill_u16(kAllBoxCellBits);
        for (size_t num = 0; num < Numbers; num++) {
            mask_2.saveAligned((void *)&this->init_state_.num_row_cols[num]);
            mask_3.saveAligned((void *)&this->init_state_.num_col_rows[num]);
            mask_4.saveAligned((void *)&this->init_state_.num_box_cells[num]);
        }
#else
        this->init_state_.box_cell_nums.fill(kAllNumberBits);
        this->init_state_.num_row_cols.fill(kAllColBits);
        this->init_state_.num_col_rows.fill(kAllRowBits);
        this->init_state_.num_box_cells.fill(kAllBoxCellBits);
#endif
        if (kSearchMode > SearchMode::OneAnswer) {
            this->answers_.clear();
        }

        size_t empties = this->calc_empties(board);
        this->empties_ = empties;

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            size_t box_y = (row / BoxCellsY) * BoxCountX;
            size_t cell_y = (row % BoxCellsY) * BoxCellsX;
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board.cells[pos];
                if (val != '.') {
                    size_t box_x = col / BoxCellsX;
                    size_t box = box_y + box_x;
                    size_t cell_x = col % BoxCellsX;
                    size_t cell = cell_y + cell_x;
                    size_t num = val - '1';

                    this->fillNum(this->init_state_, row, col, box, cell, num);
                    this->updateNeighborCells(this->init_state_, pos, box, cell, num);
                }
                pos++;
            }
        }
        assert(pos == BoardSize);
    }

public:
    size_t search_unique_candidate(Board & board, size_t empties,
                                   InitState & init_state,
                                   LiteralInfo & literalInfo) {
        size_t pos, row, col, box, cell, num, num_bits;

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

                num_bits = init_state.box_cell_nums[box][cell].to_ulong();
                assert(num_bits != 0);
                assert((num_bits & (num_bits - 1)) == 0);
                num = BitUtils::bsf(num_bits);

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    //num_bits = this->fillNum(init_state, row, col, box, cell, num);
                    this->updateNeighborCells<LiteralType::BoxCellNums>(init_state, pos, box, cell, num);

                    literalInfo = this->count_literal_size_init<LiteralType::BoxCellNums>(box, num, 0);
                    if (literalInfo.literal_size == 1) {
                        return this->search_unique_candidate(board, empties, init_state, literalInfo);
                    }
                }

                break;
            }

            case LiteralType::NumRowCols:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Rows16);

                num = literal / Rows16;
                row = literal % Rows16;

                size_t col_bits = init_state.num_row_cols[num][row].reset_and_get();
                assert(col_bits != 0);
                assert((col_bits & (col_bits - 1)) == 0);
                col = BitUtils::bsf(col_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    num_bits = this->fillNum(init_state, row, col, box, cell, num);
                    this->updateNeighborCells<LiteralType::NumRowCols>(init_state, pos, box, cell, num);

                    literalInfo = this->count_literal_size_init<LiteralType::NumRowCols>(box, num, num_bits);
                    if (literalInfo.literal_size == 1) {
                        return this->search_unique_candidate(board, empties, init_state, literalInfo);
                    }
                }

                break;
            }

            case LiteralType::NumColRows:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Cols16);

                num = literal / Cols16;
                col = literal % Cols16;

                size_t row_bits = init_state.num_col_rows[num][col].reset_and_get();
                assert(row_bits != 0);
                assert((row_bits & (row_bits - 1)) == 0);
                row = BitUtils::bsf(row_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    num_bits = this->fillNum(init_state, row, col, box, cell, num);
                    this->updateNeighborCells<LiteralType::NumColRows>(init_state, pos, box, cell, num);

                    literalInfo = this->count_literal_size_init<LiteralType::NumColRows>(box, num, num_bits);
                    if (literalInfo.literal_size == 1) {
                        return this->search_unique_candidate(board, empties, init_state, literalInfo);
                    }
                }

                break;
            }

            case LiteralType::NumBoxCells:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Boxes16);

                num = literal / Boxes16;
                box = literal % Boxes16;

                size_t cell_bits = init_state.num_box_cells[num][box].reset_and_get();
                assert(cell_bits != 0);
                assert((cell_bits & (cell_bits - 1)) == 0);
                cell = BitUtils::bsf(cell_bits);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box * BoxSize16 + cell];
                row = boxesInfo.row;
                col = boxesInfo.col;
                pos = boxesInfo.pos;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    num_bits = this->fillNum(init_state, row, col, box, cell, num);
                    this->updateNeighborCells<LiteralType::NumBoxCells>(init_state, pos, box, cell, num);

                    literalInfo = this->count_literal_size_init<LiteralType::NumBoxCells>(box, num, num_bits);
                    if (literalInfo.literal_size == 1) {
                        return this->search_unique_candidate(board, empties, init_state, literalInfo);
                    }
                }

                break;
            }

            default:
                assert(false);
                break;
        }

        return empties;
    }

    void do_unique_literal(Board & board, InitState & init_state, const LiteralInfo & literalInfo) {
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
                assert(num_bits != 0);
                assert((num_bits & (num_bits - 1)) == 0);
                num = BitUtils::bsf(num_bits);

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                //this->fillNum(init_state, row, col, box, cell, num);
                this->updateNeighborCells<LiteralType::BoxCellNums>(init_state, pos, box, cell, num);

                break;
            }

            case LiteralType::NumRowCols:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Rows16);

                num = literal / Rows16;
                row = literal % Rows16;

                size_t col_bits = init_state.num_row_cols[num][row].reset_and_get();
                assert(col_bits != 0);
                assert((col_bits & (col_bits - 1)) == 0);
                col = BitUtils::bsf(col_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->fillNum(init_state, row, col, box, cell, num);
                this->updateNeighborCells<LiteralType::NumRowCols>(init_state, pos, box, cell, num);

                break;
            }

            case LiteralType::NumColRows:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Cols16);

                num = literal / Cols16;
                col = literal % Cols16;

                size_t row_bits = init_state.num_col_rows[num][col].reset_and_get();
                assert(row_bits != 0);
                assert((row_bits & (row_bits - 1)) == 0);
                row = BitUtils::bsf(row_bits);
                pos = row * Cols + col;

                const CellInfo & cellInfo = Sudoku::cell_info[pos];
                box = cellInfo.box;
                cell = cellInfo.cell;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->fillNum(init_state, row, col, box, cell, num);
                this->updateNeighborCells<LiteralType::NumColRows>(init_state, pos, box, cell, num);

                break;
            }

            case LiteralType::NumBoxCells:
            {
                size_t literal = literalInfo.literal_index;
                assert(literal < Numbers * Boxes16);

                num = literal / Boxes16;
                box = literal % Boxes16;

                size_t cell_bits = init_state.num_box_cells[num][box].reset_and_get();
                assert(cell_bits != 0);
                assert((cell_bits & (cell_bits - 1)) == 0);
                cell = BitUtils::bsf(cell_bits);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box * BoxSize16 + cell];
                row = boxesInfo.row;
                col = boxesInfo.col;
                pos = boxesInfo.pos;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->fillNum(init_state, row, col, box, cell, num);
                this->updateNeighborCells<LiteralType::NumBoxCells>(init_state, pos, box, cell, num);

                break;
            }

            default:
                assert(false);
                break;
        }
    }

    bool search(Board & board, size_t empties, const LiteralInfo & literalInfo) {
        if (empties == 0) {
            if (kSearchMode > SearchMode::OneAnswer) {
                this->answers_.push_back(board);
                if (kSearchMode == SearchMode::MoreThanOneAnswer) {
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

    bool solve(Board & board) {
        this->init_board(board);

        size_t empties = this->calc_empties(board);

        LiteralInfo literalInfo = this->count_literal_size_init();
        assert(literalInfo.literal_size > 0);
#if 1
        assert(literalInfo.literal_size > 0);
        while (literalInfo.literal_size == 1) {
            this->do_unique_literal(board, this->init_state_, literalInfo);
            literalInfo = this->count_literal_size_init();
            assert(literalInfo.literal_size > 0);
            empties--;
            if (empties == 0)
                break;
        }
#else
        if (literalInfo.literal_size == 1) {
            empties = this->search_unique_candidate(board, empties, this->init_state_, literalInfo);
        }
#endif

        this->min_info_ = literalInfo;

        bool success = this->search(board, empties, this->min_info_);
        return success;
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        basic_solver_t::display_result<kSearchMode>(board, elapsed_time, print_answer, print_all_answers);
    }
};

Solver::StaticData Solver::Static;

} // namespace v4a
} // namespace gzSudoku

#endif // GZ_SUDOKU_SOLVER_V4A_H
