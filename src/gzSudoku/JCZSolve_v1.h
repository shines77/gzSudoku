
#ifndef GZ_SUDOKU_JCZSOLVE_V1_H
#define GZ_SUDOKU_JCZSOLVE_V1_H

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

#define JCZ_USE_SIMD_INIT   1

namespace gzSudoku {
namespace JCZ {
namespace v1 {

static const size_t kSearchMode = SearchMode::OneAnswer;

class Solver : public BasicSolver {
public:
    typedef BasicSolver                         basic_solver_t;
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

    static const bool kAllDimIsSame = Sudoku::kAllDimIsSame;

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

    struct alignas(32) InitState {
        PackedBitSet3D<Numbers, Rows16, Cols16>     num_row_cols;     // [num][row][col]
        PackedBitSet3D<Numbers, Cols16, Rows16>     num_col_rows;     // [num][col][row]
        PackedBitSet3D<Numbers, Boxes16, BoxSize16> num_box_cells;    // [num][box][cell]

        PackedBitSet2D<Rows16, Cols16>              row_solved;
        PackedBitSet2D<Cols16, Rows16>              col_solved;
        PackedBitSet2D<Boxes16, BoxSize16>          box_solved;
    };

    struct alignas(32) State {
        PackedBitSet3D<Numbers, Rows16, Cols16>     num_row_cols;     // [num][row][col]
        PackedBitSet3D<Numbers, Cols16, Rows16>     num_col_rows;     // [num][col][row]
        PackedBitSet3D<Numbers, Boxes16, BoxSize16> num_box_cells;    // [num][box][cell]

        PackedBitSet2D<Rows16, Cols16>              row_solved;
        PackedBitSet2D<Cols16, Rows16>              col_solved;
        PackedBitSet2D<Boxes16, BoxSize16>          box_solved;

        BandConfigure h_band[BoxCountX];
        BandConfigure v_band[BoxCountY];
    };

    union LiteralInfoEx {
        struct {
            uint32_t literal_size;
            uint16_t literal_type;
            uint16_t literal_index;
        };

        uint64_t value;

        LiteralInfoEx(uint64_t _value = 0) : value(_value) {}
        LiteralInfoEx(uint32_t size, uint32_t type, uint32_t index)
            : literal_size(size), literal_type((uint16_t)type), literal_index((uint16_t)index) {}
        LiteralInfoEx(const LiteralInfoEx & src) : value(src.value) {}

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
        LiteralInfo(uint32_t type, uint32_t index)
            : literal_type(type), literal_index(index) {}
        LiteralInfo(const LiteralInfo & src) : value(src.value) {}

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

    template <size_t nBoxCountX, size_t nBoxCountY>
    struct PeerBoxes {
        static const uint32_t kBoxesCount = (uint32_t)((nBoxCountX - 1) + (nBoxCountY - 1));

        uint32_t boxes_count() const { return kBoxesCount; }

        int boxes[kBoxesCount];
    };

    typedef PeerBoxes<BoxCountX, BoxCountY>     peer_boxes_t;

    struct alignas(32) StaticData {
        PackedBitSet3D<BoardSize, Rows16, Cols16>       num_row_mask;
        PackedBitSet3D<BoardSize, Cols16, Rows16>       num_col_mask;
        PackedBitSet3D<BoardSize, Boxes16, BoxSize16>   num_box_mask;

        PackedBitSet3D<BoardSize, Rows16, Cols16>       row_fill_mask;
        PackedBitSet3D<BoardSize, Cols16, Rows16>       col_fill_mask;
        PackedBitSet3D<BoardSize, Boxes16, BoxSize16>   box_fill_mask;
        
        peer_boxes_t    peer_boxes[Boxes];
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

    static StaticData Static;

public:
    Solver() : basic_solver_t() {
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
                for (size_t box_i = 0; box_i < BoxCellsX; box_i++) {
                    if (box_i != box_x) {
                        peerBoxes.boxes[index++] = uint32_t(box_y * BoxCellsX + box_i);
                    }
                }
                for (size_t box_j = 0; box_j < BoxCellsY; box_j++) {
                    if (box_j != box_y) {
                        peerBoxes.boxes[index++] = uint32_t(box_j * BoxCellsX + box_x);
                    }
                }
                assert(index == peerBoxes.boxes_count());

                //std::sort(&peerBoxes.boxes[1], &peerBoxes.boxes[peerBoxes.boxes_count()]);
                Static.peer_boxes[box] = peerBoxes;
            }
        }
    }

    static void make_flip_mask(size_t fill_pos, size_t row, size_t col) {
        PackedBitSet2D<Rows16, Cols16> & rows_mask      = Static.num_row_mask[fill_pos];
        PackedBitSet2D<Cols16, Rows16> & cols_mask      = Static.num_col_mask[fill_pos];
        PackedBitSet2D<Boxes16, BoxSize16> & boxes_mask = Static.num_box_mask[fill_pos];

        const CellInfo * pCellInfo = Sudoku::cell_info;
        size_t box = pCellInfo[fill_pos].box;
        size_t cell = pCellInfo[fill_pos].cell;

        Static.row_fill_mask[fill_pos][row].set(col);
        Static.col_fill_mask[fill_pos][col].set(row);
        Static.box_fill_mask[fill_pos][box].set(cell);

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

            cols_mask[col].reset(row);
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

            box = pCellInfo[fill_pos].box;
            cell = pCellInfo[fill_pos].cell;
            boxes_mask[box].reset(cell);
        }
    }

    static void init_flip_mask() {
        Static.num_row_mask.reset();
        Static.num_col_mask.reset();
        Static.num_box_mask.reset();

        Static.row_fill_mask.reset();
        Static.col_fill_mask.reset();
        Static.box_fill_mask.reset();

        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_flip_mask(fill_pos, row, col);
                fill_pos++;
            }
        }
    }

    static void init_mask() {
        printf("JCZ::v1::Solve::StaticData::init_mask()\n");

        init_peer_boxes();
        init_flip_mask();
    }

    void init_board(Board & board) {
#if JCZ_USE_SIMD_INIT
        BitVec16x16_AVX full_mask;
        full_mask.fill_u16(kAllColBits);

        for (size_t num = 0; num < Numbers; num++) {
            full_mask.saveAligned((void *)&this->init_state_.num_row_cols[num]);
            full_mask.saveAligned((void *)&this->init_state_.num_col_rows[num]);
            full_mask.saveAligned((void *)&this->init_state_.num_box_cells[num]);
        }

        BitVec16x16_AVX zeros;
        zeros.setAllZeros();
        zeros.saveAligned((void *)&this->init_state_.row_solved);
        zeros.saveAligned((void *)&this->init_state_.col_solved);
        zeros.saveAligned((void *)&this->init_state_.box_solved);
#else
        this->init_state_.num_row_cols.fill(kAllColBits);
        this->init_state_.num_col_rows.fill(kAllRowBits);
        this->init_state_.num_box_cells.fill(kAllBoxCellBits);

        this->init_state_.row_solved.reset();
        this->init_state_.col_solved.reset();
        this->init_state_.box_solved.reset();
#endif
        if (kSearchMode > SearchMode::OneAnswer) {
            this->answers_.clear();
        }

        size_t pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                unsigned char val = board.cells[pos];
                if (val != '.') {
                    size_t num = val - '1';
                    this->update_peer_cells(this->init_state_, pos, num);
                }
                pos++;
            }
        }
        assert(pos == BoardSize);
    }

    inline void update_peer_cells(InitState & init_state, size_t fill_pos, size_t fill_num) {
        size_t row = fill_pos / Cols;
        size_t col = fill_pos % Cols;
        assert(init_state.num_row_cols[fill_num][row].test(col));
        assert(init_state.num_col_rows[fill_num][col].test(row));
        //assert(init_state.num_box_cells[fill_num][box].test(cell));

        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;

        BitVec16x16_AVX row_fill_mask, row_solved;
        pCells16 = (void *)&init_state.row_solved;
        pMask16 = (void *)&Static.row_fill_mask[fill_pos];
        row_solved.loadAligned(pCells16);
        row_fill_mask.loadAligned(pMask16);
        row_solved |= row_fill_mask;
        row_solved.saveAligned(pCells16);

        BitVec16x16_AVX col_fill_mask, col_solved;
        pCells16 = (void *)&init_state.col_solved;
        pMask16 = (void *)&Static.col_fill_mask[fill_pos];
        col_solved.loadAligned(pCells16);
        col_fill_mask.loadAligned(pMask16);
        col_solved |= col_fill_mask;
        col_solved.saveAligned(pCells16);

        BitVec16x16_AVX box_fill_mask, box_solved;
        pCells16 = (void *)&init_state.box_solved;
        pMask16 = (void *)&Static.box_fill_mask[fill_pos];
        box_solved.loadAligned(pCells16);
        box_fill_mask.loadAligned(pMask16);
        box_solved |= box_fill_mask;
        box_solved.saveAligned(pCells16);

        for (size_t num = 0; num < Numbers; num++) {
            //init_state.num_row_cols[num][row].reset(col);
            //init_state.num_col_rows[num][col].reset(row);
            //init_state.num_box_cells[num][box].reset(cell);
            pCells16 = (void *)&init_state.num_row_cols[num];
            cells16.loadAligned(pCells16);
            cells16.and_not(row_fill_mask);
            cells16.saveAligned(pCells16);

            pCells16 = (void *)&init_state.num_col_rows[num];
            cells16.loadAligned(pCells16);
            cells16.and_not(col_fill_mask);
            cells16.saveAligned(pCells16);

            pCells16 = (void *)&init_state.num_box_cells[num];
            cells16.loadAligned(pCells16);
            cells16.and_not(box_fill_mask);
            cells16.saveAligned(pCells16);
        }

        //init_state.num_row_cols[num] &= ~Static.num_row_mask[fill_pos];
        //init_state.num_col_rows[num] &= ~Static.num_col_mask[fill_pos];
        //init_state.num_box_cells[num] &= ~Static.num_box_cells[fill_pos];

        // LiteralType::NumRowCols
        {
            pCells16 = (void *)&init_state.num_row_cols[fill_num];
            pMask16 = (void *)&Static.num_row_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16._or(row_fill_mask);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumColRows
        {
            pCells16 = (void *)&init_state.num_col_rows[fill_num];
            pMask16 = (void *)&Static.num_col_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16._or(col_fill_mask);
            cells16.saveAligned(pCells16);
        }

        // LiteralType::NumBoxCells
        {
            pCells16 = (void *)&init_state.num_box_cells[fill_num];
            pMask16 = (void *)&Static.num_box_mask[fill_pos];
            cells16.loadAligned(pCells16);
            mask16.loadAligned(pMask16);
            cells16.and_not(mask16);
            cells16._or(box_fill_mask);
            cells16.saveAligned(pCells16);
        }
    }

    inline void update_peer_cells_2(InitState & init_state, size_t fill_pos, size_t fill_num) {
        size_t row = fill_pos / Cols;
        size_t col = fill_pos % Cols;
        assert(init_state.num_row_cols[fill_num][row].test(col));
        assert(init_state.num_col_rows[fill_num][col].test(row));

        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;

        BitVec16x16_AVX row_mask, col_mask;
        row_mask.loadAligned((void *)&Static.row_fill_mask[fill_pos]);
        col_mask.loadAligned((void *)&Static.col_fill_mask[fill_pos]);

        BitVec16x16_AVX row_solved_bits, col_solved_bits;
        pCells16 = (void *)&init_state.row_solved;
        row_solved_bits.loadAligned(pCells16);
        row_solved_bits |= row_mask;
        row_solved_bits.saveAligned(pCells16);

        pCells16 = (void *)&init_state.col_solved;
        col_solved_bits.loadAligned(pCells16);
        col_solved_bits |= col_mask;
        col_solved_bits.saveAligned(pCells16);

        for (size_t num = 0; num < Numbers; num++) {
            if (num != fill_num) {
                //init_state.num_row_cols[num][row].reset(col);
                //init_state.num_col_rows[num][col].reset(row);

                pCells16 = (void *)&init_state.num_row_cols[num];
                cells16.loadAligned(pCells16);
                cells16.and_not(row_mask);
                cells16.saveAligned(pCells16);

                pCells16 = (void *)&init_state.num_col_rows[num];
                cells16.loadAligned(pCells16);
                cells16.and_not(col_mask);
                cells16.saveAligned(pCells16);
            }
            else {
                //init_state.num_row_cols[num] &= Static.num_row_mask[fill_pos];
                //init_state.num_col_rows[num] &= Static.num_col_mask[fill_pos];

                // LiteralType::NumRowCols
                {
                    pCells16 = (void *)&init_state.num_row_cols[num];
                    pMask16 = (void *)&Static.num_row_mask[fill_pos];
                    cells16.loadAligned(pCells16);
                    mask16.loadAligned(pMask16);
                    cells16.and_not(mask16);
                    cells16._or(row_mask);
                    cells16.saveAligned(pCells16);
                }

                // LiteralType::NumColRows
                {
                    pCells16 = (void *)&init_state.num_col_rows[num];
                    pMask16 = (void *)&Static.num_col_mask[fill_pos];
                    cells16.loadAligned(pCells16);
                    mask16.loadAligned(pMask16);
                    cells16.and_not(mask16);
                    cells16._or(col_mask);
                    cells16.saveAligned(pCells16);
                }
            }
        }
    }

    LiteralInfo find_single_literal() {
        BitVec16x16_AVX unique_mask;
        unique_mask.fill_u16(1);

        BitVec16x16_AVX row_solved;
        void * pCells16 = (void *)&this->init_state_.row_solved;
        row_solved.loadAligned(pCells16);

        // Row literal
        for (size_t num = 0; num < Numbers; num++) {
            BitVec16x16_AVX num_row_bits;
            pCells16 = (void *)&this->init_state_.num_row_cols[num];
            num_row_bits.loadAligned(pCells16);
            num_row_bits.and_not(row_solved);

            BitVec16x16_AVX popcnt16 = num_row_bits.popcount16<Rows, Cols>();

            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t row_index = (uint32_t)(num * Rows16 + min_index);
                return LiteralInfo(LiteralType::NumRowCols, row_index);
            }
        }

        BitVec16x16_AVX col_solved;
        pCells16 = (void *)&this->init_state_.col_solved;
        col_solved.loadAligned(pCells16);

        // Col literal
        for (size_t num = 0; num < Numbers; num++) {
            BitVec16x16_AVX num_col_bits;
            pCells16 = (void *)&this->init_state_.num_col_rows[num];
            num_col_bits.loadAligned(pCells16);
            num_col_bits.and_not(col_solved);

            BitVec16x16_AVX popcnt16 = num_col_bits.popcount16<Cols, Rows>();

            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t col_index = (uint32_t)(num * Cols16 + min_index);
                return LiteralInfo(LiteralType::NumColRows, col_index);
            }
        }

        BitVec16x16_AVX box_solved;
        pCells16 = (void *)&this->init_state_.box_solved;
        box_solved.loadAligned(pCells16);

        // Box literal
        for (size_t num = 0; num < Numbers; num++) {
            BitVec16x16_AVX num_box_bits;
            pCells16 = (void *)&this->init_state_.num_box_cells[num];
            num_box_bits.loadAligned(pCells16);
            num_box_bits.and_not(box_solved);

            BitVec16x16_AVX popcnt16 = num_box_bits.popcount16<Boxes, BoxSize>();

            int min_index = popcnt16.indexOfIsEqual16<false>(unique_mask);
            assert(min_index >= -1 && min_index < 16);

            if (min_index != -1) {
                uint32_t box_index = (uint32_t)(num * Boxes16 + min_index);
                return LiteralInfo(LiteralType::NumBoxCells, box_index);
            }
        }

        return LiteralInfo(-1);
    }

    int find_single_candidate_cells(Board & board) {
        BitVec16x16_AVX R1, R2, R3;
        R1.setAllZeros();
        R2.setAllZeros();
        R3.setAllZeros();
#if 1
        for (size_t num = 0; num < Numbers; num++) {
            BitVec16x16_AVX row_bits;
            void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
            row_bits.loadAligned(pCells16);

            R3 |= R2 & row_bits;
            R2 |= R1 & row_bits;
            R1 |= row_bits;
        }
#else
        //
#endif
        BitVec16x16_AVX full_mask;
        full_mask.fill_u16(kAllNumberBits);
        bool is_legal = R1.isEqual(full_mask);
        if (!is_legal) return -1;

        BitVec16x16_AVX solved_bits, zeros;
        solved_bits.loadAligned((void *)&this->init_state_.row_solved);
        zeros.setAllZeros();

        R1.and_not(R2);
        R2.and_not(R3);
        R1.and_not(solved_bits);

        int cell_count = 0;
        if (R1.isNotAllZeros()) {
#if 0
            BitVec16x16_AVX neg_R1, low_bit;
            neg_R1 = _mm256_sub_epi64(zeros.m256, R1.m256);
            low_bit = R1 & neg_R1;
            R1 ^= low_bit;
#endif
            int R1_count = R1.popcount();
            assert(R1_count > 0);
            for (size_t num = 0; num < Numbers; num++) {
                BitVec16x16_AVX row_bits;
                void * pCells16 = (void *)&this->init_state_.num_row_cols[num];
                row_bits.loadAligned(pCells16);

                row_bits &= R1;
                if (row_bits.isNotAllZeros()) {
                    // Find the position of low bit, and fill the num.
                    alignas(32) IntVec256 row_vec;
                    row_bits.saveAligned((void *)&row_vec);

 #if defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
  || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)
                    for (size_t i = 0; i < 3; i++) {
                        uint64_t bits64 = row_vec.u64[i];
                        while (bits64 != 0) {
                            size_t bit_pos = BitUtils::bsf64(bits64);
                            size_t row = bit_pos / Cols16 + i * 4;
                            size_t col = bit_pos % Cols16;

                            size_t pos = row * Cols + col;

                            assert(board.cells[pos] == '.');
                            board.cells[pos] = (char)(num + '1');

                            this->update_peer_cells(this->init_state_, pos, num);
                            cell_count++;

                            uint64_t bit = BitUtils::ls1b64(bits64);
                            bits64 ^= bit;
                        }
                    }
#else
                    for (size_t i = 0; i < 5; i++) {
                        uint32_t bits32 = row_vec.u32[i];
                        while (bits32 != 0) {
                            size_t bit_pos = BitUtils::bsf32(bits32);
                            size_t row = bit_pos / Cols16 + i * 2;
                            size_t col = bit_pos % Cols16;

                            size_t pos = row * Cols + col;

                            assert(board.cells[pos] == '.');
                            board.cells[pos] = (char)(num + '1');

                            this->update_peer_cells(this->init_state_, pos, num);
                            cell_count++;

                            uint32_t bit = BitUtils::ls1b32(bits32);
                            bits32 ^= bit;
                        }
                    }
#endif
                    if (cell_count >= R1_count) {
                        assert(cell_count > 0);
                        break;
                    }
                }
            }
            assert(cell_count > 0);
        }

        return cell_count;
    }

    void do_single_literal(InitState & init_state, Board & board, LiteralInfo literalInfo) {
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

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells(init_state, pos, num);
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

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells(init_state, pos, num);
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
                pos = boxesInfo.pos;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                this->update_peer_cells(init_state, pos, num);
                break;
            }

            default:
                assert(false);
                break;
        }
    }

    bool check_and_do_single_literal(InitState & init_state, Board & board, LiteralInfo literalInfo) {
        size_t pos, row, col, box, cell, num;

        switch (literalInfo.literal_type) {
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

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells(init_state, pos, num);
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

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells(init_state, pos, num);
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
                    pos = boxesInfo.pos;

                    assert(board.cells[pos] == '.');
                    board.cells[pos] = (char)(num + '1');

                    this->update_peer_cells(init_state, pos, num);
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

    size_t search_single_literal(InitState & init_state,
                                 Board & board, size_t empties,
                                 LiteralInfo literalInfo) {
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

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    this->update_peer_cells(init_state, pos, num);

                    LiteralInfo nextLI = this->find_single_literal();
                    if (nextLI.isValid()) {
                        return this->search_single_literal(init_state, board, empties, nextLI);
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

                size_t row_bits = init_state.num_col_rows[num][col].to_ulong();
                assert(row_bits != 0);
                assert((row_bits & (row_bits - 1)) == 0);
                row = BitUtils::bsf(row_bits);
                pos = row * Cols + col;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    this->update_peer_cells(init_state, pos, num);

                    LiteralInfo nextLI = this->find_single_literal();
                    if (nextLI.isValid()) {
                        return this->search_single_literal(init_state, board, empties, nextLI);
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

                size_t cell_bits = init_state.num_box_cells[num][box].to_ulong();
                assert(cell_bits != 0);
                assert((cell_bits & (cell_bits - 1)) == 0);
                cell = BitUtils::bsf(cell_bits);

                const BoxesInfo & boxesInfo = Sudoku::boxes_info16[box * BoxSize16 + cell];
                pos = boxesInfo.pos;

                assert(board.cells[pos] == '.');
                board.cells[pos] = (char)(num + '1');

                empties--;
                if (empties > 0) {
                    this->update_peer_cells(init_state, pos, num);

                    LiteralInfo nextLI = this->find_single_literal();
                    if (nextLI.isValid()) {
                        return this->search_single_literal(init_state, board, empties, nextLI);
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

public:
    bool search(Board & board, ptrdiff_t empties, const LiteralInfoEx & literalInfo) {
        if (empties <= 0) {
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

        ptrdiff_t empties = this->calc_empties(board);
        assert(empties >= Sudoku::kMinInitCandidates);
#if 1
        LiteralInfo literalInfo = this->find_single_literal();

        while (literalInfo.isValid()) {
            this->do_single_literal(this->init_state_, board, literalInfo);
            empties--;
            if (empties <= 0)
                break;
Next_Search:
            literalInfo = this->find_single_literal();
            if (!literalInfo.isValid()) {
                int single_cells = find_single_candidate_cells(board);
                if (single_cells <= 0)
                    break;
                empties -= single_cells;
                if (empties <= 0)
                    break;
                goto Next_Search;
            }
        }
#elif 0
        LiteralInfo literalInfo = this->find_single_literal();

        while (literalInfo.isValid()) {
            bool is_legal = this->check_and_do_single_literal(this->init_state_, board, literalInfo);
            if (is_legal) {
                literalInfo = this->find_single_literal();
                if (!literalInfo.isValid()) {
                    int has_single_cells = find_single_candidate_cells(board);
                    if (has_single_cells != 1)
                        break;
                }
                empties--;
                if (empties <= 0)
                    break;
            }
            else break;
        }
#elif 0
        LiteralInfo literalInfo = this->find_single_literal();

        if (literalInfo.isValid()) {
            empties = this->search_single_literal(this->init_state_, board, empties, literalInfo);
        }
#endif
        LiteralInfoEx LI = literalInfo.toLiteralInfoEx(1);
        bool success = this->search(board, empties, LI);
        return success;
    }

    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        basic_solver_t::display_result<kSearchMode>(board, elapsed_time, print_answer, print_all_answers);
    }
};

Solver::StaticData Solver::Static;

} // namespace v1
} // namespace JCZ
} // namespace gzSudoku

#endif // GZ_SUDOKU_JCZSOLVE_V1_H
