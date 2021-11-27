
#ifndef GZ_SUDOKU_JCZSOLVE_EX_H
#define GZ_SUDOKU_JCZSOLVE_EX_H

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

#define JCZ_USE_SIMD_INIT   0

namespace gzSudoku {
namespace JCZ {

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

    struct BandConfigure {
        alignas(32) PackedBitSet2D<Config8, Numbers16> config;          // Band[config][num]
        alignas(16) PackedBitSet2D<Config8, Numbers16> exclude;         // Band[config][num]
    };

    struct InitState {
        alignas(32) PackedBitSet3D<Numbers, Rows16, Cols16>       num_row_cols;     // [num][row][col]
        alignas(32) PackedBitSet3D<Numbers, Cols16, Rows16>       num_col_rows;     // [num][col][row]
    };

    struct State {
        BandConfigure h_band[BoxCountX];
        BandConfigure v_band[BoxCountY];

        alignas(32) PackedBitSet3D<Boxes, BoxSize16, Numbers16>   box_cell_nums;    // [box][cell][num]
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

    struct StaticData {
        alignas(32) PackedBitSet3D<BoardSize, Rows16, Cols16>       num_row_mask;
        alignas(32) PackedBitSet3D<BoardSize, Cols16, Rows16>       num_col_mask;

        bool                    mask_is_inited;
        peer_boxes_t            peer_boxes[Boxes];

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

        const CellInfo * pCellInfo = Sudoku::cell_info;
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
    }

    static void init_flip_mask() {
        Static.num_row_mask.reset();
        Static.num_col_mask.reset();

        size_t fill_pos = 0;
        for (size_t row = 0; row < Rows; row++) {
            for (size_t col = 0; col < Cols; col++) {
                make_flip_mask(fill_pos, row, col);
                fill_pos++;
            }
        }
    }

    static void init_mask() {
        printf("JCZSolve::StaticData::init_mask()\n");

        init_peer_boxes();
        init_flip_mask();
    }

    void init_board(Board & board) {
#if JCZ_USE_SIMD_INIT
        BitVec16x16_AVX full_mask;
        full_mask.fill_u16(kAllRowBits);

        for (size_t num = 0; num < Numbers; num++) {
            full_mask.saveAligned((void *)&this->init_state_.num_row_cols[num]);
            full_mask.saveAligned((void *)&this->init_state_.num_col_rows[num]);
        }
#else
        this->init_state_.num_row_cols.fill(kAllColBits);
        this->init_state_.num_col_rows.fill(kAllRowBits);
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

                    this->fill_num(this->init_state_, row, col, num);
                    this->update_peer_cells(this->init_state_, pos, num);
                }
                pos++;
            }
        }
        assert(pos == BoardSize);
    }

    inline void fill_num(InitState & init_state, size_t row, size_t col, size_t fill_num) {
        assert(init_state.num_row_cols[fill_num][row].test(col));
        assert(init_state.num_col_rows[fill_num][col].test(row));

        for (size_t num = 0; num < Numbers; num++) {
            init_state.num_row_cols[num][row].reset(col);
            init_state.num_col_rows[num][col].reset(row);
        }
    }

    template <size_t nLiteralType = LiteralType::Unknown>
    inline void update_peer_cells(InitState & init_state, size_t fill_pos, size_t num) {
        BitVec16x16_AVX cells16, mask16;
        void * pCells16, * pMask16;

        //init_state.num_row_cols[num] &= Static.num_row_mask[fill_pos];
        //init_state.num_col_rows[num] &= Static.num_col_mask[fill_pos];

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
    }

    LiteralInfo find_single_literal() {
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
                return LiteralInfo(LiteralType::NumRowCols, row_index);
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
                return LiteralInfo(LiteralType::NumColRows, col_index);
            }
        }

        return LiteralInfo(-1);
    }

    void do_single_literal(InitState & init_state, Board & board, LiteralInfo literalInfo) {
        size_t pos, row, col, num;

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

                this->fill_num(init_state, row, col, num);
                this->update_peer_cells<LiteralType::NumRowCols>(init_state, pos, num);

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

                this->fill_num(init_state, row, col, num);
                this->update_peer_cells<LiteralType::NumColRows>(init_state, pos, num);

                break;
            }

            default:
                assert(false);
                break;
        }
    }

    bool check_and_do_single_literal(InitState & init_state, Board & board, LiteralInfo literalInfo) {
        size_t pos, row, col, num;

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

                    this->fill_num(init_state, row, col, num);
                    this->update_peer_cells<LiteralType::NumRowCols>(init_state, pos, num);
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

                    this->fill_num(init_state, row, col, num);
                    this->update_peer_cells<LiteralType::NumColRows>(init_state, pos, num);
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
        size_t pos, row, col, num;

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
                    this->fill_num(init_state, row, col, num);
                    this->update_peer_cells<LiteralType::NumRowCols>(init_state, pos, num);

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
                    this->fill_num(init_state, row, col, num);
                    this->update_peer_cells<LiteralType::NumColRows>(init_state, pos, num);

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
    bool search(Board & board, size_t empties, const LiteralInfoEx & literalInfo) {
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
        assert(empties >= Sudoku::kMinInitCandidates);
#if 1
        LiteralInfo literalInfo = this->find_single_literal();

        while (literalInfo.isValid()) {
            bool is_legal = this->check_and_do_single_literal(this->init_state_, board, literalInfo);
            if (is_legal) {
                literalInfo = this->find_single_literal();
                empties--;
                if (empties == 0)
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

} // namespace JCZ
} // namespace gzSudoku

#endif // GZ_SUDOKU_JCZSOLVE_EX_H
