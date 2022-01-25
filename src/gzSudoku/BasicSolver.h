
#ifndef GZ_SUDOKU_BASIC_SOLVER_H
#define GZ_SUDOKU_BASIC_SOLVER_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>

#include <cstdint>
#include <cstddef>
#include <vector>

#include "Sudoku.h"

namespace gzSudoku {

class BasicSolver {
public:
    typedef BasicSolver this_type;

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

    static size_t num_guesses;
    static size_t num_unique_candidate;
    static size_t num_failed_return;

protected:
    size_t              empties_;
    std::vector<Board>  answers_;

public:
    BasicSolver() : empties_(0) {
        init_statistics();
    }
    ~BasicSolver() {}

    static size_t get_num_guesses() { return this_type::num_guesses; }
    static size_t get_num_unique_candidate() { return this_type::num_unique_candidate; }
    static size_t get_num_failed_return() { return this_type::num_failed_return; }

    static size_t get_total_search_counter() {
        return (this_type::num_guesses + this_type::num_unique_candidate + this_type::num_failed_return);
    }

    static double get_guess_percent() {
        return calc_percent(this_type::num_guesses, this_type::get_total_search_counter());
    }

    static double get_failed_return_percent() {
        return calc_percent(this_type::num_failed_return, this_type::get_total_search_counter());
    }

    static double get_unique_candidate_percent() {
        return calc_percent(this_type::num_unique_candidate, this_type::get_total_search_counter());
    }

private:
    void init_statistics() {
        num_guesses = 0;
        num_unique_candidate = 0;
        num_failed_return = 0;
    }

public:
    size_t calc_empties(const Board & board) {
        size_t empties = 0;
        for (size_t pos = 0; pos < BoardSize; pos++) {
            unsigned char val = board.cells[pos];
            if (val == '.') {
                empties++;
            }
        }
        return empties;
    }

    void display_board(Board & board) {
        Sudoku::display_board(board, true);
    }

    template <size_t nSearchMode = SearchMode::OneSolution>
    void display_result(Board & board, double elapsed_time,
                        bool print_answer = true,
                        bool print_all_answers = true) {
        if (print_answer) {
            if (nSearchMode == SearchMode::OneSolution)
                Sudoku::display_board(board);
            else
                Sudoku::display_board(this->answers_);
        }
        printf("elapsed time: %0.3f ms, recur_counter: %" PRIuPTR "\n\n"
               "num_guesses: %" PRIuPTR ", num_failed_return: %" PRIuPTR ", num_unique_candidate: %" PRIuPTR "\n"
               "guess %% = %0.1f %%, failed_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
                elapsed_time,
                this_type::get_total_search_counter(),
                this_type::get_num_guesses(),
                this_type::get_num_failed_return(),
                this_type::get_num_unique_candidate(),
                this_type::get_guess_percent(),
                this_type::get_failed_return_percent(),
                this_type::get_unique_candidate_percent());
    }
};

} // namespace gzSudoku

#endif // GZ_SUDOKU_BASIC_SOLVER_H
