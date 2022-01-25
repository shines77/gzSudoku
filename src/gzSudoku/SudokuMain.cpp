
#if defined(_MSC_VER)
#define __MMX__
#define __SSE__
#define __SSE2__
#define __SSE3__
#define __SSSE3__
#define __SSE4A__
#define __SSE4a__
#define __SSE4_1__
#define __SSE4_2__
#define __POPCNT__
#define __LZCNT__
#define __AVX__
#define __AVX2__
#define __3dNOW__
#else
//#undef __SSE4_1__
//#undef __SSE4_2__
#endif

//#undef __SSE4_1__
//#undef __SSE4_2__

#if defined(_MSC_VER) && !defined(NDEBUG)
#include <vld.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cstring>      // For std::memset()
#include <vector>
#include <bitset>

#include "Sudoku.h"
#include "TestCase.h"

#include "BasicSolver.hpp"
#include "SudokuSolver_v4.h"
#include "SudokuSolver_v4a.h"
#include "SudokuSolver_v4b.h"
#include "SudokuSolver_v5.h"
#include "JCZSolve_v1.h"
#include "JCZSolve_v2.h"

#include "CPUWarmUp.h"
#include "StopWatch.h"

using namespace gzSudoku;

static const size_t kEnableDlxV1Solution = 1;
static const size_t kEnableDlxV2Solution = 1;
static const size_t kEnableDlxV3Solution = 1;

static const size_t kEnableV1Solution = 1;
static const size_t kEnableV2Solution = 1;
static const size_t kEnableV3Solution = 1;
static const size_t kEnableV4Solution = 1;

static std::vector<Board> bm_puzzles;
static size_t bm_puzzleTotal = 0;

// Index: [0 - 4]
#define TEST_CASE_INDEX         4

void make_sudoku_board(Board & board, size_t index)
{
    for (size_t row = 0; row < Sudoku::kRows; row++) {
        size_t row_base = row * 9;
        size_t col = 0;
        const char * prows = test_case[index].rows[row];
        char val;
        while ((val = *prows) != '\0') {
            if (val >= '0' && val <= '9') {
                if (val != '0')
                    board.cells[row_base + col] = val;
                else
                    board.cells[row_base + col] = '.';
                col++;
                assert(col <= Sudoku::kCols);
            }
            else if (val == '.') {
                board.cells[row_base + col] = '.';
                col++;
                assert(col <= Sudoku::kCols);
            }
            prows++;
        }
        assert(col == Sudoku::kCols);
    }
}

size_t read_sudoku_board(Board & board, char line[256])
{
    char * pline = line;
    // Skip the white spaces
    while (*pline == ' ' || *pline == '\t') {
        pline++;
    }
    // Is a comment ?
    if ((*pline == '#') || ((*pline == '/') && (pline[1] = '/'))) {
        return 0;
    }

    size_t pos = 0;
    char val;
    while ((val = *pline++) != '\0') {
        if (val >= '0' && val <= '9') {
            if (val != '0')
                board.cells[pos] = val;
            else
                board.cells[pos] = '.';
            pos++;
            assert(pos <= Sudoku::kBoardSize);
        }
        else if ((val == '.') || (val == ' ') || (val == '-')) {
            board.cells[pos] = '.';
            pos++;
            assert(pos <= Sudoku::kBoardSize);
        }
    }
    assert(pos <= Sudoku::kBoardSize);
    return pos;
}

size_t load_sudoku_puzzles(const char * filename, std::vector<Board> & puzzles)
{
    size_t puzzleCount = 0;

    std::ifstream ifs;
    try {
        ifs.open(filename, std::ios::in);
        if (ifs.good()) {
            ifs.seekg(0, std::ios::end);
            std::fstream::pos_type total_size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);

            //std::cout << std::endl;
            std::cout << "File name: " << filename << std::endl;
            std::cout << "File size: " << total_size << " Byte(s)" << std::endl;

            size_t predictedSize = total_size / (Sudoku::kBoardSize + 1) + 200;
            puzzles.resize(predictedSize);

            std::cout << "Predicted Size: " << predictedSize << std::endl << std::endl;

            while (!ifs.eof()) {
                char line[256];
                std::memset(line, 0, 16);
                ifs.getline(line, sizeof(line) - 1);

                Board board;
                size_t num_grids = read_sudoku_board(board, line);
                // Sudoku::BoardSize = 81
                if (num_grids >= Sudoku::kBoardSize) {
                    if (puzzleCount < predictedSize)
                        puzzles[puzzleCount] = board;
                    else
                        puzzles.push_back(board);
                    puzzleCount++;
                }
            }

            ifs.close();
        }
    }
    catch (std::exception & ex) {
        std::cout << "Exception info: " << ex.what() << std::endl << std::endl;
    }

    return puzzleCount;
}

template <typename SudokuSlover>
void run_solver_testcase(size_t index)
{
    Board board, solution;
    make_sudoku_board(board, index);

    SudokuSlover solver;
    solver.display_board(board);

    jtest::StopWatch sw;
    sw.start();
    int success = solver.solve(board, solution, 1);
    sw.stop();

    double elapsed_time = sw.getElapsedMillisec();
    solver.display_result(solution, elapsed_time);
}

void run_a_testcase(size_t index)
{
    if (kEnableV4Solution)
    {
        printf("------------------------------------------\n\n");
        printf("gzSudoku: v4::Solution - dfs\n\n");

        run_solver_testcase<v4::Solver>(index);

        printf("------------------------------------------\n\n");
        printf("gzSudoku: v4a::Solution - dfs\n\n");

        run_solver_testcase<v4a::Solver>(index);

        printf("------------------------------------------\n\n");
        printf("gzSudoku: v4b::Solution - dfs\n\n");

        run_solver_testcase<v4b::Solver>(index);
    }

    printf("------------------------------------------\n\n");
}

template <typename SudokuSolver>
void run_sudoku_test(std::vector<Board> & puzzles, size_t puzzleTotal, const char * name)
{
    typedef typename SudokuSolver::basic_solver_t   BasicSolverTy;

    //printf("------------------------------------------\n\n");
    printf("gzSudoku: %s::Solver\n\n", name);

    size_t total_guesses = 0;
    size_t total_unique_candidate = 0;
    size_t total_failed_return = 0;
    size_t total_no_guess = 0;

    size_t puzzleCount = 0;
    size_t puzzleInvalid = 0;
    size_t puzzleSolved = 0;
    size_t puzzleMultiSolution = 0;
    double total_time = 0.0;

    Board solution;
    SudokuSolver solver;
    BasicSolver basicSolver;
    jtest::StopWatch sw;
    sw.start();

    for (size_t i = 0; i < puzzleTotal; i++) {
        Board & board = puzzles[i];
        int solutions = solver.solve(board, solution, 1);
        if (solutions == 1) {
            total_guesses += BasicSolverTy::num_guesses;
            total_unique_candidate += BasicSolverTy::num_unique_candidate;
            total_failed_return += BasicSolverTy::num_failed_return;

            if (BasicSolverTy::num_guesses == 0) {
                total_no_guess++;
            }

            puzzleSolved++;
        }
        else if (solutions > 1) {
            puzzleMultiSolution++;
        }
        else {
            puzzleInvalid++;
        }
        puzzleCount++;
#ifndef NDEBUG
        if (puzzleCount > 100000)
            break;
#endif
    }

    sw.stop();
    total_time = sw.getElapsedMillisec();

    size_t total_recur_counter = total_guesses + total_unique_candidate + total_failed_return;
    double unique_candidate_percent = calc_percent(total_unique_candidate, total_recur_counter);
    double failed_return_percent = calc_percent(total_failed_return, total_recur_counter);
    double guesses_percent = calc_percent(total_guesses, total_recur_counter);
    double no_guess_percent = calc_percent(total_no_guess, puzzleCount);

    printf("Total puzzle count = %u, puzzle solved = %u, puzzle invalid = %u, puzzle multi-solution = %u\n"
           "total_no_guess: %" PRIuPTR ", no_guess %% = %0.1f %%\n\n",
           (uint32_t)puzzleCount, (uint32_t)puzzleSolved, (uint32_t)puzzleInvalid, (uint32_t)puzzleMultiSolution,
           total_no_guess, no_guess_percent);
    printf("Total elapsed time: %0.3f ms\n\n", total_time);
    printf("recur_counter: %" PRIuPTR "\n\n"
           "total_guesses: %" PRIuPTR ", total_failed_return: %" PRIuPTR ", total_unique_candidate: %" PRIuPTR "\n\n"
           "guess %% = %0.1f %%, failed_return %% = %0.1f %%, unique_candidate %% = %0.1f %%\n\n",
           total_recur_counter,
           total_guesses, total_failed_return, total_unique_candidate,
           guesses_percent, failed_return_percent, unique_candidate_percent);

    if (puzzleCount != 0) {
        printf("%0.1f usec/puzzle, %0.2f guesses/puzzle, %0.1f puzzles/sec\n\n",
               total_time * 1000.0 / puzzleCount,
               (double)total_guesses / puzzleCount,
               puzzleCount / (total_time / 1000.0));
    }
    else {
        printf("NaN usec/puzzle, NaN guesses/puzzle, %0.1f puzzles/sec\n\n",
               puzzleCount / (total_time / 1000.0));
    }

    printf("------------------------------------------\n\n");
}

void run_all_benchmark(const char * filename)
{
    // Read the puzzles data
    bm_puzzleTotal = load_sudoku_puzzles(filename, bm_puzzles);

#if defined(NDEBUG)
//  run_sudoku_test<v4::Solver     >(bm_puzzles, bm_puzzleTotal, "dfs::v4");
    run_sudoku_test<v4a::Solver    >(bm_puzzles, bm_puzzleTotal, "dfs::v4a");
    run_sudoku_test<v4b::Solver    >(bm_puzzles, bm_puzzleTotal, "dfs::v4b");
    run_sudoku_test<v5::Solver     >(bm_puzzles, bm_puzzleTotal, "dfs::v5");
    run_sudoku_test<JCZ::v1::Solver>(bm_puzzles, bm_puzzleTotal, "JCZ::v1");
    run_sudoku_test<JCZ::v2::Solver>(bm_puzzles, bm_puzzleTotal, "JCZ::v2");
#else
//  run_sudoku_test<v4b::Solver    >(bm_puzzles, bm_puzzleTotal, "dfs::v4b");
//  run_sudoku_test<JCZ::v1::Solver>(bm_puzzles, bm_puzzleTotal, "JCZ::v1");
    run_sudoku_test<JCZ::v2::Solver>(bm_puzzles, bm_puzzleTotal, "JCZ::v2");
#endif
}

int main(int argc, char * argv [])
{
    const char * filename = nullptr;
    const char * out_file = nullptr;
    if (argc > 2) {
        filename = argv[1];
        out_file = argv[2];
    }
    else if (argc > 1) {
        filename = argv[1];
    }

    jtest::CPU::warmup(1000);

    Sudoku::initialize();

    if (1)
    {
        if (filename == nullptr) {
            run_a_testcase(TEST_CASE_INDEX);
        }
    }

    if (1)
    {
        if (filename != nullptr) {
            run_all_benchmark(filename);
        }
    }

    Sudoku::finalize();

#if !defined(NDEBUG) && defined(_MSC_VER)
    ::system("pause");
#endif

    return 0;
}
