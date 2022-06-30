
#ifndef GZ_SUDOKU_H
#define GZ_SUDOKU_H

#if defined(_MSC_VER) && (_MSC_VER > 1200)
#pragma once
#endif

#if defined(GZ_SUDOKU)

#ifdef __cplusplus
extern "C" {
#endif

int GzSudoku(const char * input, char * output, int limit);

#ifdef __cplusplus
}
#endif

#endif // GZ_SUDOKU

#endif // GZ_SUDOKU_H
