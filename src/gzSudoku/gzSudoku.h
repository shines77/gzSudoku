
#ifndef GZUDOKU_H
#define GZUDOKU_H

#if defined(_MSC_VER) && (_MSC_VER > 1200)
#pragma once
#endif

#ifdef __cplusplus
extern "C" {
#endif

int GzSudoku(const char * input, char * output, int limit);

#ifdef __cplusplus
}
#endif

#endif // GZUDOKU_H
