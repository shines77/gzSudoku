/*
	JCZSolve - Sudoku Solver by JasonLion based on champagne's version of the zhouyundong_2012's solver

	Based on code posted to:
		<http://forum.enjoysudoku.com/3-77us-solver-2-8g-cpu-testcase-17sodoku-t30470.html>
	by user zhouyundong_2012.
	Key insights include storing bands by digit and the Update routine
	The copyright is not specified.

	Subsequently updated by user champagne for 128 bit registers and speed optimization.
	Key insights include the pairing of ApplySingleOrEmptyCells and GuessBiValueInCell
	The copyright is not specified.

	Converted back to 32 bit data with further speed optimizations by JasonLion.
	The copyright is not specified.
*/

#ifdef __cplusplus
extern "C" {
#endif

extern int JCZSolver(const char *puzzle,char *solution,int limit);

#ifdef __cplusplus
	}
#endif

// END OF JCZSolve.h
