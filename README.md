# gzSudoku

## 简介

本程序基于 [JCZSolve](./others/jczslover/JCZSolve.c) 改写，比同类型里最快的数独求解程序 [rust-sudoku](https://github.com/Emerentius/sudoku) 还要快 3-5 %左右。JCZSolve 在求解较为简单的数独时，速度较为领先。

特色：

`BitVec.h`: 一个跨平台，支持 SSE, SSE2, AVX, AVX2 和 AVX512 等 SIMD 指令的向量库。[点这里查看](./src/gudoku/BitVec.h)

包含以下类：

- BitVec08x16
- BitVec16x16_SSE
- BitVec16x16_AVX

对 SSE, SSE2, AVX, AVX2 和 AVX512 指令的常用操作做了比较完整的封装，主要服务于 `数独` 求解程序，但也尽量做了扩展。

## 关于 JCZSolve

JCZSolve 是由一个老外 [JasonLion](http://forum.enjoysudoku.com/member2711.html) 基于中国人 [zhouyundong_2012](http://forum.enjoysudoku.com/member3701.html) 写的代码整理而来的数独求解程序，原代码发表在一个国外的数独论坛 [forum.enjoysudoku.com](http://forum.enjoysudoku.com/) 上，帖子在这里：[3.77us Solver(2.8G CPU, TestCase:17Sodoku)](http://forum.enjoysudoku.com/3-77us-solver-2-8g-cpu-testcase-17sodoku-t30470.html)。原代码有些乱码，国外网友整理了，放在这里下载：[JCZSolve 源码压缩包](http://forum.enjoysudoku.com/download/file.php?id=436&sid=2906ffe2e2c5cf10c4d004f184eafe10) 。如果下载不了，本仓库的 `./others/jczslover` 目录下面也有：[JCZSolve.c](./others/jczslover/JCZSolve.c)、[JCZSolve.h](./others/jczslover/JCZSolve.h)、[jczsolve.zip](./others/jczslover/jczsolve.zip) 。

## 关于 rust-sudoku

是一个用 Rust 语言写的数独求解程序，基于 JCZSolve 改写的，我曾经研究了很久，不知道它为什么比 JCZSolve 快。后来，我用 C++ 照着它重写了一遍，才找到关键的地方。其实他也没做什么修改，就是去掉了一个 if 判断，就稍微快了一点。

我后来在它的基础上又做了一些改进，稍微比它快一点点。JCZSolve 这个方法很多地方看起来并不完美，但是它确实是在求解较为简单的数独时，速度是比较快的。而 [tdoku](https://github.com/t-dillon/tdoku) 则在求解较为困难的数独是更快的，也是综合速度最快的数独求解程序。

Git: [https://github.com/Emerentius/sudoku](https://github.com/Emerentius/sudoku)

## 测试集

数独的测试集放在 `./data` 目录下：

```bash
# 这两个测试集都是所有剩17个格子数独的集合，属于是比较简单的数独
# 这两个文件基本相同，但稍微有一点差异，只测试其中一个即可（建议）
./data/puzzles2_17_clue
./data/puzzles_17_clue_49151

# 这个是最难的 1905 个数独的测试集，属于是比较困难的数独
./data/puzzles5_forum_hardest_1905_11+
```

## 编译与测试

### 1. 在 Linux 下：

请先选择或安装 gcc 和 clang 的不同版本。

- **gcc**

编译：

```bash
cmake .
make
```

运行测试：

```bash
./gzSudoku ./data/puzzles2_17_clue
./gzSudoku ./data/puzzles_17_clue_49151
./gzSudoku ./data/puzzles5_forum_hardest_1905_11+
```

- **clang**

请先切换到 ./clang 目录下，再用 CMake 来构建 makefile 和编译。

```bash
cd ./clang
cmake -G "Unix Makefiles" -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ ./
make
```

运行测试：

```bash
./gzSudoku ../data/puzzles2_17_clue
./gzSudoku ../data/puzzles_17_clue_49151
./gzSudoku ../data/puzzles5_forum_hardest_1905_11+
```

### 2. 在 Windows 下

请切换到 `.\bin\vc2015\x64-Release` 目录下，执行下面命令。

运行测试：

```bash
.\gzSudoku.exe ..\..\..\data\puzzles2_17_clue
.\gzSudoku.exe ..\..\..\data\puzzles_17_clue_49151
.\gzSudoku.exe ..\..\..\data\puzzles5_forum_hardest_1905_11+
```

## LeetCode

在 LeetCode 上，也有一道关于数独的题目：[LeetCode problem 37: SudokuSolver](https://leetcode.cn/problems/sudoku-solver/) 。
