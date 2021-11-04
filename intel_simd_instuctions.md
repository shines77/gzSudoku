# Intel SIMD Instuction set

## 1. SSE#, AVX# 指令

### 1.1 `_mm_cvtepu8_epi16`

**Synopsis**

```cpp
__m128i _mm_cvtepu8_epi16(__m128i a);

#include <smmintrin.h>
Instruction: pmovzxbw xmm, xmm
CPUID Flags: SSE4.1
```

**Description**

```text
Zero extend packed unsigned 8-bit integers in a to packed 16-bit integers, and store the results in dst.
```

**描述**

```text
将 a 中的无符号 8 位整数补零扩展为 16 位有符号整数，并将结果存储在 dst 中。
```

**Operation**

```pascal
FOR j := 0 to 7
    i := j * 8
    k := j * 16
    dst[k+15:k] := ZeroExtend16(a[i+7:i])
ENDFOR
```

**Performance**

|Architecture|Latency|Throughput (CPI)|
|------------|:-----:|:--------------:|
|Skylake     |   1   |        1       |

### 1.2 `_mm256_extractf128_si256`

**Synopsis**

```cpp
__m128i _mm256_extractf128_si256(__m256i a, const int imm8);

#include <immintrin.h>
Instruction: vextractf128 xmm, ymm, imm8
CPUID Flags: AVX
```

**Description**

```text
Extract 128 bits (composed of integer data) from a, selected with imm8, and store the result in dst.
```

**描述**

```text
从使用 imm8 选择的数据中提取 128 位（由整数数据组成），并将结果存储在 dst 的低 128 位中，高 128 位为 0。
```

**Operation**

```pascal
CASE imm8[0] OF
    0: dst[127:0] := a[127:0]
    1: dst[127:0] := a[255:128]
ESAC
dst[MAX:128] := 0
```

**Performance**

|Architecture|Latency|Throughput (CPI)|
|------------|:-----:|:--------------:|
|Icelake     |   3   |       1        |
|Skylake     |   3   |       1        |
|Broadwell   |   3   |       1        |
|Haswell     |   3   |       1        |
|Ivy Bridge  |   3   |       1        |
