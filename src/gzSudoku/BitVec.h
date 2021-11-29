
#ifndef GZ_SUDOKU_BITVEC_H
#define GZ_SUDOKU_BITVEC_H

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <memory.h>
#include <assert.h>

#include <cstdint>
#include <cstddef>
#include <bitset>
#include <cstring>          // For std::memset()
#include <initializer_list>
#include <type_traits>

#include "BitUtils.h"

#define __MMX__
#define __SSE__
#define __SSE2__
#define __SSE3__
#define __SSSE3__
#define __SSE4A__
#define __SSE4a__
#define __SSE4_1__
#define __SSE4_2__

#define __AVX__
#define __AVX2__

//#undef __SSE4_1__
//#undef __AVX2__

#undef __AVX512VL__
#undef __AVX512F__

// For SSE2, SSE3, SSSE3, SSE 4.1, AVX, AVX2
#if defined(_MSC_VER)
#include <intrin.h>
#include "msvc_x86intrin.h"
#include <immintrin.h>
#else
#include <x86intrin.h>
#include "msvc_x86intrin.h"
#include <avxintrin.h>

#define _mm_cvtss_i32 _mm_cvtss_si32
#define _mm_cvtsd_i32 _mm_cvtsd_si32
#define _mm_cvti32_sd _mm_cvtsi32_sd
#define _mm_cvti32_ss _mm_cvtsi32_ss
#ifdef __x86_64__
#define _mm_cvtss_i64 _mm_cvtss_si64
#define _mm_cvtsd_i64 _mm_cvtsd_si64
#define _mm_cvti64_sd _mm_cvtsi64_sd
#define _mm_cvti64_ss _mm_cvtsi64_ss
#endif

#define _mm_setr_epi64x(high, low) \
        _mm_setr_epi64(_mm_cvtsi64_m64(high), _mm_cvtsi64_m64(low))

#if !defined(_mm256_set_m128i)

#define _mm256_set_m128(hi, lo) \
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)

#define _mm256_set_m128d(hi, lo) \
        _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), (hi), 0x1)

#define _mm256_set_m128i(hi, lo) \
        _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)

#endif // !_mm256_set_m128i

#if !defined(_mm256_setr_m128i)

#define _mm256_setr_m128(lo, hi)    _mm256_set_m128((hi), (lo))
#define _mm256_setr_m128d(lo, hi)   _mm256_set_m128d((hi), (lo))
#define _mm256_setr_m128i(lo, hi)   _mm256_set_m128i((hi), (lo))

#endif // !_mm256_setr_m128i

#define _mm256_test_all_zeros(mask, val) \
        _mm256_testz_si256((mask), (val))

#define _mm256_test_all_ones(val) \
        _mm256_testc_si256((val), _mm256_cmpeq_epi32((val), (val)))

#define _mm256_test_mix_ones_zeros(mask, val) \
        _mm256_testnzc_si256((mask), (val))
#endif // _MSC_VER

//
// Missing in MSVC (before 2017) & gcc (before 11.0)
//
#if !defined(_mm256_cvtsi256_si32)
#define _mm256_cvtsi256_si32(val) \
        _mm_cvtsi128_si32(_mm256_castsi256_si128(val))
#endif // _mm256_cvtsi256_si32

#if !defined(_mm256_insert_epi16)
#define _mm256_insert_epi16(target, value, index) \
        AVX2::template mm256_insert_epi16<index>(target, value)
#endif // _mm256_insert_epi16

//
// Intel Intrinsics Guide (SIMD)
//
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.htm
//

//
// __m128i _mm_alignr_epi8(__m128i a, __m128i b, int imm8);
//
// Concatenate 16-byte blocks in a and b into a 32-byte temporary result,
// shift the result right by imm8 bytes, and store the low 16 bytes in dst.
//
//   tmp[255:0] := ((a[127:0] << 128)[255:0] OR b[127:0]) >> (imm8*8)
//   dst[127:0] := tmp[127:0]
//

namespace gzSudoku {

#pragma pack(push, 1)

union IntVec128 {
    alignas(16) int8_t   i8[16];
    alignas(16) uint8_t  u8[16];
    alignas(16) int16_t  i16[8];
    alignas(16) uint16_t u16[8];
    alignas(16) int32_t  i32[4];
    alignas(16) uint32_t u32[4];
    alignas(16) int64_t  i64[2];
    alignas(16) uint64_t u64[2];
};

union IntVec256 {
    alignas(32) int8_t   i8[32];
    alignas(32) uint8_t  u8[32];
    alignas(32) int16_t  i16[16];
    alignas(32) uint16_t u16[16];
    alignas(32) int32_t  i32[8];
    alignas(32) uint32_t u32[8];
    alignas(32) int64_t  i64[4];
    alignas(32) uint64_t u64[4];
};

#pragma pack(pop)

struct SSE {

static inline
uint32_t mm_cvtsi128_si32_low(__m128i m128)
{
    uint32_t low32 = _mm_cvtsi128_si32(m128);
    return (low32 & 0xFFFFUL);
}

static inline
uint32_t mm_cvtsi128_si32_high(__m128i m128)
{
    uint32_t low32 = _mm_cvtsi128_si32(m128);
    return (low32 >> 16U);
}

}; // SSE Wrapper

struct AVX {

static inline
uint32_t mm256_cvtsi256_si32_low(__m256i m256)
{
    return (uint32_t)(_mm256_cvtsi256_si32(m256) & 0xFFFFUL);
}

static inline
uint32_t mm256_cvtsi256_si32_high(__m256i m256)
{
    return (uint32_t)(_mm256_cvtsi256_si32(m256) >> 16U);
}

}; // AVX Wrapper

struct AVX2 {

//
// See: https://agner.org/optimize/
//
// See: https://stackoverflow.com/questions/54048226/move-an-int64-t-to-the-high-quadwords-of-an-avx2-m256i-vector
// See: https://stackoverflow.com/questions/58303958/how-to-implement-16-and-32-bit-integer-insert-and-extract-operations-with-avx-51
//
template <int index>
static inline
__m256i mm256_insert_epi16(__m256i target, int value)
{
    __m256i result;
    switch (index) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        {
            __m128i result128 = _mm_insert_epi16(_mm256_castsi256_si128(target), value, (index < 8) ? index : 0);
            result = _mm256_castsi128_si256(result128);
            break;
        }

        case 8:
        {
#if 1
            __m128i high128 = _mm256_extracti128_si256(target, 1);
            __m128i value128 = _mm_cvtsi32_si128(value);
            __m128i mixed128 = _mm_blend_epi16(high128, value128, 0x00000001);
            result = _mm256_inserti128_si256(target, mixed128, 1);
#else
            __m128i result128 = _mm_insert_epi16(_mm256_castsi256_si128(target), value, 0);
            __m256i result256 = _mm256_inserti128_si256(_mm256_castsi128_si256(result128), result128, 1);
            result = _mm256_blend_epi16(target, result256, 0b00000001);
#endif
            break;
        }

        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        {
            __m128i high128 = _mm256_extracti128_si256(target, 1);
            __m128i mixed128 = _mm_insert_epi16(high128, value, (index >= 8) ? (index - 8) : 0);
            result = _mm256_inserti128_si256(target, mixed128, 1);
            break;
        }

        default:
            static_assert((index < 16), "AVX2::_mm256_insert_epi16(target, value, index), index is out of range [0, 15].");
            break;
    }
    return result;
}

}; // AVX2 Wrapper

struct AVX512 {

#if !(defined(__AVX512__) && defined(__AVX512_FP16__))

static inline uint32_t mm_cvtsi128_si16(__m128i m128)
{
    uint32_t low32 = _mm_cvtsi128_si32(m128);   // SSE2
    return (low32 & 0xFFFFUL);
}

#endif

}; // AVX512 Wrapper

#if defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4A__) || defined(__SSE4a__) \
 || defined(__SSE4_1__) || defined(__SSE4_2__)

struct BitVec08x16 {
    __m128i m128;

    BitVec08x16() noexcept {}
    // non-explicit conversions intended
    BitVec08x16(const __m128i & m128i) noexcept : m128(m128i) {}
    BitVec08x16(const BitVec08x16 & src) noexcept : m128(src.m128) {}

    BitVec08x16(uint8_t c00, uint8_t c01, uint8_t c02, uint8_t c03,
                uint8_t c04, uint8_t c05, uint8_t c06, uint8_t c07,
                uint8_t c08, uint8_t c09, uint8_t c10, uint8_t c11,
                uint8_t c12, uint8_t c13, uint8_t c14, uint8_t c15) :
            m128(_mm_setr_epi8(c00, c01, c02, c03, c04, c05, c06, c07,
                               c08, c09, c10, c11, c12, c13, c14, c15)) {}

    BitVec08x16(uint16_t w00, uint16_t w01, uint16_t w02, uint16_t w03,
                uint16_t w04, uint16_t w05, uint16_t w06, uint16_t w07) :
            m128(_mm_setr_epi16(w00, w01, w02, w03, w04, w05, w06, w07)) {}

    BitVec08x16(uint32_t i00, uint32_t i01, uint32_t i02, uint32_t i03) :
            m128(_mm_setr_epi32(i00, i01, i02, i03)) {}

    BitVec08x16(uint64_t q00, uint64_t q01) :
            m128(_mm_setr_epi64x(q00, q01)) {}

    inline BitVec08x16 & operator = (const BitVec08x16 & right) {
        this->m128 = right.m128;
        return *this;
    }

    inline BitVec08x16 & operator = (const __m128i & right) {
        this->m128 = right;
        return *this;
    }

    inline void loadAligned(const void * mem_addr) {
        this->m128 = _mm_load_si128((const __m128i *)mem_addr);
    }

    inline void loadUnaligned(const void * mem_addr) {
        this->m128 = _mm_loadu_si128((const __m128i *)mem_addr);
    }

    inline void saveAligned(void * mem_addr) const {
        _mm_store_si128((__m128i *)mem_addr, this->m128);
    }

    inline void saveUnaligned(void * mem_addr) const {
        _mm_storeu_si128((__m128i *)mem_addr, this->m128);
    }

    static inline void copyAligned(const void * src_mem_addr, void * dest_mem_addr) {
        __m128i tmp = _mm_load_si128((const __m128i *)src_mem_addr);
        _mm_store_si128((__m128i *)dest_mem_addr, tmp);
    }

    static void copyUnaligned(const void * src_mem_addr, void * dest_mem_addr) {
        __m128i tmp = _mm_loadu_si128((const __m128i *)src_mem_addr);
        _mm_storeu_si128((__m128i *)dest_mem_addr, tmp);
    }

    inline bool operator == (const BitVec08x16 & other) const {
        return this->isEqual(other);
    }

    inline bool operator != (const BitVec08x16 & other) const {
        return this->isNotEqual(other);
    }

    // Logical operation
    inline BitVec08x16 operator & (const BitVec08x16 & vec) {
        BitVec08x16 tmp(this->m128);
        tmp._and(vec);
        return tmp;
    }

    inline BitVec08x16 operator | (const BitVec08x16 & vec) {
        BitVec08x16 tmp(this->m128);
        tmp._or(vec);
        return tmp;
    }

    inline BitVec08x16 operator ^ (const BitVec08x16 & vec) {
        BitVec08x16 tmp(this->m128);
        tmp._xor(vec);
        return tmp;
    }

    inline BitVec08x16 operator ~ () {
        BitVec08x16 tmp(this->m128);
        tmp._not();
        return tmp;
    }

    inline BitVec08x16 operator ! () {
        BitVec08x16 tmp(this->m128);
        tmp._not();
        return tmp;
    }

    // Logical operation
    inline BitVec08x16 & operator &= (const BitVec08x16 & vec) {
        this->_and(vec);
        return *this;
    }

    inline BitVec08x16 & operator |= (const BitVec08x16 & vec) {
        this->_or(vec);
        return *this;
    }

    inline BitVec08x16 & operator ^= (const BitVec08x16 & vec) {
        this->_xor(vec);
        return *this;
    }

    // Logical operation
    inline BitVec08x16 & _and(const BitVec08x16 & vec) {
        this->m128 = _mm_and_si128(this->m128, vec.m128);
        return *this;
    }

    inline BitVec08x16 & and_not(const BitVec08x16 & vec) {
        this->m128 = _mm_andnot_si128(vec.m128, this->m128);
        return *this;
    }

    inline BitVec08x16 & _or(const BitVec08x16 & vec) {
        this->m128 = _mm_or_si128(this->m128, vec.m128);
        return *this;
    }

    inline BitVec08x16 & _xor(const BitVec08x16 & vec) {
        this->m128 = _mm_xor_si128(this->m128, vec.m128);
        return *this;
    }

    // Logical operation
    inline BitVec08x16 & _and(__m128i value) {
        this->m128 = _mm_and_si128(this->m128, value);
        return *this;
    }

    inline BitVec08x16 & and_not(__m128i value) {
        this->m128 = _mm_andnot_si128(value, this->m128);
        return *this;
    }

    inline BitVec08x16 & _or(__m128i value) {
        this->m128 = _mm_or_si128(this->m128, value);
        return *this;
    }

    inline BitVec08x16 & _xor(__m128i value) {
        this->m128 = _mm_xor_si128(this->m128, value);
        return *this;
    }

    // Logical not: !
    inline BitVec08x16 & _not() {
        BitVec08x16 ones;
        ones.setAllOnes();
        this->m128 = _mm_andnot_si128(this->m128, ones.m128);
        return *this;
    }

    // fill
    inline void fill_u8(uint8_t value) {
        this->m128 = _mm_set1_epi8(value);       // SSE2
    }

    inline void fill_u16(uint16_t value) {
        this->m128 = _mm_set1_epi16(value);      // SSE2
    }

    inline void fill_u32(uint32_t value) {
        this->m128 = _mm_set1_epi32(value);      // SSE2
    }

    inline void fill_u64(uint64_t value) {
        this->m128 = _mm_set1_epi64x(value);     // SSE2
    }

    static bool isMemEqual(const void * mem_addr_1, const void * mem_addr_2) {
        const IntVec128 * pIntVec128_1 = (const IntVec128 *)mem_addr_1;
        const IntVec128 * pIntVec128_2 = (const IntVec128 *)mem_addr_2;
        bool isEqual = (pIntVec128_1->u64[0] == pIntVec128_2->u64[0]) && (pIntVec128_1->u64[1] == pIntVec128_2->u64[1]);
        return isEqual;
    }

    inline void setAllZeros() {
        // Equivalent to SSE2: _mm_setzero_si128()
        this->m128 = _mm_xor_si128(this->m128, this->m128);
    }

    inline void setAllOnes() {
        // Reference from _mm_test_all_ones() macro
        this->m128 = _mm_cmpeq_epi32(this->m128, this->m128);
    }

    inline bool isAllZeros() const {
#if defined(__SSE4_1__)
        return (_mm_test_all_zeros(this->m128, this->m128) == 1);
#else
        BitVec08x16 zeros;
        zeros.setAllZeros();
        BitVec08x16 compare_mask = this->whichIsEqual(zeros);
        return (_mm_movemask_epi8(compare_mask.m128) == 0xFFFFU);
#endif
    }

    inline bool isAllOnes() const {
#if defined(__SSE4_1__)
        return (_mm_test_all_ones(this->m128) == 1);
#else
        BitVec08x16 ones;
        ones.setAllOnes();
        BitVec08x16 compare_mask = this->whichIsEqual(ones);
        return (_mm_movemask_epi8(compare_mask.m128) == 0xFFFFU);
#endif
    }

    inline bool isNotAllZeros() const {
#if defined(__SSE4_1__)
        return (_mm_test_all_zeros(this->m128, this->m128) == 0);
#else
        BitVec08x16 zeros;
        zeros.setAllZeros();
        BitVec08x16 compare_mask = this->whichIsEqual(zeros);
        return (_mm_movemask_epi8(compare_mask.m128) != 0xFFFFU);
#endif
    }

    inline bool isNotAllOnes() const {
#if defined(__SSE4_1__)
        return (_mm_test_all_ones(this->m128) == 0);
#else
        BitVec08x16 ones;
        ones.setAllOnes();
        BitVec08x16 compare_mask = this->whichIsEqual(ones);
        return (_mm_movemask_epi8(compare_mask.m128) != 0xFFFFU);
#endif
    }

    // It is not all zeros and is not all ones
    inline bool isNotAllZerosAndAllOnes(const BitVec08x16 & other) const {
        return (_mm_test_mix_ones_zeros(this->m128, other.m128) == 0);
    }

    // It is mixed by zeros and ones
    inline bool isMixZerosAndOnes(const BitVec08x16 & other) const {
        return (_mm_test_mix_ones_zeros(this->m128, other.m128) == 1);
    }

    inline bool isEqual(const BitVec08x16 & other) const {
#if defined(__AVX512VL__) && defined(__AVX512BW__)
        return (_mm_cmp_epi16_mask(this->m128, other.m128, _MM_CMPINT_EQ) == 0x0F);
#elif 1
        // isAllZeros() faster than isAllOnes(), because it's instructions less than isAllOnes().
        BitVec08x16 is_neq_mask = _mm_xor_si128(this->m128, other.m128);
        return is_neq_mask.isAllZeros();
#else
        BitVec08x16 is_eq_mask = _mm_cmpeq_epi16(this->m128, other.m128);
        return is_eq_mask.isAllOnes();
#endif
    }

    inline bool isNotEqual(const BitVec08x16 & other) const {
#if 1
        // isNotAllZeros() faster than isNotAllOnes(), because it's instructions less than isNotAllOnes().
        BitVec08x16 is_neq_mask = _mm_xor_si128(this->m128, other.m128);
        return is_neq_mask.isNotAllZeros();
#else
        BitVec08x16 is_eq_mask = _mm_cmpeq_epi16(this->m128, other.m128);
        return is_eq_mask.isNotAllOnes();
#endif
    }

    inline bool hasAnyZero() const {
#if 1
        return this->isNotAllOnes();
#else
        BitVec08x16 which_is_zero = this->whichIsZeros();
        return (_mm_movemask_epi8(which_is_zero.m128) != 0);
#endif
    }

    inline bool hasAnyOne() const {
#if 1
        return this->isNotAllZeros();
#else
        BitVec08x16 which_is_non_zero = this->whichIsNonZero();
        return (_mm_movemask_epi8(which_is_non_zero.m128) != 0);
#endif
    }

    inline bool hasIntersects(const BitVec08x16 & other) const {
#if defined(__SSE4_1__)
        return (_mm_testz_si128(this->m128, other.m128) == 0);
#else
        BitVec08x16 intersects = _mm_and_si128(this->m128, other.m128);
        return intersects.isNotAllZeros();
#endif
    }

    inline bool isSubsetOf(const BitVec08x16 & other) const {
#if defined(__SSE4_1__)
        return (_mm_testc_si128(other.m128, this->m128) == 1);
#else
        BitVec08x16 subset = _mm_andnot_si128(other.m128, this->m128);
        return subset.isAllZeros();
#endif
    }

    inline BitVec08x16 whichIsEqual(const BitVec08x16 & other) const {
        return _mm_cmpeq_epi16(this->m128, other.m128);
    }

    inline BitVec08x16 whichIsNotEqual(const BitVec08x16 & other) const {
        __m128i is_eq_mask = _mm_cmpeq_epi16(this->m128, other.m128);
        BitVec08x16 ones;
        ones.setAllOnes();
        __m128i is_neq_mask = _mm_andnot_si128(is_eq_mask, ones.m128);
        return is_neq_mask;
    }

    inline BitVec08x16 whichIsMoreThan(const BitVec08x16 & other) const {
        return _mm_cmpgt_epi16(this->m128, other.m128);
    }

    inline BitVec08x16 whichIsLessThan(const BitVec08x16 & other) const {
        return _mm_cmplt_epi16(this->m128, other.m128);
    }

    inline BitVec08x16 whichIsZeros() const {
        BitVec08x16 zeros;
        zeros.setAllZeros();
        return this->whichIsEqual(zeros);
    }

    inline BitVec08x16 whichIsOnes() const {
        BitVec08x16 ones;
        ones.setAllOnes();
        return this->whichIsEqual(ones);
    }

    inline BitVec08x16 whichIsNonZero() const {
        BitVec08x16 zeros;
        zeros.setAllZeros();
        return this->whichIsNotEqual(zeros);
    }

    inline BitVec08x16 whichIsMoreThanZero() const {
        BitVec08x16 zeros;
        zeros.setAllZeros();
        return this->whichIsMoreThan(zeros);
    }

    inline BitVec08x16 whichIsLessThanZero() const {
        BitVec08x16 zeros;
        zeros.setAllZeros();
        return this->whichIsLessThan(zeros);
    }

    inline BitVec08x16 whichIsEqual16(uint16_t num) const {
        __m128i num_mask = _mm_set1_epi16((int16_t)num);
        return this->whichIsEqual(num_mask);
    }

    inline BitVec08x16 whichIsNotEqual16(uint16_t num) const {
        __m128i num_mask = _mm_set1_epi16((int16_t)num);
        return this->whichIsNotEqual(num_mask);
    }

    template <bool isNonZeros>
    inline int indexOfIsEqual16(uint32_t num) const {
        BitVec08x16 is_equal_mask = this->whichIsEqual16(num);
        return this->template firstIndexOfOnes16<isNonZeros>(is_equal_mask);
    }

    template <bool isNonZeros, bool isRepeat = true>
    inline int indexOfIsEqual16(const BitVec08x16 & in_num_mask) const {
        BitVec08x16 num_mask = in_num_mask;
        if (!isRepeat) {
#if defined(__AVX2__)
            num_mask = _mm_broadcastw_epi16(num_mask.m128);
#elif defined(__SSSE3__)
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shuffle_epi8(num_mask.m128, lookup_mask);
#else
            // SSE2
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shufflelo_epi16(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
            num_mask = _mm_shuffle_epi32(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
#endif
        }
        BitVec08x16 is_equal_mask = this->whichIsEqual(num_mask);
        return this->template firstIndexOfOnes16<isNonZeros>(is_equal_mask);
    }

    template <bool isRepeat = true>
    inline int maskOfIsEqual16(const BitVec08x16 & in_num_mask) const {
        BitVec08x16 num_mask = in_num_mask;
        if (!isRepeat) {
#if defined(__AVX2__)
            num_mask = _mm_broadcastw_epi16(num_mask.m128);
#elif defined(__SSSE3__)
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shuffle_epi8(num_mask.m128, lookup_mask);
#else
            // SSE2
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shufflelo_epi16(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
            num_mask = _mm_shuffle_epi32(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
#endif
        }
        BitVec08x16 is_equal_mask = this->whichIsEqual(num_mask);
        return this->maskOfOnes16(is_equal_mask);
    }

    template <bool isNonZeros>
    inline int firstIndexOfOnes16(const BitVec08x16 & compare_mask) const {
 #if !(defined(WIN64) || defined(_WIN64) || defined(_M_X64) || defined(_M_AMD64) \
    || defined(_M_IA64) || defined(__amd64__) || defined(__x86_64__)) || 1
        int compare_mask_16 = _mm_movemask_epi8(compare_mask.m128);
        if (isNonZeros || (compare_mask_16 != 0)) {
            assert(!isNonZeros || (compare_mask_16 != 0));
            uint32_t first_offset = BitUtils::bsf(compare_mask_16);
            int first_index = (int)(first_offset >> 1U);
            return first_index;
        }
        else return -1;
#else
        // Under x64 mode, we maybe can use this version,
        // but it's not sure faster than the below version.
        BitVec08x16 zeros;
        zeros.setAllZeros();
       __m128i mask64 = _mm_packus_epi16(compare_mask.m128, zeros.m128);
        uint64_t compare_mask_64 = (uint64_t)_mm_cvtsi128_si64(mask64);
        if (isNonZeros || (compare_mask_64 != 0)) {
            assert(!isNonZeros || (compare_mask_64 != 0));
            uint32_t first_offset = BitUtils::bsf(compare_mask_64);
            int first_index = (int)(first_offset >> 3U);
            return first_index;
        }
        else return -1;
#endif
    }

    inline uint32_t maskOfOnes16(const BitVec08x16 & compare_mask) const {
        uint32_t compare_mask_16 = (uint32_t)_mm_movemask_epi8(compare_mask.m128);
        return compare_mask_16;
    }

    inline uint16_t extract(int index) const {
        #define CASE(x) case x: return (uint16_t)_mm_extract_epi16(this->m128, x);
        switch (index) {
            CASE(0)
            CASE(1)
            CASE(2)
            CASE(3)
            CASE(4)
            CASE(5)
            CASE(6)
            CASE(7)
            default:
                assert(false);
                return uint16_t(-1);
        }
        #undef CASE
    }

    inline void insert(int index, uint16_t value) {
        #define CASE(x) case x: this->m128 = _mm_insert_epi16(this->m128, value, x); break;
        switch (index) {
            CASE(0)
            CASE(1)
            CASE(2)
            CASE(3)
            CASE(4)
            CASE(5)
            CASE(6)
            CASE(7)
            default:
                assert(false);
                break;
        }
        #undef CASE
    }

    template <size_t MaxLength, size_t MaxBits>
    BitVec08x16 popcount16() const {
#if (defined(__AVX512_BITALG__) && defined(__AVX512VL__)) || defined(__AVX512VPOPCNTW__)
        return _mm_popcnt_epi16(this->m128);
#elif defined(__SSSE3__)
        if (MaxBits <= 8) {
            // Note: Ensure that the highest 8 bits must be 0.
            __m128i lookup     = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m128i mask4      = _mm_set1_epi16(0x000F);
            __m128i sum_odd_4  = _mm_shuffle_epi8(lookup, _mm_and_si128(this->m128, mask4));
            __m128i sum_even_4 = _mm_shuffle_epi8(lookup, _mm_srli_epi16(this->m128, 4));
            __m128i sum_0_7    = _mm_add_epi16(sum_odd_4, sum_even_4);
            __m128i result     = sum_0_7;
            return result;
        }
        else if (MaxBits == 9) {
            // Note: Ensure that the highest 7 bits must be 0.
            __m128i lookup     = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m128i mask4      = _mm_set1_epi16(0x000F);
            __m128i sum_odd_4  = _mm_shuffle_epi8(lookup, _mm_and_si128(this->m128, mask4));
            __m128i sum_even_4 = _mm_shuffle_epi8(lookup, _mm_srli_epi16(this->m128, 4));
            __m128i sum_0_7    = _mm_add_epi16(sum_odd_4, sum_even_4);
            __m128i result     = _mm_add_epi16(sum_0_7, _mm_srli_epi16(this->m128, 8));
            return result;
        }
        else {
            __m128i lookup     = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m128i mask4      = _mm_set1_epi16(0x0F0F);
            __m128i sum_odd_4  = _mm_shuffle_epi8(lookup, _mm_and_si128(this->m128, mask4));
            __m128i even_bytes = _mm_srli_epi16(this->m128, 4);
            __m128i sum_even_4 = _mm_shuffle_epi8(lookup, _mm_and_si128(even_bytes, mask4));
            __m128i sum_00_15  = _mm_add_epi16(sum_odd_4, sum_even_4);
            __m128i mask8      = _mm_set1_epi16(0x00FF);
            __m128i sum_odd_8  = _mm_and_si128(sum_00_15, mask8);
            __m128i sum_even_8 = _mm_srli_epi16(sum_00_15, 8);
            __m128i result     = _mm_add_epi16(sum_odd_8, sum_even_8);
            return result;
        }
#else
        // SSE2 version from https://www.hackersdelight.org/hdcodetxt/pop.c.txt
        __m128i mask1 = _mm_set1_epi8(0x77);
        __m128i mask2 = _mm_set1_epi8(0x0F);
        __m128i mask3 = _mm_set1_epi16(0x00FF);
        __m128i x = this->m128;
        __m128i n = _mm_and_si128(mask1, _mm_srli_epi64(x, 1));
        x = _mm_sub_epi8(x, n);
        n = _mm_and_si128(mask1, _mm_srli_epi64(n, 1));
        x = _mm_sub_epi8(x, n);
        n = _mm_and_si128(mask1, _mm_srli_epi64(n, 1));
        x = _mm_sub_epi8(x, n);
        x = _mm_add_epi8(x, _mm_srli_epi16(x, 4));
        x = _mm_and_si128(mask2, x);
        // Tip: _mm_bsrli_si128() is missing on MSVC (Windows)
        x = _mm_add_epi16(_mm_and_si128(x, mask3),
                          _mm_and_si128(_mm_srli_si128(x, 1), mask3));
        return x;
#endif
    }

    template <size_t MaxLength>
    void min_u8(BitVec08x16 & minpos) const {
#if defined(__SSE4_1__)
        //
        // See: https://blog.csdn.net/weixin_34378767/article/details/86257834
        //
        if (MaxLength <= 8) {
            __m128i zero = _mm_setzero_si128();
            __m128i low64 = _mm_unpacklo_epi8(this->m128, zero);
            __m128i minpos_u16 = _mm_minpos_epu16(low64);
            //minpos = _mm_packus_epi16(minpos_u16, zero);
            minpos = minpos_u16;
        }
        else if (MaxLength == 9) {
            // Note: Ensure that the highest 7 byets must be maximum value.
            __m128i zero = _mm_setzero_si128();
            __m128i low64  = _mm_unpacklo_epi8(this->m128, zero);
            __m128i minpos_low = _mm_minpos_epu16(low64);
            __m128i high64 = _mm_unpackhi_epi8(this->m128, zero);
            high64 = _mm_cvtepu8_epi16(high64);
            __m128i high_9_index = _mm_setr_epi16(0, 9, 0, 0, 0, 0, 0, 0);
            high64 = _mm_or_si128(high64, high_9_index);
            __m128i min_num_all = _mm_min_epu16(minpos_low, high64);
            __m128i min_in_high_mask = _mm_cmpgt_epi16(min_num_all, high64);
            __m128i minpos_8_low = _mm_andnot_si128(minpos_low, min_in_high_mask);
            __m128i minpos_8_high = _mm_and_si128(high64, min_in_high_mask);
            __m128i minpos_u16 = _mm_and_si128(minpos_8_low, minpos_8_high);
            //minpos = _mm_packus_epi16(minpos_u16, zero);
            minpos = minpos_u16;
        }
        else {
            __m128i zero = _mm_setzero_si128();
            __m128i low64  = _mm_unpacklo_epi8(this->m128, zero);
            __m128i high64 = _mm_unpackhi_epi8(this->m128, zero);
            __m128i minpos_low  = _mm_minpos_epu16(low64);
            __m128i minpos_high = _mm_minpos_epu16(high64);
            __m128i high_index_addition = _mm_setr_epi16(0, 8, 0, 0, 0, 0, 0, 0);
            minpos_high = _mm_add_epi16(minpos_high, high_index_addition);
            __m128i min_num_all = _mm_min_epu16(minpos_low, minpos_high);
            __m128i min_in_high_mask = _mm_cmpgt_epi16(min_num_all, minpos_high);
            __m128i minpos_8_low = _mm_andnot_si128(minpos_low, min_in_high_mask);
            __m128i minpos_8_high = _mm_and_si128(high64, min_in_high_mask);
            __m128i minpos_u16 = _mm_and_si128(minpos_8_low, minpos_8_high);
            //minpos = _mm_packus_epi16(minpos_u16, zero);
            minpos = minpos_u16;
        }
#else
        //
        // Horizontal minimum and maximum using SSE
        // See: https://stackoverflow.com/questions/22256525/horizontal-minimum-and-maximum-using-sse
        //
        if (MaxLength <= 8) {
            BitVec08x16 nums = this->m128;
            nums = _mm_min_epu8(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm_min_epu8(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            BitVec08x16 minnum_u8 = _mm_min_epu8(nums, _mm_srli_epi16(nums, 8));
            minpos = minnum_u8;
        }
        else {
            BitVec08x16 nums = this->m128;
            nums = _mm_min_epu8(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm_min_epu8(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm_min_epu8(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            BitVec08x16 minnum_u8 = _mm_min_epu8(nums, _mm_srli_epi16(nums, 8));
            minpos = minnum_u8;
        }
#endif // __SSE4_1__
    }

    template <size_t MaxLength>
    uint32_t min_u8() const {
        BitVec08x16 min_nums;
        this->min_u8<MaxLength>(min_nums);
        uint32_t min_num = (uint32_t)SSE::mm_cvtsi128_si32_low(min_nums.m128);
        return min_num;
    }

    template <size_t MaxLength>
    void minpos8(BitVec08x16 & minpos) const {
        this->min_u8<MaxLength>(minpos);
    }

    template <size_t MaxLength>
    uint32_t minpos8() const {
        BitVec08x16 minpos;
        this->minpos8<MaxLength>(minpos);
        uint32_t min_and_pos = (uint32_t)_mm_cvtsi128_si32(minpos.m128);
        return min_and_pos;
    }

    template <size_t MaxLength>
    void min_i16(BitVec08x16 & min_num) const {
        __m128i nums = this->m128;
        if (MaxLength <= 4) {
            nums = _mm_min_epi16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));

            // The minimum number of first 4 x int16_t
            min_num = _mm_min_epi16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else {
            nums = _mm_min_epi16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm_min_epi16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of total 8 x int16_t
            min_num = _mm_min_epi16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
    }

    template <size_t MaxLength>
    void min_u16(BitVec08x16 & min_num) const {
#if defined(__SSE4_1__)
        min_num = _mm_minpos_epu16(this->m128);
#else
        __m128i nums = this->m128;
        if (MaxLength <= 4) {
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));

            // The minimum number of first 4 x int16_t
            min_num = _mm_min_epu16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else {
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of total 8 x int16_t
            min_num = _mm_min_epu16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
#endif
    }   

    template <size_t MaxLength>
    int32_t min_i16() const {
        BitVec08x16 min_nums;
        this->min_i16<MaxLength>(min_nums);
        // _mm_cvtsi128_si32(m128i) is faster than _mm_extract_epi16(m128i, index)
        return (int32_t)SSE::mm_cvtsi128_si32_low(min_nums.m128);
    }

    template <size_t MaxLength>
    uint32_t min_u16() const {
        BitVec08x16 min_nums;
        this->min_u16<MaxLength>(min_nums);
        return (uint32_t)SSE::mm_cvtsi128_si32_low(min_nums.m128);
    }

    static uint32_t minpos16_get_num(const BitVec08x16 & minpos) {
        // _mm_cvtsi128_si32(m128i) is faster than _mm_extract_epi16(m128i, index)
        uint32_t min_num = (uint32_t)SSE::mm_cvtsi128_si32_low(minpos.m128);
        return min_num;
    }

    static uint32_t minpos16_get_index(const BitVec08x16 & minpos) {
        // _mm_cvtsi128_si32(m128i) is faster than _mm_extract_epi16(m128i, index)
        uint32_t index = (uint32_t)SSE::mm_cvtsi128_si32_high(minpos.m128);
        return index;
    }

    template <size_t MaxLength>
    void minpos16(BitVec08x16 & minpos) const {
#if defined(__SSE4_1__)
        minpos = _mm_minpos_epu16(this->m128);
#else
        __m128i minnum_u16;
        __m128i nums = this->m128;
        if (MaxLength <= 4) {
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));

            // The minimum number of first 4 x int16_t
            minnum_u16 = _mm_min_epu16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else {
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm_min_epu16(nums, _mm_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of total 8 x int16_t
            minnum_u16 = _mm_min_epu16(nums, _mm_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }

        // Mixed the min_num (16bit) and index (3bit) to minpos info (32bit).

        // Get the index of minimum number
        uint32_t min_num = SSE::mm_cvtsi128_si32_low(minnum_u16);
        uint32_t min_index = (uint32_t)this->template indexOfIsEqual16<true>(min_num);

        uint32_t min_and_index = min_num | (min_index << 16);
        minpos = _mm_cvtsi32_si128(min_and_index);
#endif
    }

    template <size_t MaxLength>
    inline uint32_t minpos16() const {
        BitVec08x16 minpos;
        this->minpos16<MaxLength>(minpos);
        return (uint32_t)_mm_cvtsi128_si32(minpos.m128);
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(uint32_t & old_min_num, int & min_index) const {
        BitVec08x16 minpos;
        this->minpos16<MaxLength>(minpos);
        uint32_t min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos.m128);
        uint32_t min_num = min_and_index & 0xFFFFUL;
        if (min_num < old_min_num) {
            old_min_num = min_num;
            min_index = (int)(min_and_index >> 16U);
        }
        return min_num;
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(int & min_index) const {
        uint32_t min_and_index = this->minpos16<MaxLength>();
        uint32_t min_num = min_and_index & 0xFFFFUL;
        min_index = (int)(min_and_index >> 16U);
        return min_num;
    }
};

#endif // >= SSE2

#if defined(__SSE2__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__SSE4A__) || defined(__SSE4a__) \
 || defined(__SSE4_1__) || defined(__SSE4_2__)

struct BitVec16x16 {
    BitVec08x16 low;
    BitVec08x16 high;

    BitVec16x16() noexcept : low(), high() {}

    // non-explicit conversions intended
    BitVec16x16(const BitVec16x16 & src) noexcept
        : low(src.low), high(src.high) {}

    BitVec16x16(const BitVec08x16 & _low, const BitVec08x16 & _high) noexcept
        : low(_low), high(_high) {}

    BitVec16x16(uint8_t c00, uint8_t c01, uint8_t c02, uint8_t c03,
                uint8_t c04, uint8_t c05, uint8_t c06, uint8_t c07,
                uint8_t c08, uint8_t c09, uint8_t c10, uint8_t c11,
                uint8_t c12, uint8_t c13, uint8_t c14, uint8_t c15,
                uint8_t c16, uint8_t c17, uint8_t c18, uint8_t c19,
                uint8_t c20, uint8_t c21, uint8_t c22, uint8_t c23,
                uint8_t c24, uint8_t c25, uint8_t c26, uint8_t c27,
                uint8_t c28, uint8_t c29, uint8_t c30, uint8_t c31) :
            low(c00, c01, c02, c03, c04, c05, c06, c07,
                c08, c09, c10, c11, c12, c13, c14, c15),
            high(c16, c17, c18, c19, c20, c21, c22, c23,
                 c24, c25, c26, c27, c28, c29, c30, c31) {}

    BitVec16x16(uint16_t w00, uint16_t w01, uint16_t w02, uint16_t w03,
                uint16_t w04, uint16_t w05, uint16_t w06, uint16_t w07,
                uint16_t w08, uint16_t w09, uint16_t w10, uint16_t w11,
                uint16_t w12, uint16_t w13, uint16_t w14, uint16_t w15) :
            low(w00, w01, w02, w03, w04, w05, w06, w07),
            high(w08, w09, w10, w11, w12, w13, w14, w15) {}

    BitVec16x16(uint32_t i00, uint32_t i01, uint32_t i02, uint32_t i03,
                uint32_t i04, uint32_t i05, uint32_t i06, uint32_t i07) :
            low(i00, i01, i02, i03), high(i04, i05, i06, i07)  {}

    BitVec16x16(uint64_t q00, uint64_t q01, uint64_t q02, uint64_t q03) :
            low(q00, q01), high(q02, q03) {}

    BitVec16x16 & setLow(const BitVec08x16 & _low) {
        this->low = _low;
        return *this;
    }

    BitVec16x16 & setHigh(const BitVec08x16 & _high) {
        this->high = _high;
        return *this;
    }

    BitVec16x16 & mergeFrom(const BitVec08x16 & _low, const BitVec08x16 & _high) {
        this->low = _low;
        this->high = _high;
        return *this;
    }

    inline void castTo(BitVec08x16 & _low) const {
        _low = this->low;
    }

    inline void castTo(BitVec16x16 & vec) const {
        vec = *this;
    }

    inline void splitTo(BitVec08x16 & _low, BitVec08x16 & _high) const {
        _low = this->low;
        _high = this->high;
    }

    inline BitVec16x16 & operator = (const BitVec16x16 & right) {
        this->low = right.low;
        this->high = right.high;
        return *this;
    }

    inline void loadAligned(const void * mem_addr) {
        const void * mem_128i_low = mem_addr;
        const void * mem_128i_high = (const void *)((const __m128i *)mem_addr + 1);
        this->low.loadAligned(mem_128i_low);
        this->high.loadAligned(mem_128i_high);
    }

    inline void loadUnaligned(const void * mem_addr) {
        const void * mem_128i_low = mem_addr;
        const void * mem_128i_high = (const void *)((const __m128i *)mem_addr + 1);
        this->low.loadUnaligned(mem_128i_low);
        this->high.loadUnaligned(mem_128i_high);
    }

    inline void saveAligned(void * mem_addr) const {
        void * mem_128i_low = mem_addr;
        void * mem_128i_high = (void *)((__m128i *)mem_addr + 1);
        this->low.saveAligned(mem_128i_low);
        this->high.saveAligned(mem_128i_high);
    }

    inline void saveUnaligned(void * mem_addr) const {
        void * mem_128i_low = mem_addr;
        void * mem_128i_high = (void *)((__m128i *)mem_addr + 1);
        this->low.saveUnaligned(mem_128i_low);
        this->high.saveUnaligned(mem_128i_high);
    }

    static void copyAligned(const void * src_mem_addr, void * dest_mem_addr) {
#ifndef __AVX__
        const void * src_mem_addr_low = src_mem_addr;
        const void * src_mem_addr_high = (const void *)((const __m128i *)src_mem_addr + 1);
        void * dest_mem_addr_low = dest_mem_addr;
        void * dest_mem_addr_high = (void * )((__m128i *)dest_mem_addr + 1);

        BitVec08x16::copyAligned(src_mem_addr_low, dest_mem_addr_low);
        BitVec08x16::copyAligned(src_mem_addr_high, dest_mem_addr_high);
#else
        __m256i tmp = _mm256_load_si256((const __m256i *)src_mem_addr);
        _mm256_store_si256((__m256i *)dest_mem_addr, tmp);
#endif
    }

    static void copyUnaligned(const void * src_mem_addr, void * dest_mem_addr) {
#ifndef __AVX__
        const void * src_mem_addr_low = src_mem_addr;
        const void * src_mem_addr_high = (const void *)((const __m128i *)src_mem_addr + 1);
        void * dest_mem_addr_low = dest_mem_addr;
        void * dest_mem_addr_high = (void * )((__m128i *)dest_mem_addr + 1);

        BitVec08x16::copyUnaligned(src_mem_addr_low, dest_mem_addr_low);
        BitVec08x16::copyUnaligned(src_mem_addr_high, dest_mem_addr_high);
#else
        __m256i tmp = _mm256_loadu_si256((const __m256i *)src_mem_addr);
        _mm256_storeu_si256((__m256i *)dest_mem_addr, tmp);
#endif
    }

    inline bool operator == (const BitVec16x16 & other) const {
        return this->isEqual(other);
    }

    inline bool operator != (const BitVec16x16 & other) const {
        return this->isNotEqual(other);
    }

    // Logical operation
    inline BitVec16x16 operator & (const BitVec16x16 & vec) {
        BitVec16x16 tmp(this->low, this->high);
        tmp._and(vec);
        return tmp;
    }

    inline BitVec16x16 operator | (const BitVec16x16 & vec) {
        BitVec16x16 tmp(this->low, this->high);
        tmp._or(vec);
        return tmp;
    }

    inline BitVec16x16 operator ^ (const BitVec16x16 & vec) {
        BitVec16x16 tmp(this->low, this->high);
        tmp._xor(vec);
        return tmp;
    }

    inline BitVec16x16 operator ~ () {
        BitVec16x16 tmp(this->low, this->high);
        tmp._not();
        return tmp;
    }

    inline BitVec16x16 operator ! () {
        BitVec16x16 tmp(this->low, this->high);
        tmp._not();
        return tmp;
    }

    // Logical operation
    inline BitVec16x16 & operator &= (const BitVec16x16 & vec) {
        this->_and(vec);
        return *this;
    }

    inline BitVec16x16 & operator |= (const BitVec16x16 & vec) {
        this->_or(vec);
        return *this;
    }

    inline BitVec16x16 & operator ^= (const BitVec16x16 & vec) {
        this->_xor(vec);
        return *this;
    }

    // Logical operation
    inline BitVec16x16 & _and(const BitVec16x16 & vec) {
        this->low._and(vec.low);
        this->high._and(vec.high);
        return *this;
    }

    inline BitVec16x16 & and_not(const BitVec16x16 & vec) {
        this->low.and_not(vec.low);
        this->high.and_not(vec.high);
        return *this;
    }

    inline BitVec16x16 & _or(const BitVec16x16 & vec) {
        this->low._or(vec.low);
        this->high._or(vec.high);
        return *this;
    }

    inline BitVec16x16 & _xor(const BitVec16x16 & vec) {
        this->low._xor(vec.low);
        this->high._xor(vec.high);
        return *this;
    }

    // Logical not: !
    inline BitVec16x16 & _not() {
        this->low._not();
        this->high._not();
        return *this;
    }

    // fill
    inline void fill_u8(uint8_t value) {
        this->low.fill_u8(value);
        this->high.fill_u8(value);
    }

    inline void fill_u16(uint16_t value) {
        this->low.fill_u16(value);
        this->high.fill_u16(value);
    }

    inline void fill_u32(uint32_t value) {
        this->low.fill_u32(value);
        this->high.fill_u32(value);
    }

    inline void fill_u64(uint64_t value) {
        this->low.fill_u64(value);
        this->high.fill_u64(value);
    }

    static bool isMemEqual(const void * mem_addr_1, const void * mem_addr_2) {
        const IntVec256 * pIntVec256_1 = (const IntVec256 *)mem_addr_1;
        const IntVec256 * pIntVec256_2 = (const IntVec256 *)mem_addr_2;
        bool isEqual = (pIntVec256_1->u64[0] == pIntVec256_2->u64[0]) && (pIntVec256_1->u64[1] == pIntVec256_2->u64[1]) &&
                       (pIntVec256_1->u64[2] == pIntVec256_2->u64[2]) && (pIntVec256_1->u64[3] == pIntVec256_2->u64[3]);
        return isEqual;
    }

    inline void setAllZeros() {
        this->low.setAllZeros();
        this->high.setAllZeros();
    }

    inline void setAllOnes() {
        this->low.setAllOnes();
        this->high.setAllOnes();
    }

    inline bool isAllZeros() const {
        return (this->low.isAllZeros() && this->high.isAllZeros());
    }

    inline bool isAllOnes() const {
        return (this->low.isAllOnes() && this->high.isAllOnes());
    }

    inline bool isNotAllZeros() const {
        return (this->low.isNotAllZeros() && this->high.isNotAllZeros());
    }

    inline bool isNotAllOnes() const {
        return (this->low.isNotAllOnes() && this->high.isNotAllOnes());
    }

    // Is mixed by zeros and ones
    inline bool isMixZerosAndOnes(const BitVec16x16 & other) const {
        return (this->low.isMixZerosAndOnes(other.low) && this->high.isMixZerosAndOnes(other.high));
    }

    // Is not all zeros and all ones
    inline bool isNotAllZerosAndAllOnes(const BitVec16x16 & other) const {
        return (this->low.isNotAllZerosAndAllOnes(other.low) && this->high.isNotAllZerosAndAllOnes(other.high));
    }

    inline bool isEqual(const BitVec16x16 & other) const {
        return (this->low.isEqual(other.low) && this->high.isEqual(other.high));
    }

    inline bool isNotEqual(const BitVec16x16 & other) const {
        return (this->low.isNotEqual(other.low) && this->high.isNotEqual(other.high));
    }

    inline bool hasAnyZero() const {
        return (this->low.hasAnyZero() && this->high.hasAnyZero());
    }

    inline bool hasAnyOne() const {
        return (this->low.hasAnyOne() && this->high.hasAnyOne());
    }

    inline bool hasIntersects(const BitVec16x16 & other) const {
        return (this->low.hasIntersects(other.low) && this->high.hasIntersects(other.high));
    }

    inline bool isSubsetOf(const BitVec16x16 & other) const {
        return (this->low.isSubsetOf(other.low) && this->high.isSubsetOf(other.high));
    }

    inline BitVec16x16 whichIsEqual(const BitVec16x16 & other) const {
        return BitVec16x16(this->low.whichIsEqual(other.low), this->high.whichIsEqual(other.high));
    }

    inline BitVec16x16 whichIsNotEqual(const BitVec16x16 & other) const {
        return BitVec16x16(this->low.whichIsNotEqual(other.low), this->high.whichIsNotEqual(other.high));
    }

    inline BitVec16x16 whichIsMoreThan(const BitVec16x16 & other) const {
        return BitVec16x16(this->low.whichIsMoreThan(other.low), this->high.whichIsMoreThan(other.high));
    }

    inline BitVec16x16 whichIsLessThan(const BitVec16x16 & other) const {
        return BitVec16x16(this->low.whichIsLessThan(other.low), this->high.whichIsLessThan(other.high));
    }

    inline BitVec16x16 whichIsZeros() const {
        return BitVec16x16(this->low.whichIsZeros(), this->high.whichIsZeros());
    }

    inline BitVec16x16 whichIsOnes() const {
        return BitVec16x16(this->low.whichIsOnes(), this->high.whichIsOnes());
    }

    inline BitVec16x16 whichIsNonZero() const {
        return BitVec16x16(this->low.whichIsNonZero(), this->high.whichIsNonZero());
    }

    inline BitVec16x16 whichIsMoreThanZero() const {
        return BitVec16x16(this->low.whichIsMoreThanZero(), this->high.whichIsMoreThanZero());
    }

    inline BitVec16x16 whichIsLessThanZero() const {
        return BitVec16x16(this->low.whichIsLessThanZero(), this->high.whichIsLessThanZero());
    }

    inline BitVec16x16 whichIsEqual16(uint16_t num) const {
        return BitVec16x16(this->low.whichIsEqual16(num), this->high.whichIsEqual16(num));
    }

    inline BitVec16x16 whichIsNotEqual16(uint16_t num) const {
        return BitVec16x16(this->low.whichIsNotEqual16(num), this->high.whichIsNotEqual16(num));
    }

    template <bool isNonZeros>
    inline int indexOfIsEqual16(uint32_t num) const {
        __m128i num_mask = _mm_set1_epi16((int16_t)num);
        int index_low = this->low.template indexOfIsEqual16<false, true>(num_mask);
        if (index_low != -1) {
            return index_low;
        }
        else {
            int index_high = this->high.template indexOfIsEqual16<isNonZeros, true>(num_mask);
            if (isNonZeros || (index_high != -1)) {
                return (8 + index_high);
            }
        }
        return -1;
    }

    template <bool isNonZeros, bool isRepeat = true>
    inline int indexOfIsEqual16(const BitVec08x16 & in_num_mask) const {
        BitVec08x16 num_mask;
        if (!isRepeat) {
            // If the num_mask is not a repeat mask,
            // we repeat the first 16bit integer first.
#if defined(__AVX2__)
            num_mask = _mm_broadcastw_epi16(in_num_mask.m128);
#elif defined(__SSSE3__)
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shuffle_epi8(in_num_mask.m128, lookup_mask);
#else
            // SSE2
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = in_num_mask;
            num_mask = _mm_shufflelo_epi16(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
            num_mask = _mm_shuffle_epi32(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
#endif
        }
        else {
            num_mask = in_num_mask;
        }

#if 0
        return this->firstIndexOfOnes16<isNonZeros>(num_mask);
#elif 0
        BitVec08x16 compare_mask_low = this->low.whichIsEqual(num_mask);
        BitVec08x16 compare_mask_high = this->high.whichIsEqual(num_mask);
        int compare_mask_low_16 = _mm_movemask_epi8(compare_mask_low.m128);
        int compare_mask_high_16 = _mm_movemask_epi8(compare_mask_high.m128);

        if (compare_mask_low_16 != 0) {
            uint32_t first_offset = BitUtils::bsf(compare_mask_low_16);
            int index_low = (int)(first_offset >> 1U);
            return index_low;
        }
        else if (isNonZeros || (compare_mask_high_16 != 0)) {
            assert(!isNonZeros || (compare_mask_high_16 != 0));
            uint32_t first_offset = BitUtils::bsf(compare_mask_high_16);
            int index_high = (int)(first_offset >> 1U);
            return (8 + index_high);
        }
        else return -1;
#else
        int index_low = this->low.template indexOfIsEqual16<false, true>(num_mask);
        if (index_low != -1) {
            return index_low;
        }
        else {
            int index_high = this->high.template indexOfIsEqual16<isNonZeros, true>(num_mask);
            if (isNonZeros || (index_high != -1)) {
                return (8 + index_high);
            }
            else return -1;
        }
#endif
    }

    template <bool isNonZeros, bool isRepeat = true>
    inline int indexOfIsEqual16(const BitVec16x16 & num_mask) const {
        return this->template indexOfIsEqual16<isNonZeros, isRepeat>(num_mask.low);
    }

    template <bool isRepeat = true>
    inline int maskOfIsEqual16(const BitVec08x16 & in_num_mask) const {
        BitVec08x16 num_mask;
        if (!isRepeat) {
            // If the num_mask is not a repeat mask,
            // we repeat the first 16bit integer first.
#if defined(__AVX2__)
            num_mask = _mm_broadcastw_epi16(in_num_mask.m128);
#elif defined(__SSSE3__)
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = _mm_shuffle_epi8(in_num_mask.m128, lookup_mask);
#else
            // SSE2
            __m128i lookup_mask = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
            num_mask = in_num_mask;
            num_mask = _mm_shufflelo_epi16(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
            num_mask = _mm_shuffle_epi32(num_mask.m128, _MM_SHUFFLE(0, 0, 0, 0));
#endif
        }
        else {
            num_mask = in_num_mask;
        }

        BitVec16x16 is_equal_mask;
        is_equal_mask.low = this->low.whichIsEqual(num_mask);
        is_equal_mask.high = this->high.whichIsEqual(num_mask);

        uint32_t compare_mask_low16 = this->low.maskOfOnes16(is_equal_mask.low);
        uint32_t compare_mask_high16 = this->high.maskOfOnes16(is_equal_mask.high);
        uint32_t compare_mask_32 = (compare_mask_high16 << 16U) | compare_mask_low16;
        return compare_mask_32;
    }

    template <bool isRepeat = true>
    inline int maskOfIsEqual16(const BitVec16x16 & num_mask) const {
        return this->template maskOfIsEqual16<isRepeat>(num_mask.low);
    }

    template <bool isNonZeros>
    inline int firstIndexOfOnes16(const BitVec08x16 & compare_mask) const {
#if 0
        BitVec08x16 compare_mask_low = this->low.whichIsEqual(compare_mask);
        BitVec08x16 compare_mask_high = this->high.whichIsEqual(compare_mask);
        int compare_mask_low_16 = _mm_movemask_epi8(compare_mask_low.m128);
        int compare_mask_high_16 = _mm_movemask_epi8(compare_mask_high.m128);

        if (compare_mask_low_16 != 0) {
            uint32_t first_offset = BitUtils::bsf(compare_mask_low_16);
            int index_low = (int)(first_offset >> 1U);
            return index_low;
        }
        else if (isNonZeros || (compare_mask_high_16 != 0)) {
            assert(!isNonZeros || (compare_mask_high_16 != 0));
            uint32_t first_offset = BitUtils::bsf(compare_mask_high_16);
            int index_high = (int)(first_offset >> 1U);
            return (8 + index_high);
        }
        else return -1;
#else
        int index_low = this->low.template firstIndexOfOnes16<false>(compare_mask);
        if (index_low != -1) {
            return index_low;
        }
        else {
            int index_high = this->high.template firstIndexOfOnes16<isNonZeros>(compare_mask);
            if (isNonZeros || (index_high != -1)) {
                return (8 + index_high);
            }
            else return -1;
        }
#endif
    }

    template <bool isNonZeros>
    inline int firstIndexOfOnes16(const BitVec16x16 & compare_mask) const {
        return this->template firstIndexOfOnes16<isNonZeros>(compare_mask.low);
    }

    inline uint32_t maskOfOnes16(const BitVec08x16 & compare_mask) const {
        uint32_t compare_mask_low16 = this->low.maskOfOnes16(compare_mask);
        uint32_t compare_mask_high16 = this->high.maskOfOnes16(compare_mask);
        uint32_t compare_mask_32 = (compare_mask_high16 << 16U) | compare_mask_low16;
        return compare_mask_32;
    }

    inline uint32_t maskOfOnes16(const BitVec16x16 & compare_mask) const {
        uint32_t compare_mask_low16 = this->low.maskOfOnes16(compare_mask.low);
        uint32_t compare_mask_high16 = this->high.maskOfOnes16(compare_mask.high);
        uint32_t compare_mask_32 = (compare_mask_high16 << 16U) | compare_mask_low16;
        return compare_mask_32;
    }

    inline uint16_t extract(int index) const {
        #define CASE_LOW(x)  case x: return (uint16_t)_mm_extract_epi16(this->low.m128, x);
        #define CASE_HIGH(x) case x: return (uint16_t)_mm_extract_epi16(this->high.m128, (x - 8));
        switch (index) {
            CASE_LOW(0)
            CASE_LOW(1)
            CASE_LOW(2)
            CASE_LOW(3)
            CASE_LOW(4)
            CASE_LOW(5)
            CASE_LOW(6)
            CASE_LOW(7)
            CASE_HIGH(8)
            CASE_HIGH(9)
            CASE_HIGH(10)
            CASE_HIGH(11)
            CASE_HIGH(12)
            CASE_HIGH(13)
            CASE_HIGH(14)
            CASE_HIGH(15)
            default:
                return uint16_t(-1);
        }
        #undef CASE_LOW
        #undef CASE_HIGH
    }

    inline void insert(int index, uint16_t value) {
        #define CASE_LOW(x)  case x: this->low.m128  = _mm_insert_epi16(this->low.m128, value, x); break;
        #define CASE_HIGH(x) case x: this->high.m128 = _mm_insert_epi16(this->high.m128, value, (x - 8)); break;
        switch (index) {
            CASE_LOW(0)
            CASE_LOW(1)
            CASE_LOW(2)
            CASE_LOW(3)
            CASE_LOW(4)
            CASE_LOW(5)
            CASE_LOW(6)
            CASE_LOW(7)
            CASE_HIGH(8)
            CASE_HIGH(9)
            CASE_HIGH(10)
            CASE_HIGH(11)
            CASE_HIGH(12)
            CASE_HIGH(13)
            CASE_HIGH(14)
            CASE_HIGH(15)
            default:
                break;
        }
        #undef CASE_LOW
        #undef CASE_HIGH
    }

    template <size_t MaxLength, size_t MaxBits>
    inline BitVec16x16 popcount16() const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;
        return BitVec16x16(this->low.popcount16<MaxLenLow, MaxBits>(), this->high.popcount16<MaxLenHigh, MaxBits>());
    }

    template <size_t MaxLength>
    void min_i16(BitVec16x16 & min_num) const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        if (MaxLength <= 8) {
            this->low.min_i16<MaxLenLow>(min_num.low);
        }
        else if (MaxLength == 9) {
            this->low.min_i16<MaxLenLow>(min_num.low);

            BitVec08x16 minnum_16 = _mm_min_epi16(min_num.low.m128, this->high.m128);
            min_num.setLow(minnum_16);
        }
        else {
            this->low.min_i16<MaxLenLow>(min_num.low);
            this->high.min_i16<MaxLenHigh>(min_num.high);

            BitVec08x16 minnum_16 = _mm_min_epi16(min_num.low.m128, min_num.high.m128);
            min_num.setLow(minnum_16);
        }
    }

    template <size_t MaxLength>
    void min_u16(BitVec16x16 & min_num) const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        if (MaxLength <= 8) {
            this->low.min_u16<MaxLenLow>(min_num.low);
        }
        else if (MaxLength == 9) {
            this->low.min_u16<MaxLenLow>(min_num.low);

            BitVec08x16 minnum_16 = _mm_min_epu16(min_num.low.m128, this->high.m128);
            min_num.setLow(minnum_16);
        }
        else {
            this->low.min_u16<MaxLenLow>(min_num.low);
            this->high.min_u16<MaxLenHigh>(min_num.high);

            BitVec08x16 minnum_16 = _mm_min_epu16(min_num.low.m128, min_num.high.m128);
            min_num.setLow(minnum_16);
        }
    }

    template <size_t MaxLength>
    inline int32_t min_i16() const {
        BitVec16x16 min_num;
        this->min_i16<MaxLength>(min_num);
        // _mm_cvtsi128_si32(m128i) faster than _mm_extract_epi16(m128i, index)
        return (int32_t)SSE::mm_cvtsi128_si32_low(min_num.low.m128);
    }

    template <size_t MaxLength>
    inline uint32_t min_u16() const {
        BitVec16x16 min_num;
        this->min_u16<MaxLength>(min_num);
        return (uint32_t)SSE::mm_cvtsi128_si32_low(min_num.low.m128);
    }

    template <size_t MaxLength>
    void minpos8(BitVec16x16 & minpos) const {
        static const size_t MaxLenLow = (MaxLength < 16) ? MaxLength : 16;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 16) ? (MaxLength - MaxLenLow) : 16;

        if (MaxLength <= 8) {
            this->low.minpos8<MaxLenLow>(minpos.low);
        }
        else if (MaxLength == 9) {
            this->low.minpos8<MaxLenLow>(minpos.low);
            BitVec08x16 minpos_u8 = _mm_min_epu8(minpos.low.m128, this->high.m128);
            minpos.setLow(minpos_u8);
        }
        else {
            this->low.minpos8<MaxLenLow>(minpos.low);
            this->high.minpos8<MaxLenHigh>(minpos.high);
            BitVec08x16 minpos_u8 = _mm_min_epu8(minpos.low.m128, minpos.high.m128);
            minpos.setLow(minpos_u8);
        }
    }

    template <size_t MaxLength>
    inline uint32_t minpos8() const {
        BitVec16x16 minpos;
        this->minpos8<MaxLength>(minpos);
        uint32_t min_and_pos = (uint32_t)_mm_cvtsi128_si32(minpos.low.m128);
        return min_and_pos;
    }

    static inline uint32_t minpos16_get_num(const BitVec16x16 & minpos) {
        return BitVec08x16::minpos16_get_num(minpos.low);
    }

    static inline uint32_t minpos16_get_index(const BitVec16x16 & minpos) {
        return BitVec08x16::minpos16_get_index(minpos.low);
    }

#if defined(__SSE4_1__)

    template <size_t MaxLength>
    void minpos16_intermediate_result(BitVec16x16 & minpos, BitVec08x16 & min_result) const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        if (MaxLength <= 8) {
            this->low.minpos16<MaxLenLow>(minpos.low);
            // Here we'll never use the min_result value.
            // min_result = minpos.low;
        }
        else if (MaxLength == 9) {
            this->low.minpos16<MaxLenLow>(minpos.low);
            min_result = _mm_min_epu16(minpos.low.m128, this->high.m128);
        }
        else {
            this->low.minpos16<MaxLenLow>(minpos.low);
            this->high.minpos16<MaxLenHigh>(minpos.high);
            min_result = _mm_min_epu16(minpos.low.m128, minpos.high.m128);
        }
    }

    template <size_t MaxLength>
    int minpos16_get_index(BitVec16x16 & minpos, const BitVec08x16 & min_result) const {
        int min_index;
        if (MaxLength <= 8) {
            min_index = _mm_extract_epi16(minpos.low.m128, 1);
        }
        else if (MaxLength == 9) {
            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                min_index = _mm_extract_epi16(minpos.low.m128, 1);
            }
            else {
#ifndef NDEBUG
                this->high._minpos16<MaxLength>(minpos.high);
                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);
                assert(_mm_extract_epi16(minpos.high.m128, 1) == 0);
#endif
                min_index = 8;
            }
        }
        else {
            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                min_index = _mm_extract_epi16(minpos.low.m128, 1);
            }
            else {
                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);
                min_index = 8 + _mm_extract_epi16(minpos.high.m128, 1);
            }
        }
        return min_index;
    }

    template <size_t MaxLength>
    uint32_t minpos16_and_index(int & out_min_index) const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        uint32_t min_num;
        BitVec16x16 minpos;
        if (MaxLength <= 8) {
            this->low.minpos16<MaxLenLow>(minpos.low);
            uint32_t min_and_index = _mm_cvtsi128_si32(minpos.low.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);
            out_min_index = min_and_index >> 16U;
        }
        else if (MaxLength == 9) {
            this->low.minpos16<MaxLenLow>(minpos.low);

            BitVec08x16 min_result = _mm_min_epu16(minpos.low.m128, this->high.m128);
            uint32_t min_and_index = _mm_cvtsi128_si32(min_result.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);

            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                out_min_index = min_and_index >> 16U;
            }
            else {
#ifndef NDEBUG
                this->high.minpos16<MaxLenHigh>(minpos.high);
                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);
                assert(_mm_extract_epi16(minpos.high.m128, 1) == 0);
#endif
                out_min_index = 8;
            }
        }
        else {
            this->low.minpos16<MaxLenLow>(minpos.low);
            this->high.minpos16<MaxLenHigh>(minpos.high);

            BitVec08x16 min_result = _mm_min_epu16(minpos.low.m128, minpos.high.m128);
            uint32_t min_and_index = _mm_cvtsi128_si32(min_result.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);

            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                out_min_index = min_and_index >> 16U;
            }
            else {
                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);
                out_min_index = (_mm_cvtsi128_si32(minpos.high.m128) >> 16U) + 8;
            }
        }
        return min_num;
    }

#endif // __SSE4_1__

    template <size_t MaxLength>
    void minpos16(BitVec08x16 & minpos) const {
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        if (MaxLength <= 8) {
            this->low.minpos16<MaxLenLow>(minpos);
        }
        else if (MaxLength == 9) {
            this->low.minpos16<MaxLenLow>(minpos);

            BitVec08x16 min_result = _mm_min_epu16(minpos.m128, this->high.m128);
            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                // Do nothing !
            }
            else {
                uint32_t min_num = (uint32_t)(_mm_cvtsi128_si32(min_result.m128) & 0xFFFFUL);
#ifndef NDEBUG
                BitVec08x16 minpos_high;
                this->high.minpos16<MaxLenHigh>(minpos_high);
                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos_high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);
                assert(_mm_extract_epi16(minpos_high.m128, 1) == 0);
#endif
                const uint32_t min_index = 8;
                uint32_t min_and_index = min_num | (min_index << 16U);
                minpos = _mm_cvtsi32_si128(min_and_index);
            }
        }
        else {
            BitVec08x16 minpos_high;
            this->low.minpos16<MaxLenLow>(minpos);
            this->high.minpos16<MaxLenHigh>(minpos_high);

            BitVec08x16 min_result = _mm_min_epu16(minpos.m128, minpos_high.m128);
            __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.m128);
            int equal_mask_low = _mm_movemask_epi8(equal_result_low);
            if ((equal_mask_low & 0x00000003U) != 0) {
                // Do nothing !
            }
            else {
                uint32_t min_and_index = _mm_cvtsi128_si32(min_result.m128);
                uint32_t min_num = (uint32_t)(min_and_index & 0xFFFFUL);

                __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos_high.m128);
                int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                assert((equal_mask_high & 0x00000003U) != 0);

                uint32_t min_index = (_mm_cvtsi128_si32(minpos_high.m128) >> 16U) + 8;

                min_and_index = min_num | (min_index << 16U);
                minpos = _mm_cvtsi32_si128(min_and_index);
            }
        }
    }

    template <size_t MaxLength>
    inline uint32_t minpos16() const {
        BitVec08x16 minpos;
        this->minpos16<MaxLength>(minpos);
        return (uint32_t)_mm_cvtsi128_si32(minpos.m128);
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(uint32_t & old_min_num, int & out_min_index) const {
#if 1
        static const size_t MaxLenLow = (MaxLength < 8) ? MaxLength : 8;
        static const size_t MaxLenHigh = ((MaxLength - MaxLenLow) < 8) ? (MaxLength - MaxLenLow) : 8;

        uint32_t min_num;
        BitVec16x16 minpos;
        if (MaxLength <= 8) {
            this->low.minpos16<MaxLenLow>(minpos.low);
            uint32_t min_and_index = _mm_cvtsi128_si32(minpos.low.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);
            if (min_num < old_min_num) {
                old_min_num = min_num;
                out_min_index = min_and_index >> 16U;
            }
        }
        else if (MaxLength == 9) {
            this->low.minpos16<MaxLenLow>(minpos.low);

            BitVec08x16 min_result = _mm_min_epu16(minpos.low.m128, this->high.m128);
            uint32_t min_and_index = _mm_cvtsi128_si32(min_result.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);

            if (min_num < old_min_num) {
                old_min_num = min_num;
                __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
                int equal_mask_low = _mm_movemask_epi8(equal_result_low);
                if ((equal_mask_low & 0x00000003U) != 0) {
                    out_min_index = min_and_index >> 16U;
                }
                else {
    #ifndef NDEBUG
                    this->high.minpos16<MaxLenHigh>(minpos.high);
                    __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                    int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                    assert((equal_mask_high & 0x00000003U) != 0);
                    assert(_mm_extract_epi16(minpos.high.m128, 1) == 0);
    #endif
                    out_min_index = 8;
                }
            }
        }
        else {
            this->low.minpos16<MaxLenLow>(minpos.low);
            this->high.minpos16<MaxLenHigh>(minpos.high);

            BitVec08x16 min_result = _mm_min_epu16(minpos.low.m128, minpos.high.m128);
            uint32_t min_and_index = _mm_cvtsi128_si32(min_result.m128);
            min_num = (uint32_t)(min_and_index & 0xFFFFUL);

            if (min_num < old_min_num) {
                old_min_num = min_num;
                __m128i equal_result_low = _mm_cmpeq_epi16(min_result.m128, minpos.low.m128);
                int equal_mask_low = _mm_movemask_epi8(equal_result_low);
                if ((equal_mask_low & 0x00000003U) != 0) {
                    out_min_index = min_and_index >> 16U;
                }
                else {
                    __m128i equal_result_high = _mm_cmpeq_epi16(min_result.m128, minpos.high.m128);
                    int equal_mask_high = _mm_movemask_epi8(equal_result_high);
                    assert((equal_mask_high & 0x00000003U) != 0);
                    out_min_index = (_mm_cvtsi128_si32(minpos.high.m128) >> 16U) + 8;
                }
            }
        }
        return min_num;
#elif 1
        BitVec08x16 minpos;
        this->minpos16<MaxLength>(minpos);
        uint32_t min_and_index = (uint32_t)_mm_cvtsi128_si32(minpos.m128);
        uint32_t min_num = min_and_index & 0xFFFFUL;
        if (min_num < old_min_num) {
            old_min_num = min_num;
            out_min_index = (int)(min_and_index >> 16U);
        }
#else
        BitVec08x16 minpos;
        int min_index;
        uint32_t min_num = this->minpos16_and_index<MaxLength>(min_index);
        if (min_num < old_min_num) {
            old_min_num = min_num;
            out_min_index = min_index;
        }
#endif
        return min_num;
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(int & min_index) const {
        uint32_t min_and_index = this->minpos16<MaxLength>();
        uint32_t min_num = min_and_index & 0xFFFFUL;
        min_index = (int)(min_and_index >> 16U);
        return min_num;
    }
};

#endif // >= SSE2 && !__AVX2__

#if defined(__AVX2__) || defined(__AVX512VL__) || defined(__AVX512F__)

struct BitVec16x16_AVX {
    __m256i m256;

    BitVec16x16_AVX() noexcept {}
    BitVec16x16_AVX(__m256i m256i) noexcept : m256(m256i) {}
    BitVec16x16_AVX(__m128i low, __m128i high) noexcept
        : m256(_mm256_setr_m128i(low, high)) {}

    // non-explicit conversions intended
    BitVec16x16_AVX(const BitVec16x16_AVX & src) noexcept : m256(src.m256) {}
    
    BitVec16x16_AVX(const BitVec08x16 & low, const BitVec08x16 & high) noexcept
        : m256(_mm256_setr_m128i(low.m128, high.m128)) {}

    BitVec16x16_AVX(uint8_t c00, uint8_t c01, uint8_t c02, uint8_t c03,
                uint8_t c04, uint8_t c05, uint8_t c06, uint8_t c07,
                uint8_t c08, uint8_t c09, uint8_t c10, uint8_t c11,
                uint8_t c12, uint8_t c13, uint8_t c14, uint8_t c15,
                uint8_t c16, uint8_t c17, uint8_t c18, uint8_t c19,
                uint8_t c20, uint8_t c21, uint8_t c22, uint8_t c23,
                uint8_t c24, uint8_t c25, uint8_t c26, uint8_t c27,
                uint8_t c28, uint8_t c29, uint8_t c30, uint8_t c31) :
            m256(_mm256_setr_epi8(
                   c00, c01, c02, c03, c04, c05, c06, c07,
                   c08, c09, c10, c11, c12, c13, c14, c15,
                   c16, c17, c18, c19, c20, c21, c22, c23,
                   c24, c25, c26, c27, c28, c29, c30, c31)) {}

    BitVec16x16_AVX(uint16_t w00, uint16_t w01, uint16_t w02, uint16_t w03,
                uint16_t w04, uint16_t w05, uint16_t w06, uint16_t w07,
                uint16_t w08, uint16_t w09, uint16_t w10, uint16_t w11,
                uint16_t w12, uint16_t w13, uint16_t w14, uint16_t w15) :
            m256(_mm256_setr_epi16(
                   w00, w01, w02, w03, w04, w05, w06, w07,
                   w08, w09, w10, w11, w12, w13, w14, w15)) {}

    BitVec16x16_AVX(uint32_t i00, uint32_t i01, uint32_t i02, uint32_t i03,
                uint32_t i04, uint32_t i05, uint32_t i06, uint32_t i07) :
            m256(_mm256_setr_epi32(i00, i01, i02, i03, i04, i05, i06, i07)) {}

    BitVec16x16_AVX(uint64_t q00, uint64_t q01, uint64_t q02, uint64_t q03) :
            m256(_mm256_setr_epi64x(q00, q01, q02, q03)) {}

    BitVec16x16_AVX & mergeFrom(const BitVec08x16 & low, const BitVec08x16 & high) {
        this->m256 = _mm256_setr_m128i(low.m128, high.m128);
        return *this;
    }

    inline void castTo(BitVec08x16 & low) const {
        low = _mm256_castsi256_si128(this->m256);
    }

    inline void castTo(BitVec16x16 & xmm) const {
#if 0
        xmm.low = _mm256_extracti128_si256(this->m256, 0);
        xmm.high = _mm256_extracti128_si256(this->m256, 1);
#else
        // __m128i _mm256_extracti128_si256(__m256i a, const int imm8);
        xmm.high = _mm256_extracti128_si256(this->m256, 1);
        xmm.low = _mm256_castsi256_si128(this->m256);
#endif
    }

    inline void splitTo(BitVec08x16 & low, BitVec08x16 & high) const {
#if 1
        low = _mm256_extracti128_si256(this->m256, 0);
        high = _mm256_extracti128_si256(this->m256, 1);
#else
        high = _mm256_extracti128_si256(this->m256, 1);
        low = _mm256_castsi256_si128(this->m256);
#endif
    }

    inline BitVec08x16 getLow() const {
#if 1
        return _mm256_castsi256_si128(this->m256);
#else
        return _mm256_extracti128_si256(this->m256, 0);
#endif
    }

    inline BitVec08x16 getHigh() const {
        return _mm256_extracti128_si256(this->m256, 1);
    }

    inline BitVec16x16_AVX & operator = (const BitVec16x16_AVX & right) {
        this->m256 = right.m256;
        return *this;
    }

    inline BitVec16x16_AVX & operator = (const __m256i & right) {
        this->m256 = right;
        return *this;
    }

    inline void loadAligned(const void * mem_addr) {
        this->m256 = _mm256_load_si256((const __m256i *)mem_addr);
    }

    inline void loadUnaligned(const void * mem_addr) {
        this->m256 = _mm256_loadu_si256((const __m256i *)mem_addr);
    }

    inline void saveAligned(void * mem_addr) const {
        _mm256_store_si256((__m256i *)mem_addr, this->m256);
    }

    inline void saveUnaligned(void * mem_addr) const {
        _mm256_storeu_si256((__m256i *)mem_addr, this->m256);
    }

    static inline void copyAligned(const void * src_mem_addr, void * dest_mem_addr) {
        __m256i tmp = _mm256_load_si256((const __m256i *)src_mem_addr);
        _mm256_store_si256((__m256i *)dest_mem_addr, tmp);
    }

    static inline void copyUnaligned(const void * src_mem_addr, void * dest_mem_addr) {
        __m256i tmp = _mm256_loadu_si256((const __m256i *)src_mem_addr);
        _mm256_storeu_si256((__m256i *)dest_mem_addr, tmp);
    }

    inline bool operator == (const BitVec16x16_AVX & other) const {
        return this->isEqual(other);
    }

    inline bool operator != (const BitVec16x16_AVX & other) const {
        return this->isNotEqual(other);
    }

    // Logical operation
    inline BitVec16x16_AVX operator & (const BitVec16x16_AVX & vec) {
        return _mm256_and_si256(this->m256, vec.m256);
    }

    inline BitVec16x16_AVX operator | (const BitVec16x16_AVX & vec) {
        return _mm256_or_si256(this->m256, vec.m256);
    }

    inline BitVec16x16_AVX operator ^ (const BitVec16x16_AVX & vec) {
        return _mm256_xor_si256(this->m256, vec.m256);
    }

    inline BitVec16x16_AVX operator ~ () {
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        return _mm256_andnot_si256(this->m256, ones.m256);
    }

    inline BitVec16x16_AVX operator ! () {
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        return _mm256_andnot_si256(this->m256, ones.m256);
    }

    // Logical operation
    inline BitVec16x16_AVX & operator &= (const BitVec16x16_AVX & vec) {
        this->_and(vec);
        return *this;
    }

    inline BitVec16x16_AVX & operator |= (const BitVec16x16_AVX & vec) {
        this->_or(vec);
        return *this;
    }

    inline BitVec16x16_AVX & operator ^= (const BitVec16x16_AVX & vec) {
        this->_xor(vec);
        return *this;
    }

    // Logical operation
    inline BitVec16x16_AVX & _and(const BitVec16x16_AVX & vec) {
        this->m256 = _mm256_and_si256(this->m256, vec.m256);
        return *this;
    }

    inline BitVec16x16_AVX & and_not(const BitVec16x16_AVX & vec) {
        this->m256 = _mm256_andnot_si256(vec.m256, this->m256);
        return *this;
    }

    inline BitVec16x16_AVX & _or(const BitVec16x16_AVX & vec) {
        this->m256 = _mm256_or_si256(this->m256, vec.m256);
        return *this;
    }

    inline BitVec16x16_AVX & _xor(const BitVec16x16_AVX & vec) {
        this->m256 = _mm256_xor_si256(this->m256, vec.m256);
        return *this;
    }

    // Logical not: !
    inline BitVec16x16_AVX & _not() {
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        this->m256 = _mm256_andnot_si256(this->m256, ones.m256);
        return *this;
    }

    // fill
    inline void fill_u8(uint8_t value) {
        this->m256 = _mm256_set1_epi8(value);
    }

    inline void fill_u16(uint16_t value) {
        this->m256 = _mm256_set1_epi16(value);
    }

    inline void fill_u32(uint32_t value) {
        this->m256 = _mm256_set1_epi32(value);
    }

    inline void fill_u64(uint64_t value) {
        this->m256 = _mm256_set1_epi64x(value);
    }

    static bool isMemEqual(const void * mem_addr_1, const void * mem_addr_2) {
        const IntVec256 * pIntVec256_1 = (const IntVec256 *)mem_addr_1;
        const IntVec256 * pIntVec256_2 = (const IntVec256 *)mem_addr_2;
        bool isEqual = (pIntVec256_1->u64[0] == pIntVec256_2->u64[0]) && (pIntVec256_1->u64[1] == pIntVec256_2->u64[1]) &&
                       (pIntVec256_1->u64[2] == pIntVec256_2->u64[2]) && (pIntVec256_1->u64[3] == pIntVec256_2->u64[3]);
        return isEqual;
    }

    inline void setAllZeros() {
        // Equivalent to AVX: _mm256_setzero_si256()
        this->m256 = _mm256_xor_si256(this->m256, this->m256);
    }

    inline void setAllOnes() {
        // Reference from _mm256_test_all_ones() macro
        this->m256 = _mm256_cmpeq_epi32(this->m256, this->m256);
    }

    inline bool isAllZeros() const {
        return (_mm256_test_all_zeros(this->m256, this->m256) == 1);
    }

    inline bool isAllOnes() const {
        return (_mm256_test_all_ones(this->m256) == 1);
    }

    inline bool isNotAllZeros() const {
        return (_mm256_test_all_zeros(this->m256, this->m256) == 0);
    }

    inline bool isNotAllOnes() const {
        return (_mm256_test_all_ones(this->m256) == 0);
    }

    // Is mixed by zeros and ones
    inline bool isMixZerosAndOnes(const BitVec16x16_AVX & other) const {
        return (_mm256_test_mix_ones_zeros(this->m256, other.m256) == 1);
    }

    // Is not all zeros and all ones
    inline bool isNotAllZerosAndAllOnes(const BitVec16x16_AVX & other) const {
        return (_mm256_test_mix_ones_zeros(this->m256, other.m256) == 0);
    }

    inline bool isEqual(const BitVec16x16_AVX & other) const {
#if 1
        // isAllZeros() faster than isAllOnes(), because it's instructions less than isAllOnes().
        BitVec16x16_AVX is_neq_mask = _mm256_xor_si256(this->m256, other.m256);
        return is_neq_mask.isAllZeros();
#else
        BitVec16x16_AVX is_eq_mask = _mm256_cmpeq_epi16(this->m256, other.m256);
        return is_eq_mask.isAllOnes();
#endif
    }

    inline bool isNotEqual(const BitVec16x16_AVX & other) const {
#if 1
        // isNotAllZeros() faster than isNotAllOnes(), because it's instructions less than isNotAllOnes().
        BitVec16x16_AVX is_neq_mask = _mm256_xor_si256(this->m256, other.m256);
        return is_neq_mask.isNotAllZeros();
#else
        BitVec16x16_AVX is_eq_mask = _mm256_cmpeq_epi16(this->m256, other.m256);
        return is_eq_mask.isNotAllOnes();
#endif
    }

    inline bool hasAnyZero() const {
#if 1
        return this->isNotAllOnes();
#else
#if defined(__AVX512VL__) && defined(__AVX512BW__)
        return (_mm256_cmp_epi16_mask(this->m256, _mm256_setzero_si256(), _MM_CMPINT_EQ) != 0);
#else
        BitVec16x16_AVX which_is_zero = this->whichIsZeros();
        return (_mm256_movemask_epi8(which_is_zero.m256) != 0);
#endif
#endif
    }

    inline bool hasAnyOne() const {
#if 1
        return this->isNotAllZeros();
#else
#if defined(__AVX512VL__) && defined(__AVX512BW__)
        return (_mm256_cmp_epi16_mask(this->m256, _mm256_setzero_si256(), _MM_CMPINT_GT) != 0);
#else
        BitVec16x16_AVX which_is_non_zero = this->whichIsNonZero();
        return (_mm256_movemask_epi8(which_is_non_zero.m256) != 0);
#endif
#endif
    }

    inline bool hasIntersects(const BitVec16x16_AVX & other) const {
        return (_mm256_testz_si256(this->m256, other.m256) == 0);
    }

    inline bool isSubsetOf(const BitVec16x16_AVX & other) const {
        return (_mm256_testc_si256(other.m256, this->m256) == 1);
    }

    inline BitVec16x16_AVX whichIsEqual(const BitVec16x16_AVX & other) const {
        return _mm256_cmpeq_epi16(this->m256, other.m256);
    }

    inline BitVec16x16_AVX whichIsNotEqual(const BitVec16x16_AVX & other) const {
        __m256i is_eq_mask = _mm256_cmpeq_epi16(this->m256, other.m256);
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        __m256i is_neq_mask = _mm256_andnot_si256(is_eq_mask, ones.m256);
        return is_neq_mask;
    }

    inline BitVec16x16_AVX whichIsMoreThan(const BitVec16x16_AVX & other) const {
        return _mm256_cmpgt_epi16(this->m256, other.m256);
    }

    inline BitVec16x16_AVX whichIsLessThan(const BitVec16x16_AVX & other) const {
        // Bad news: AVX2 have no _mm256_cmplt_epi16() ???
        __m256i is_eq_mask = _mm256_cmpeq_epi16(this->m256, other.m256);
        __m256i is_gt_mask = _mm256_cmpgt_epi16(this->m256, other.m256);
        __m256i is_ge_mask = _mm256_and_si256(is_eq_mask, is_gt_mask);
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        __m256i is_lt_mask = _mm256_andnot_si256(is_ge_mask, ones.m256);
        return is_lt_mask;
    }

    inline BitVec16x16_AVX whichIsZeros() const {
        BitVec16x16_AVX zeros;
        zeros.setAllZeros();
        return this->whichIsEqual(zeros);
    }

    inline BitVec16x16_AVX whichIsOnes() const {
        BitVec16x16_AVX ones;
        ones.setAllOnes();
        return this->whichIsEqual(ones);
    }

    inline BitVec16x16_AVX whichIsNonZero() const {
        BitVec16x16_AVX zeros;
        zeros.setAllZeros();
        return this->whichIsNotEqual(zeros);
    }

    inline BitVec16x16_AVX whichIsMoreThanZero() const {
        BitVec16x16_AVX zeros;
        zeros.setAllZeros();
        return this->whichIsMoreThan(zeros);
    }

    inline BitVec16x16_AVX whichIsLessThanZero() const {
        BitVec16x16_AVX zeros;
        zeros.setAllZeros();
        return this->whichIsLessThan(zeros);
    }

    inline BitVec16x16_AVX whichIsEqual16(uint16_t num) const {
#if 1
        //
        // _mm256_set1_epi16(int16_t) <=> _mm256_broadcastw_epi16(_mm_cvtsi32_si128(int));
        //
        __m256i num_mask = _mm256_set1_epi16((int16_t)num);
#else
        __m256i num_mask = _mm256_broadcastw_epi16(_mm_cvtsi32_si128((int)num));
#endif
        return this->whichIsEqual(num_mask);
    }

    inline BitVec16x16_AVX whichIsNotEqual16(uint16_t num) const {
        __m256i num_mask = _mm256_set1_epi16((int16_t)num);
        return this->whichIsNotEqual(num_mask);
    }

    template <bool isNonZeros>
    inline int indexOfIsEqual16(uint32_t num) const {
        BitVec16x16_AVX is_equal_mask = this->whichIsEqual16(num);
        return this->template firstIndexOfOnes16<isNonZeros>(is_equal_mask);
    }

    template <bool isNonZeros, bool isRepeat = true>
    inline int indexOfIsEqual16(const BitVec16x16_AVX & in_num_mask) const {
        BitVec16x16_AVX num_mask = in_num_mask;
        if (!isRepeat) {
            // If the num_mask is not a repeat mask,
            // we repeat the first 16bit integer first.
            num_mask = _mm256_broadcastw_epi16(_mm256_castsi256_si128(num_mask.m256));
        }
        BitVec16x16_AVX is_equal_mask = this->whichIsEqual(num_mask);
        return this->template firstIndexOfOnes16<isNonZeros>(is_equal_mask);
    }

    template <bool isRepeat = true>
    inline int maskOfIsEqual16(const BitVec16x16_AVX & in_num_mask) const {
        BitVec16x16_AVX num_mask = in_num_mask;
        if (!isRepeat) {
            // If the num_mask is not a repeat mask,
            // we repeat the first 16bit integer first.
            num_mask = _mm256_broadcastw_epi16(_mm256_castsi256_si128(num_mask.m256));
        }
        BitVec16x16_AVX is_equal_mask = this->whichIsEqual(num_mask);
        return this->maskOfOnes16(is_equal_mask);
    }

    template <bool isNonZeros>
    inline int firstIndexOfOnes16(const BitVec16x16_AVX & compare_mask) const {
        int compare_mask_32 = _mm256_movemask_epi8(compare_mask.m256);
        if (isNonZeros || (compare_mask_32 != 0)) {
            assert(!isNonZeros || (compare_mask_32 != 0));
            uint32_t first_offset = BitUtils::bsf(compare_mask_32);
            int first_index = (int)(first_offset >> 1U);
            return first_index;
        }
        else return -1;
    }

    inline uint32_t maskOfOnes16(const BitVec16x16_AVX & compare_mask) const {
#if 1
        uint32_t compare_mask_32 = (uint32_t)_mm256_movemask_epi8(compare_mask.m256);
        return compare_mask_32;
#else
        __m256i compare_mask_pack = _mm256_packs_epi16(compare_mask.m256, _mm256_setzero_si256());
        uint32_t compare_mask_32 = (uint32_t)_mm256_movemask_epi8(compare_mask_pack);
        uint32_t compare_mask_16 = (compare_mask_32 & 0x000000FFUL) | (compare_mask_32 >> 8U);
        return compare_mask_16;
#endif
    }

    inline uint16_t extract(int index) const {
#if !defined(_mm256_extract_epi16)
        #define CASE_LOW(x) case x: \
            low = _mm256_castsi256_si128(this->m256); \
            return (uint16_t)_mm_extract_epi16(low, x);
        #define CASE_HIGH(x) case x: \
            low = _mm256_extracti128_si256(this->m256, 1); \
            return (uint16_t)_mm_extract_epi16(low, (x - 8));
        __m128i low;
        switch (index) {
            CASE_LOW(0)
            CASE_LOW(1)
            CASE_LOW(2)
            CASE_LOW(3)
            CASE_LOW(4)
            CASE_LOW(5)
            CASE_LOW(6)
            CASE_LOW(7)
            CASE_HIGH(8)
            CASE_HIGH(9)
            CASE_HIGH(10)
            CASE_HIGH(11)
            CASE_HIGH(12)
            CASE_HIGH(13)
            CASE_HIGH(14)
            CASE_HIGH(15)
            default:
                assert(false);
                return uint16_t(-1);
        }
        #undef CASE_LOW
        #undef CASE_HIGH
#else
        #define CASE(x) case x: return (uint16_t)_mm256_extract_epi16(this->m256, x);
        switch (index) {
            CASE(0)
            CASE(1)
            CASE(2)
            CASE(3)
            CASE(4)
            CASE(5)
            CASE(6)
            CASE(7)
            CASE(8)
            CASE(9)
            CASE(10)
            CASE(11)
            CASE(12)
            CASE(13)
            CASE(14)
            CASE(15)
            default:
                assert(false);
                return uint16_t(-1);
        }
        #undef CASE
#endif // !_mm256_extract_epi16
    }

    inline void insert(int index, uint16_t value) {
#if !defined(_mm256_insert_epi16)
        #define CASE_LOW(x) case x: \
            low = _mm256_castsi256_si128(this->m256); \
            low = _mm_insert_epi16(low, (int)value, x); \
            break;
        #define CASE_HIGH(x) case x: \
            high = _mm256_extracti128_si256(this->m256, 1); \
            high = _mm_insert_epi16(high, value, (x - 8)); \
            this->m256 = _mm256_inserti128_si256(this->m256, high, 1); \
            break;
        __m128i low, high;
        switch (index) {
            CASE_LOW(0)
            CASE_LOW(1)
            CASE_LOW(2)
            CASE_LOW(3)
            CASE_LOW(4)
            CASE_LOW(5)
            CASE_LOW(6)
            CASE_LOW(7)
            CASE_HIGH(8)
            CASE_HIGH(9)
            CASE_HIGH(10)
            CASE_HIGH(11)
            CASE_HIGH(12)
            CASE_HIGH(13)
            CASE_HIGH(14)
            CASE_HIGH(15)
            default:
                assert(false);
                break;
        }
        #undef CASE_LOW
        #undef CASE_HIGH
#else
        #define CASE(x) case x: this->m256 = _mm256_insert_epi16(this->m256, value, x); break;
        switch (index) {
            CASE(0)
            CASE(1)
            CASE(2)
            CASE(3)
            CASE(4)
            CASE(5)
            CASE(6)
            CASE(7)
            CASE(8)
            CASE(9)
            CASE(10)
            CASE(11)
            CASE(12)
            CASE(13)
            CASE(14)
            CASE(15)
            default:
                assert(false);
                break;
        }
        #undef CASE
#endif // !_mm256_insert_epi16
    }

    template <size_t MaxLength, size_t MaxBits>
    BitVec16x16_AVX popcount16() const {
#if defined(__AVX512VPOPCNTW__) || (defined(__AVX512_BITALG__) && defined(__AVX512VL__))
        return _mm256_popcnt_epi16(this->m256);
#else
        if (MaxBits <= 8) {
            // Note: Ensure that the highest 8 bits must be 0.
            __m256i lookup     = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                                  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m256i mask4      = _mm256_set1_epi16(0x000F);
            __m256i sum_odd_4  = _mm256_shuffle_epi8(lookup, _mm256_and_si256(this->m256, mask4));
            __m256i sum_even_4 = _mm256_shuffle_epi8(lookup, _mm256_srli_epi16(this->m256, 4));
            __m256i sum_0_7    = _mm256_add_epi16(sum_odd_4, sum_even_4);
            __m256i result     = sum_0_7;
            return result;
        }
        else if (MaxBits == 9) {
            // Note: Ensure that the highest 7 bits must be 0.
            __m256i lookup     = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                                  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m256i mask4      = _mm256_set1_epi16(0x000F);
            __m256i sum_odd_4  = _mm256_shuffle_epi8(lookup, _mm256_and_si256(this->m256, mask4));
            __m256i sum_even_4 = _mm256_shuffle_epi8(lookup, _mm256_srli_epi16(this->m256, 4));
            __m256i sum_0_7    = _mm256_add_epi16(sum_odd_4, sum_even_4);
            __m256i result     = _mm256_add_epi16(sum_0_7, _mm256_srli_epi16(this->m256, 8));
            return result;
        }
        else {
            __m256i lookup     = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                                  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            __m256i mask4      = _mm256_set1_epi16(0x0F0F);
            __m256i sum_odd_4  = _mm256_shuffle_epi8(lookup, _mm256_and_si256(this->m256, mask4));
            __m256i even_bytes = _mm256_srli_epi16(this->m256, 4);
            __m256i sum_even_4 = _mm256_shuffle_epi8(lookup, _mm256_and_si256(even_bytes, mask4));
            __m256i sum_00_15  = _mm256_add_epi16(sum_odd_4, sum_even_4);
            __m256i mask8      = _mm256_set1_epi16(0x00FF);
            __m256i sum_odd_8  = _mm256_and_si256(sum_00_15, mask8);
            __m256i sum_even_8 = _mm256_srli_epi16(sum_00_15, 8);
            __m256i result     = _mm256_add_epi16(sum_odd_8, sum_even_8);
            return result;
        }
#endif // __AVX512__
    }

    template <size_t MaxLength>
    void min_i8(BitVec16x16_AVX & minpos) const {
        __m256i nums = this->m256;
        if (MaxLength <= 8) {
            nums = _mm256_min_epi8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));
            nums = _mm256_min_epi8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 8 x int8_t
            minpos = _mm256_min_epi8(nums, _mm256_srli_epi16(nums, 8));
        }
        else if (MaxLength <= 16) {
            nums = _mm256_min_epi8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epi8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epi8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 16 x int8_t
            minpos = _mm256_min_epi8(nums, _mm256_srli_epi16(nums, 8));
        }
        else {
            nums = _mm256_min_epi8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epi8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epi8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epi8(nums, _mm256_srli_epi16(nums, 8));

            // The minimum number of total 32 x int8_t
            minpos = _mm256_min_epi8(nums, _mm256_srli_si256(nums, 16));
        }
    }

    template <size_t MaxLength>
    int8_t min_i8(BitVec16x16_AVX & min_nums) const {
        this->min_i8<MaxLength>(min_nums);
#if 1 || (defined(_MSC_VER) && (_MSC_VER < 2000))
        BitVec08x16 min_nums_128;
        min_nums.castTo(min_nums_128);
        uint32_t min_num = SSE::mm_cvtsi128_si32_low(min_nums_128.m128);
        return (int8_t)(min_num & 0xFFUL);
#else
  #if 1
        uint32_t min_num = (uint32_t)_mm256_cvtsi256_si32(min_nums.m256);
        return (int8_t)(min_num & 0xFFUL);
  #else
        // AVX:  _mm256_extract_epi32() [Instruction: Sequence],
        // AVX2: _mm256_extract_epi16() [Instruction: Sequence]
        // AVX2: _mm256_extract_epi8()  [Instruction: Sequence]
        // AVX:  _mm256_cvtsi256_si32() [Instruction: vmovd r32, xmm]
        int min_num = _mm256_extract_epi8(min_nums.m256, 0);
        return (int8_t)min_num;
  #endif
#endif
    }

    template <size_t MaxLength>
    void min_u8(BitVec16x16_AVX & min_num) const {
        __m256i nums = this->m256;
        if (MaxLength <= 8) {
            nums = _mm256_min_epu8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));
            nums = _mm256_min_epu8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 8 x uint8_t
            min_num = _mm256_min_epu8(nums, _mm256_srli_epi16(nums, 8));
        }
        else if (MaxLength <= 16) {
            nums = _mm256_min_epu8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epu8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epu8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 16 x uint8_t
            min_num = _mm256_min_epu8(nums, _mm256_srli_epi16(nums, 8));
        }
        else {
            nums = _mm256_min_epu8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epu8(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epu8(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epu8(nums, _mm256_srli_epi16(nums, 8));

            // The minimum number of total 32 x uint8_t
            min_num = _mm256_min_epu8(nums, _mm256_srli_si256(nums, 16));
        }
    }

    template <size_t MaxLength>
    inline uint8_t min_u8(BitVec16x16_AVX & min_nums) const {
        this->min_u8<MaxLength>(min_nums);
#if 1 || (defined(_MSC_VER) && (_MSC_VER < 2000))
        BitVec08x16 min_nums_128;
        min_nums.castTo(min_nums_128);
        uint32_t min_num = SSE::mm_cvtsi128_si32_low(min_nums_128.m128);
        return (uint8_t)(min_num & 0xFFUL);
#else
  #if 1
        uint32_t min_num = (uint32_t)_mm256_cvtsi256_si32(min_nums.m256);
        return (uint8_t)(min_num & 0xFFUL);
  #else
        // AVX:  _mm256_extract_epi32() [Instruction: Sequence],
        // AVX2: _mm256_extract_epi16() [Instruction: Sequence]
        // AVX2: _mm256_extract_epi8()  [Instruction: Sequence]
        // AVX:  _mm256_cvtsi256_si32() [Instruction: vmovd r32, xmm]
        int min_num = _mm256_extract_epi8(min_nums.m256, 0);
        return (uint8_t)min_num;
  #endif
#endif
    }

    template <size_t MaxLength>
    inline uint32_t minpos16() const {
        BitVec08x16 minpos;
        __m256i nums = this->m256;
        __m256i minnum_u16;
        if (MaxLength <= 4) {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));

            // The minimum number of first 4 x int16_t
            minnum_u16 = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else if (MaxLength <= 8) {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 8 x int16_t
            minnum_u16 = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of total 16 x int16_t
            __m256i nums_high = _mm256_permute4x64_epi64(nums, _MM_SHUFFLE(1, 0, 3, 2));
            minnum_u16 = _mm256_min_epu16(nums, nums_high);
        }

        // Get the minimum number
        uint32_t min_num = (uint32_t)_mm256_cvtsi256_si32(minnum_u16) & 0xFFFFUL;
        // Get the index of minimum number
        uint32_t min_index = (uint32_t)this->template indexOfIsEqual16<true>(min_num);
        assert(min_index != -1);

        uint32_t min_and_index = min_num | (min_index << 16);
        return min_and_index;
    }

    template <size_t MaxLength>
    inline void minpos16(BitVec08x16 & minpos) const {
#if 1
        uint32_t min_and_index = this->minpos16<MaxLength>();
        minpos = _mm_cvtsi32_si128(min_and_index);
#else
        BitVec08x16 minpos;
        BitVec16x16_AVX minpos256;
        __m256i nums = this->m256;
        __m256i minnum_u16;
        if (MaxLength <= 4) {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(2, 3, 0, 1)));

            // The minimum number of first 4 x int16_t
            minnum_u16 = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else if (MaxLength <= 8) {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 0, 3, 2)));
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of first 8 x int16_t
            minnum_u16 = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));
        }
        else {
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(3, 2, 3, 2)));
            nums = _mm256_min_epu16(nums, _mm256_shuffle_epi32(nums, _MM_SHUFFLE(1, 1, 1, 1)));
            nums = _mm256_min_epu16(nums, _mm256_shufflelo_epi16(nums, _MM_SHUFFLE(1, 1, 1, 1)));

            // The minimum number of total 16 x int16_t
            //__m256i nums_high = _mm256_permute4x64_epi64(nums, _MM_SHUFFLE(1, 0, 3, 2));
            //
            // Use _mm256_permute2x128_si256() can reset high 128 bits to 0,
            // but low bits[32:127] is not reset to 0 yet.
            //
            __m256i nums_high = _mm256_permute2x128_si256(nums, nums, 0b10000001);
            minnum_u16 = _mm256_min_epu16(nums, nums_high);
        }

        // Get the index of minimum number
        uint32_t min_index = (uint32_t)this->template indexOfIsEqual16<true, false>(minnum_u16);
        assert(min_index != -1);

        __m256i minpos256 = _mm256_castsi128_si256(minpos.m128);
        minpos256 = _mm256_insert_epi16(minnum_u16, min_index, 1);
        minpos = _mm256_castsi256_si128(minpos256);
#endif
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(uint32_t & old_min_num, int & min_index) const {
        uint32_t min_and_index = this->minpos16<MaxLength>();
        uint32_t min_num = min_and_index & 0xFFFFUL;
        if (min_num < old_min_num) {
            old_min_num = min_num;
            min_index = (int)(min_and_index >> 16U);
        }
        return min_num;
    }

    template <size_t MaxLength>
    inline uint32_t minpos16(int & min_index) const {
        uint32_t min_and_index = this->minpos16<MaxLength>();
        uint32_t min_num = min_and_index & 0xFFFFUL;
        min_index = (int)(min_and_index >> 16U);
        return min_num;
    }
};

#else

typedef BitVec16x16_AVX BitVec16x16;

#endif // __AVX2__

} // namespace gzSudoku

#endif // GZ_SUDOKU_BITVEC_H
