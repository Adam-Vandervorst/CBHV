#ifndef BHV_CORE_H
#define BHV_CORE_H

#ifndef NOPARALLELISM
#ifdef _OPENMP
#include <omp.h>
#else
#include <execution>
#endif
#endif
#include <bit>
#include <functional>
#include <queue>
#include <ranges>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "immintrin.h"
#include "shared.h"
#include "pq.h"
#ifdef __AVX2__
#include "simdpcg.h"
#endif
#ifdef __AVX512__
#include "TurboSHAKE_AVX512/TurboSHAKE.h"
#else
#include "TurboSHAKE_opt/TurboSHAKE.h"
#endif


namespace bhv {
    constexpr word_t ONE_WORD = 0xffffffffffffffff;
    constexpr word_t EVEN_WORD = 0x5555555555555555;
    constexpr word_t ODD_WORD = 0xaaaaaaaaaaaaaaaa;

    template<word_t W>
    word_t *const_bhv() {
        static word_t x[WORDS];
        for (size_t i = 0; i < WORDS; ++i) x[i] = W;
        return x;
    }

    static word_t *ZERO = const_bhv<0>();
    static word_t *ONE = const_bhv<ONE_WORD>();
    static word_t *EVEN = const_bhv<EVEN_WORD>();
    static word_t *ODD = const_bhv<ODD_WORD>();

    std::mt19937_64 rng;

    inline word_t *empty() {
        return (word_t *) aligned_alloc(64, BYTES);
    }

    word_t *zero() {
        word_t * e = empty();
        memset(e, 0, BYTES);
        return e;
    }

    word_t *one() {
        word_t *x = empty();
        memset(x, 0xff, BYTES);
        return x;
    }

    bool eq(word_t *x, word_t *y) {
        for (word_iter_t i = 0; i < WORDS; ++i)
            if (x[i] != y[i])
                return false;
        return true;
    }

    void xor_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i)
            target[i] = x[i] ^ y[i];
    }

    void and_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i)
            target[i] = x[i] & y[i];
    }

    void or_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i)
            target[i] = x[i] | y[i];
    }

    void invert_into(word_t *x, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i)
            target[i] = ~x[i];
    }

    void unpack_into(word_t *x, bool *target_bits) {
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = x[word_id];
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                target_bits[offset + bit_id] = (word >> bit_id) & 1;
            }
        }
    }

    void pack_into(bool *bits, word_t *target) {
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (bits[offset + bit_id])
                    word |= 1ULL << bit_id;
            }
            target[word_id] = word;
        }
    }

    inline bool get(word_t *x, bit_iter_t i) {
        return (x[i/BITS_PER_WORD] >> (i % BITS_PER_WORD)) & 1;
    }

    inline void toggle(word_t *x, bit_iter_t i) {
        x[i/BITS_PER_WORD] ^= (1ULL << (i % BITS_PER_WORD));
    }

    inline void level_into(size_t l, word_t *target) {
        uint8_t *target_bytes = (uint8_t*)target;

        size_t needed = (l + 7) / 8;
        size_t full = l / 8;
        size_t start = full + (full < needed);

        memset(target_bytes, 0xFF, full);
        if (full < needed) target_bytes[full] = (1 << (l % 8)) - 1;
        if (start < BYTES) memset(target_bytes + start, 0x00, BYTES - start);
    }

    #include "io.h"

    #include "ternary.h"

    #include "distance.h"

    #include "random.h"

    #include "parity.h"

    #include "threshold.h"

    #include "majority.h"

    #include "weighted_threshold.h"

    #include "representative.h"

    #include "window.h"

    #include "permutation.h"

    #include "hash.h"

    #include "lookup.h"

    #include "optimization.h"
}
#endif //BHV_CORE_H
