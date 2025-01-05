#include <tuple>
#include <iostream>
#include "shared.h"


void drop_into(int8_t* bits, word_t *target) {
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        bit_iter_t offset = word_id * BITS_PER_WORD;
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if (bits[offset + bit_id] < 0)
                word |= 1ULL << bit_id;
        }
        target[word_id] = word;
    }
}

void invert_fuzzy_into(word_t *mask, int8_t *src, int8_t *dst) {
    for (size_t i = 0; i < BITS; ++i) {
        dst[i] = get(mask, i) ? -src[i] : src[i];
    }
}

void interpolate_into(int8_t* left, int8_t* right, int8_t *dst, float p) {
    for (size_t i = 0; i < BITS; ++i) {
        dst[i] = (int8_t)(p*(float)left[i] + (1.f - p)*(float)right[i]);
    }
}

void sum_into(word_t **codebook, size_t n, int8_t *dst) {
    for (size_t k = 0; k < n; ++k) {
        unpack_update_into<2, int8_t>(codebook[k], dst, 127/n);
    }
}

template <uint8_t n>
int8_t fuzzy_parity(int8_t *fs) {
    int32_t t = 0;
    for_sequence<(1 << n)>([&fs, &t](auto i) {
        if constexpr (std::popcount((uint32_t)i) % 2 != 0) {
        int32_t p = 1;
        for (uint8_t k = 0; k < n; ++k) {
            if (fs[k]) p *= (i & (1 << k)) ? fs[k] : -fs[k];
        }
        t += p;
    }
    });

    return std::add_sat((int8_t)(t/(1 << 8*(n - 1))), (int8_t)copysign(1, t));
}

void fuzzy_parity4_into(int8_t *f0, int8_t *f1, int8_t *f2, int8_t *f3, int8_t *out) {
    for (size_t i = 0; i < BITS; ++i) {
        int8_t fs[4] = {f0[i], f1[i], f2[i], f3[i]};
        out[i] = fuzzy_parity<4>(fs);
    }
}

void automat(word_t **codebook, size_t n, int8_t *out) {
    assert(n <= 8);
    uint8_t ins[BITS] = {};
    for (auto k = 0; k < n; ++k)
        unpack_update_into<1, uint8_t>(codebook[k], ins, k);

    for (size_t i = 0; i < BITS; ++i) {
        for (size_t j = 0; j < BITS; ++j) {
            out[i*BITS + j] = (int8_t)n - 2*(int8_t)std::popcount<uint8_t>(ins[i] ^ ins[j]);
//            for (size_t k = 0; k < n; ++k) {
//                out[i*BITS + j] += (get(codebook[k], i) == get(codebook[k], j)) ? 1 : -1;
//            }
        }
    }
}

void matvecw(int8_t *__restrict m, int8_t *__restrict v, int8_t *__restrict out) {
    for (size_t i = 0; i < BITS; ++i) {
        float s = 0.;
        for (size_t j = 0; j < BITS; ++j) {
            s += (float)m[i*BITS + j]*(float)v[j];
        }
        out[i] = std::add_sat(std::saturate_cast<int8_t>((int32_t)(s/(float)256)), (int8_t)copysign((int8_t)1, s));
    }
}

struct Resonator5 {
    size_t n0, n1, n2, n3, n4;
    word_t **codebook0; word_t **codebook1; word_t **codebook2; word_t **codebook3; word_t **codebook4;
// pre-calculate sums of codebooks?
//    word_t* i0; word_t* i1; word_t* i2; word_t* i3; word_t* i4;
    int8_t *auto0; int8_t *auto1; int8_t *auto2; int8_t *auto3; int8_t *auto4;

    Resonator5(word_t **codebook0, size_t n0, word_t **codebook1, size_t n1, word_t **codebook2, size_t n2, word_t **codebook3, size_t n3, word_t **codebook4, size_t n4) :
               codebook0(codebook0), codebook1(codebook1), codebook2(codebook2), codebook3(codebook3), codebook4(codebook4),
               n0(n0), n1(n1), n2(n2), n3(n3), n4(n4) {
        auto0 = (int8_t *)malloc(BITS*BITS); memset(auto0, 0, BITS*BITS);
        automat(codebook0, n0, auto0);
        auto1 = (int8_t *)malloc(BITS*BITS); memset(auto1, 0, BITS*BITS);
        automat(codebook1, n1, auto1);
        auto2 = (int8_t *)malloc(BITS*BITS); memset(auto2, 0, BITS*BITS);
        automat(codebook2, n2, auto2);
        auto3 = (int8_t *)malloc(BITS*BITS); memset(auto3, 0, BITS*BITS);
        automat(codebook3, n3, auto3);
        auto4 = (int8_t *)malloc(BITS*BITS); memset(auto4, 0, BITS*BITS);
        automat(codebook4, n4, auto4);
    }

    std::array<size_t, 5> run_fuzzy(word_t *const s, size_t iter) const {
        int8_t f [5*BITS] = {}; int8_t nf [5*BITS] = {};

        int8_t *f0 = f + 0*BITS; int8_t *f1 = f + 1*BITS; int8_t *f2 = f + 2*BITS; int8_t *f3 = f + 3*BITS; int8_t *f4 = f + 4*BITS;
        int8_t *nf0 = nf + 0*BITS; int8_t *nf1 = nf + 1*BITS; int8_t *nf2 = nf + 2*BITS; int8_t *nf3 = nf + 3*BITS; int8_t *nf4 = nf + 4*BITS;
        word_t x [5*WORDS] = {};
        word_t *x0 = x + 0*WORDS; word_t *x1 = x + 1*WORDS; word_t *x2 = x + 2*WORDS; word_t *x3 = x + 3*WORDS; word_t *x4 = x + 4*WORDS;

//        invert_fuzzy_into(i0, f0, f0); invert_fuzzy_into(i1, f1, f1); invert_fuzzy_into(i2, f2, f2); invert_fuzzy_into(i3, f3, f3); invert_fuzzy_into(i4, f4, f4);
        sum_into(codebook0, n0, f0);
        sum_into(codebook1, n1, f1);
        sum_into(codebook2, n2, f2);
        sum_into(codebook3, n3, f3);
        sum_into(codebook4, n4, f4);

        for (size_t i = 0; i < iter; ++i) {
//            #pragma omp parallel
//            #pragma omp single nowait
            {
//            #pragma omp task
            {
                fuzzy_parity4_into(f1, f2, f3, f4, nf0);
                invert_fuzzy_into(s, nf0, nf0);
                int8_t z0 [BITS];
                matvecw(auto0, nf0, z0);
                invert_fuzzy_into(ONE, z0, nf0);
            }
//            #pragma omp task
            {
                fuzzy_parity4_into(f0, f2, f3, f4, nf1);
                invert_fuzzy_into(s, nf1, nf1);
                int8_t z1[BITS] = {};
                matvecw(auto1, nf1, z1);
                invert_fuzzy_into(ONE, z1, nf1);
            }
//            #pragma omp task
            {
                fuzzy_parity4_into(f0, f1, f3, f4, nf2);
                invert_fuzzy_into(s, nf2, nf2);
                int8_t z2[BITS] = {};
                matvecw(auto2, nf2, z2);
                invert_fuzzy_into(ONE, z2, nf2);
            }
//            #pragma omp task
            {
                fuzzy_parity4_into(f0, f1, f2, f4, nf3);
                invert_fuzzy_into(s, nf3, nf3);
                int8_t z3[BITS] = {};
                matvecw(auto3, nf3, z3);
                invert_fuzzy_into(ONE, z3, nf3);
            }
//            #pragma omp task
            {
                fuzzy_parity4_into(f0, f1, f2, f3, nf4);
                invert_fuzzy_into(s, nf4, nf4);
                int8_t z4[BITS] = {};
                matvecw(auto4, nf4, z4);
                invert_fuzzy_into(ONE, z4, nf4);
            }
            }

            float interpolation = .01;
            interpolate_into(f0, nf0, nf0, interpolation);
            interpolate_into(f1, nf1, nf1, interpolation);
            interpolate_into(f2, nf2, nf2, interpolation);
            interpolate_into(f3, nf3, nf3, interpolation);
            interpolate_into(f4, nf4, nf4, interpolation);

            int8_t *swap0 = f0; f0 = nf0; nf0 = swap0;
            int8_t *swap1 = f1; f1 = nf1; nf1 = swap1;
            int8_t *swap2 = f2; f2 = nf2; nf2 = swap2;
            int8_t *swap3 = f3; f3 = nf3; nf3 = swap3;
            int8_t *swap4 = f4; f4 = nf4; nf4 = swap4;

            drop_into(f0, x0); drop_into(f1, x1); drop_into(f2, x2); drop_into(f3, x3); drop_into(f4, x4);
//            std::cout << active(x0) << "," << active(x1) << "," << active(x2) << "," << active(x3) << "," << active(x4) << std::endl;
//            for (int k = 0; k < n0; ++k) std::cout << hamming(x0, codebook0[k]) << ","; std::cout << "  ";
//            for (int k = 0; k < n1; ++k) std::cout << hamming(x1, codebook1[k]) << ","; std::cout << "  ";
//            for (int k = 0; k < n2; ++k) std::cout << hamming(x2, codebook2[k]) << ","; std::cout << "  ";
//            for (int k = 0; k < n3; ++k) std::cout << hamming(x3, codebook3[k]) << ","; std::cout << "  ";
//            for (int k = 0; k < n4; ++k) std::cout << hamming(x4, codebook4[k]) << ","; std::cout << "  ";
//            std::cout << std::endl;

            bool converged = true;
            converged &= hamming(x0, codebook0[closest(codebook0, n0, x0)]) < sqrt(2*BITS);
            converged &= hamming(x1, codebook1[closest(codebook1, n1, x1)]) < sqrt(2*BITS);
            converged &= hamming(x2, codebook2[closest(codebook2, n2, x2)]) < sqrt(2*BITS);
            converged &= hamming(x3, codebook3[closest(codebook3, n3, x3)]) < sqrt(2*BITS);
            converged &= hamming(x4, codebook4[closest(codebook4, n4, x4)]) < sqrt(2*BITS);
//            std::cout << active(x0) << "," << active(x1) << "," << active(x2) << "," << active(x3) << "," << active(x4) << std::endl;
            if (converged) {
                std::cout << "converged after " << i << " iterations" << std::endl;
                break;
            }
        }

        drop_into(f0, x0); drop_into(f1, x1); drop_into(f2, x2); drop_into(f3, x3); drop_into(f4, x4);

        return {closest(codebook0, n0, x0), closest(codebook1, n1, x1), closest(codebook2, n2, x2), closest(codebook3, n3, x3), closest(codebook4, n4, x4)};
    }

    ~Resonator5() {
        free(auto0); free(auto1); free(auto2); free(auto3); free(auto4);
//        free(i0);
    }
};

