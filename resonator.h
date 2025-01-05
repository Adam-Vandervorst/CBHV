#include <tuple>
#include <iostream>
#include "shared.h"


void lift_into(word_t *x, int8_t* target_bits) {
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        bit_iter_t offset = word_id * BITS_PER_WORD;
        word_t word = x[word_id];
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if ((word >> bit_id) & 1) target_bits[offset + bit_id] = -1;
            else target_bits[offset + bit_id] = 1;
        }
    }
}

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

void dropw_into(int32_t* bits, word_t *target) {
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

void automat(word_t **codebook, size_t n, int8_t *out) {
    int8_t *ins = (int8_t *)malloc(BITS*n);
    for (auto k = 0; k < n; ++k)
        lift_into(codebook[k], ins + k*BITS);

    for (size_t i = 0; i < BITS; ++i) {
        for (size_t j = 0; j < BITS; ++j) {
            for (size_t k = 0; k < n; ++k) {
//                out[i*BITS + j] += (get(codebook[k], i) == get(codebook[k], j)) ? 1 : -1;
                out[i*BITS + j] = std::add_sat(out[i*BITS + j], std::mul_sat(ins[k*BITS + i], ins[k*BITS + j]));
            }
        }
    }

    free(ins);
}

void matvec(int8_t *m, int8_t *v, int8_t *out) {
    for (size_t i = 0; i < BITS; ++i) {
        for (size_t j = 0; j < BITS; ++j) {
            out[i] = std::add_sat(out[i], std::mul_sat(m[i*BITS + j], v[j]));
        }
    }
}

void matvecw(int8_t *m, int8_t *v, int32_t *out) {
    for (size_t i = 0; i < BITS; ++i) {
        for (size_t j = 0; j < BITS; ++j) {
            out[i] = std::add_sat(out[i], (int32_t)std::mul_sat((int16_t)m[i*BITS + j], (int16_t)v[j]));
        }
    }
}

struct Resonator5 {
    size_t n0, n1, n2, n3, n4;
    word_t **codebook0; word_t **codebook1; word_t **codebook2; word_t **codebook3; word_t **codebook4;
    word_t* i0; word_t* i1; word_t* i2; word_t* i3; word_t* i4;
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

        i0 = (word_t *) aligned_alloc(64, 5*BYTES);
        i1 = i0 + WORDS;
        i2 = i0 + 2*WORDS;
        i3 = i0 + 3*WORDS;
        i4 = i0 + 4*WORDS;

        word_t r [WORDS] = {};
        word_t rs [WORDS] = {};

        true_majority_into(codebook0, n0, i0);
        true_majority_into(codebook1, n1, i1);
        true_majority_into(codebook2, n2, i2);
        true_majority_into(codebook3, n3, i3);
        true_majority_into(codebook4, n4, i4);
    }

    std::array<size_t, 5> run(word_t *const s, size_t iter) const {
        word_t r [WORDS] = {}; word_t x [5*WORDS] = {}; word_t nx [5*WORDS] = {};
        memcpy(x, i0, 5*BYTES);
        word_t *x0 = x + 0*WORDS; word_t *x1 = x + 1*WORDS; word_t *x2 = x + 2*WORDS; word_t *x3 = x + 3*WORDS; word_t *x4 = x + 4*WORDS;
        word_t *nx0 = nx + 0*WORDS; word_t *nx1 = nx + 1*WORDS; word_t *nx2 = nx + 2*WORDS; word_t *nx3 = nx + 3*WORDS; word_t *nx4 = nx + 4*WORDS;
//        std::cout << active(x0) << "," << active(x1) << "," << active(x2) << "," << active(x3) << "," << active(x4) << std::endl;

        for (auto i = 0; i < iter; ++i) {
            xor_into(s, x1, nx0); xor_into(nx0, x2, nx0); xor_into(nx0, x3, nx0); xor_into(nx0, x4, nx0);
            int8_t f0 [BITS] = {}; lift_into(nx0, f0); int32_t z0 [BITS] = {}; matvecw(auto0, f0, z0); dropw_into(z0, nx0);
            rand2_into(r, -2); select_into(r, x0, nx0, nx0);
            xor_into(s, x0, nx1); xor_into(nx1, x2, nx1); xor_into(nx1, x3, nx1); xor_into(nx1, x4, nx1);
            int8_t f1 [BITS] = {}; lift_into(nx1, f1); int32_t z1 [BITS] = {}; matvecw(auto1, f1, z1); dropw_into(z1, nx1);
            rand2_into(r, -2); select_into(r, x1, nx1, nx1);
            xor_into(s, x0, nx2); xor_into(nx2, x1, nx2); xor_into(nx2, x3, nx2); xor_into(nx2, x4, nx2);
            int8_t f2 [BITS] = {}; lift_into(nx2, f2); int32_t z2 [BITS] = {}; matvecw(auto2, f2, z2); dropw_into(z2, nx2);
            rand2_into(r, -2); select_into(r, x2, nx2, nx2);
            xor_into(s, x0, nx3); xor_into(nx3, x1, nx3); xor_into(nx3, x2, nx3); xor_into(nx3, x4, nx3);
            int8_t f3 [BITS] = {}; lift_into(nx3, f3); int32_t z3 [BITS] = {}; matvecw(auto3, f3, z3); dropw_into(z3, nx3);
            rand2_into(r, -2); select_into(r, x3, nx3, nx3);
            xor_into(s, x0, nx4); xor_into(nx4, x1, nx4); xor_into(nx4, x2, nx4); xor_into(nx4, x3, nx4);
            int8_t f4 [BITS] = {}; lift_into(nx4, f4); int32_t z4 [BITS] = {}; matvecw(auto4, f4, z4); dropw_into(z4, nx4);
            rand2_into(r, -2); select_into(r, x4, nx4, nx4);

            bool converged = true;
            converged &= hamming(x0, nx0) < sqrt(2*BITS);
            converged &= hamming(x1, nx1) < sqrt(2*BITS);
            converged &= hamming(x2, nx2) < sqrt(2*BITS);
            converged &= hamming(x3, nx3) < sqrt(2*BITS);
            converged &= hamming(x4, nx4) < sqrt(2*BITS);
//            std::cout << active(x0) << "," << active(x1) << "," << active(x2) << "," << active(x3) << "," << active(x4) << std::endl;
            if (converged) {
                std::cout << "converged" << std::endl;
                break;
            }
            word_t *swap0 = x0; x0 = nx0; nx0 = swap0;
            word_t *swap1 = x1; x1 = nx1; nx1 = swap1;
            word_t *swap2 = x2; x2 = nx2; nx2 = swap2;
            word_t *swap3 = x3; x3 = nx3; nx3 = swap3;
            word_t *swap4 = x4; x4 = nx4; nx4 = swap4;
        }

        return {closest(codebook0, n0, x0), closest(codebook1, n1, x1), closest(codebook2, n2, x2), closest(codebook3, n3, x3), closest(codebook4, n4, x4)};
    }

    ~Resonator5() {
        free(auto0); free(auto1); free(auto2); free(auto3); free(auto4); free(i0);
    }
};

