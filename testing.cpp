#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <bitset>
#include <functional>
#include "core.h"

using namespace std;


void print_bits(int64_t w) {
    bitset<64> x(w);
    cout << x << endl;
}


void print_byte(int8_t w) {
    bitset<8> x(w);
    cout << x << endl;
}


void test_single_byte_permutation() {
    mt19937_64 rng;

    uint64_t r = rng();

    uint8_t r0 = 0b11110001;

    uint64_t id = (uint64_t) _mm_set_pi8(7, 6, 5, 4, 3, 2, 1, 0);

    uint64_t p1 = (uint64_t) _mm_set_pi8(7, 5, 3, 1, 6, 4, 2, 0);
    uint64_t p1_inv = (uint64_t) _mm_set_pi8(7, 2, 6, 2, 5, 1, 4, 0);

    uint64_t p2 = (uint64_t) _mm_set_pi8(7, 6, 5, 4, 0, 1, 2, 3);
    uint64_t p2_inv = p2;

    uint64_t rp1 = bhv::rand_byte_bits_permutation(5);
    uint64_t rp1_inv = bhv::byte_bits_permutation_invert(rp1);


    print_bits(rp1);
    print_bits(rp1_inv);

    print_byte(r0);

    uint8_t res = bhv::permute_single_byte_bits(r0, rp1);
    print_byte(res);

    uint8_t res_ = bhv::permute_single_byte_bits(res, rp1_inv);
    print_byte(res_);
}

void test_instruction_upto() {
    uint8_t to = 0;
    float_t remaining = 0;
//    uint64_t instruction = bhv::instruction_upto(.0000001, &to, &remaining);  //
    uint64_t instruction = bhv::instruction_upto(.5, &to, &remaining);  //
//    uint64_t instruction = bhv::instruction_upto(.625, &to, &remaining); // 10
//    uint64_t instruction = bhv::instruction_upto(.375, &to, &remaining);  // 01
//    uint64_t instruction = bhv::instruction_upto(.9, &to, &remaining, .005);  // 1110
//    uint64_t instruction = bhv::instruction_upto(.5625, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(123.f/256.f, &to, &remaining);  // 0111101
//    uint64_t instruction = bhv::instruction_upto(.5625001, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(.5624999, &to, &remaining);  // 100

    cout << "to: " << (uint32_t) to << endl;
    cout << "rem: " << remaining << endl;
    print_bits(instruction);

    for (uint8_t i = to - 1; i < to; --i)
        cout << ((instruction & (1 << i)) >> i);

    cout << endl;

    word_t *x = bhv::empty();
    bhv::random_into_tree_sparse_avx2(x, .5);
    cout << "active: " << (double) bhv::active(x) / (double) BITS << endl;
}

void test_ternary_instruction() {
    uint8_t buffer [24];
    uint8_t to = 0;
//    int8_t finalizer = bhv::ternary_instruction(.0000001, buffer, &to, 1e-3);
//    int8_t finalizer = bhv::ternary_instruction(.5, buffer, &to);
//    buffer: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 0
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.25, buffer, &to);
//    buffer: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 0
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.625, buffer, &to);
//    buffer: 11111000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 1
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.375, buffer, &to, 1e-3);
//    buffer: 11100000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 1
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.9, buffer, &to, .005);  // 1110
//    buffer: 11111110 11111000 11100000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 3
//    finalizer: -1

//    uint64_t instruction = bhv::instruction_upto(.5625, &to, &remaining);  // 100

    int8_t finalizer = bhv::ternary_instruction(123.f/256.f, buffer, &to);  // 0111101
//    buffer: 11100000 11111110 11111000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 3
//    finalizer: 1

//    uint64_t instruction = bhv::instruction_upto(.5625001, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(.5624999, &to, &remaining);  // 100

    cout << "buffer: ";
    for (uint8_t b : buffer)
        cout << bitset<8>(b) << " ";
    cout << endl;
    cout << "to: " << (uint32_t) to << endl;
    cout << "finalizer: " << (int32_t) finalizer << endl;

//    word_t *x = bhv::empty();
//    bhv::random_into_ternary_tree_avx512(x, .5);
//    cout << "active: " << (double) bhv::active(x) / (double) BITS << endl;

//    bhv::random_into_ternary_tree_avx512(x, 123.f/256.f);
//    cout << "expected: " << 123.f/256.f << ", active: " << (double) bhv::active(x) / (double) BITS << endl;
}

void test_independent_opt() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    cout << bhv::hamming(v_ref, v) << " |v_ref,initial|" << endl;

    auto close_to_ref = [v_ref](word_t *x) { return bhv::hamming(x, v_ref); };
    bhv::opt::independent<uint64_t>(v, close_to_ref);

    cout << bhv::hamming(v_ref, v) << " |v_ref,final|" << endl;

    auto full = bhv::one();

    auto half = [](word_t *x) { return abs(int64_t(bhv::active(x) - BITS/2)); };
    bhv::opt::independent<uint64_t>(full, half);

    cout << bhv::active(full) << " |full_opt|  (" << BITS/2 << ")" << endl;
}

void test_easy_search() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    cout << bhv::hamming(v_ref, v) << " |v_ref,initial|" << endl;

    auto close_to_ref = [v_ref](word_t *x) { return bhv::hamming(x, v_ref); };
    bhv::opt::linear_search<uint64_t>(v, close_to_ref);

    cout << bhv::hamming(v_ref, v) << " |v_ref,final|" << endl;
}

void test_harder_search() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    auto close_to_ref = [v_ref](word_t *x) { word_t tmp [WORDS]; bhv::roll_word_bits_into(x, 12, tmp); bhv::xor_into(x, tmp, tmp); return bhv::hamming(tmp, v_ref); };
    auto t1 = chrono::high_resolution_clock::now();

    bhv::opt::linear_search<uint64_t>(v, close_to_ref, 100000000, 200*BITS);
//    bhv::opt::evolutionary_search<uint64_t, 2048*16, 512>(v, close_to_ref, 100);
//    bhv::opt::evolutionary_search<uint64_t, 16, 3>(v, close_to_ref, 20);

    auto t2 = chrono::high_resolution_clock::now();

    cout << close_to_ref(v) << " loss" << endl;
    double test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) (1000000000);
    cout << test_time << " seconds" << endl;
}

void test_opt_centroid() {
    // while the min_sum vector is unique and equals the MAJ
    // there is a family of vectors min_diff vectors unrelated to each other
    auto v = bhv::rand();
    auto x = bhv::rand();
    auto y = bhv::rand();
    auto z = bhv::rand();
    word_t m [WORDS];
    bhv::majority3_into(x, y, z, m);

    auto min_sum = [x, y, z](word_t *a) { auto dx = bhv::hamming(a, x); auto dy = bhv::hamming(a, y); auto dz = bhv::hamming(a, z); return dx + dy + dz; };
//    auto min_diff = [x, y, z](word_t *a) { int64_t dx = bhv::hamming(a, x); int64_t dy = bhv::hamming(a, y); int64_t dz = bhv::hamming(a, z); return abs(dx - dy) + abs(dy - dz) + abs(dx - dz); };

    auto t1 = chrono::high_resolution_clock::now();
    bhv::opt::linear_search<uint64_t>(v, min_sum, 1000000);
//    bhv::opt::evolutionary_search<uint64_t, 16, 4>(v, min_sum, 5);
    auto t2 = chrono::high_resolution_clock::now();

    cout << min_sum(v) << " loss" << endl;
    cout << bhv::hamming(v, m) << " |maj(x,y,z),final|" << endl;
    cout << bhv::hamming(v, x) << " |x,final|, " << bhv::hamming(v, y) << " |y,final|, " << bhv::hamming(v, z) << " |z,final|" << endl;
    cout << bhv::hash(v) << " h" << endl;
    double test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) (1000000000);
    cout << test_time << " seconds" << endl;
}

void test_active_equal() {
    for (size_t i = 0; i <= DIMENSION; ++i) {
        word_t v [WORDS];
        bhv::level_into(i, v);
        assert(bhv::active(v) == i);

        bool f = false;
        for (size_t j = 0; j < DIMENSION; ++j) {
            if (bhv::get(v, j) != (j < i)) {
                cout << j << "," << endl;
                f = true;
            }
        }
        if (f)
            cout << endl << "at " << i << endl;
    }

}

void test_swap_even_odd() {
    uint64_t v1[DIMENSION/64];
    uint64_t v1_swapped[DIMENSION/64];
    uint64_t v1_swapped_twice[DIMENSION/64];

    uint64_t v1_swapped_gfni[DIMENSION/64];
    uint64_t v1_swapped_twice_gfni[DIMENSION/64];

    bhv::rand_into(v1);

    bhv::swap_even_odd_into(v1, v1_swapped);
    bhv::swap_even_odd_into(v1_swapped, v1_swapped_twice);

    // bhv::swap_even_odd_into_avx512(v1, v1_swapped_gfni);
    // bhv::swap_even_odd_into_avx512(v1_swapped_gfni, v1_swapped_twice_gfni);

    std::cout << bhv::to_string(v1) << std::endl;
    std::cout << bhv::to_string(v1_swapped) << std::endl;
    // std::cout << bhv::to_string(v1_swapped_gfni) << std::endl;
    std::cout << bhv::to_string(v1_swapped_twice) << std::endl;
    // std::cout << bhv::to_string(v1_swapped_twice_gfni) << std::endl;
    std::cout << bhv::eq(v1, v1_swapped_twice) << std::endl;
    // std::cout << bhv::eq(v1, v1_swapped_twice_gfni) << std::endl;
}

void test_hadamard() {
    word_t *hvs[BITS];
    word_t tmp [WORDS];

    for (uint32_t i = 0; i < BITS; ++i) {
        hvs[i] = bhv::empty();
        bhv::hadamard_encode_into_basis(i, tmp);
        bhv::hadamard_encode_into_reference(i, hvs[i]);
        assert(bhv::eq(tmp, hvs[i]));
//        bhv::swap_bits_of_byte_order_into_reference(hvs[i], hvs[i]);
    }

    FILE *ofile = fopen("hadamard_table.pbm", "wb");
    bhv::save_pbm(ofile, hvs, BITS);
    fclose(ofile);

    cout << bhv::to_string(hvs[0b110000000101]) << endl;

    for (uint32_t i = 0; i < BITS; ++i) {
        free(hvs[i]);
    }
}

void test_hadamard_components() {
//    uint32_t t = 3077;
//
//    cout << bitset<13>(t) << ", atomic: " << bitset<6>(t & 0b111111) << ", composite: " << bitset<13 - 6>(t >> 6) << std::endl;

    word_t *hvs[13];
    word_t *hvs_ref[13];

    for (uint32_t i = 0; i < 13; ++i) {
        hvs_ref[i] = bhv::empty();
        bhv::hadamard_encode_into(1 << i, hvs_ref[i]);
        bhv::invert_into(hvs_ref[i], hvs_ref[i]);
    }

    FILE *ofile = fopen("hadamard_basis_test.pbm", "wb");
    bhv::save_pbm(ofile, hvs, 13);
    fclose(ofile);

    ofile = fopen("hadamard_basis.pbm", "wb");
    bhv::save_pbm(ofile, hvs_ref, 13);
    fclose(ofile);

    for (uint32_t i = 0; i < 13; ++i) {
        cout << i << " eq " << bhv::eq(bhv::walsh_basis[i], hvs_ref[i]) << endl;
//        assert(bhv::eq(hvs[i], hvs_ref[i]));
        free(hvs_ref[i]);
    }
}

void test_levels() {
    word_t *hvs[BITS];

    for (uint32_t i = 0; i < BITS; ++i) {
        hvs[i] = bhv::empty();
        bhv::level_into(i, hvs[i]);
        bhv::swap_byte_order_inplace(hvs[i]);
    }

    FILE *ofile = fopen("level_table.pbm", "wb");
    bhv::save_pbm(ofile, hvs, BITS);
    fclose(ofile);

    for (uint32_t i = 0; i < BITS; ++i) {
        free(hvs[i]);
    }
}

void test_hadamard_codes() {
    for (uint32_t i = 0; i < BITS; ++i) {
        word_t codeword [WORDS];
        word_t noise [WORDS];
        word_t corrupted_codeword [WORDS];
        cout << "code" << i << endl;

        bhv::hadamard_encode_into(i, codeword);
        uint32_t reconstructed = bhv::hadamard_decode(codeword);
        cout << reconstructed << " (noise free)" << endl;
        assert(reconstructed == i);

        bhv::random_into(noise, .1);
        bhv::xor_into(codeword, noise, corrupted_codeword);
        reconstructed = bhv::hadamard_decode(corrupted_codeword);
        cout << reconstructed << " (10% noise)" << endl;
        assert(reconstructed == i);

        bhv::random_into(noise, .25);
        bhv::xor_into(codeword, noise, corrupted_codeword);
        reconstructed = bhv::hadamard_decode(corrupted_codeword);
        cout << reconstructed << " (25% noise)" << endl;
        assert(reconstructed == i);

        bhv::random_into(noise, .45);
        bhv::xor_into(codeword, noise, corrupted_codeword);
        reconstructed = bhv::hadamard_decode(corrupted_codeword);
        cout << reconstructed << " (45% noise)" << endl;
        assert(reconstructed == i);
    }
}

static void normalize(float_t* ws, size_t n) {
    if (n == 0) return;
    double minw = numeric_limits<double>::max();
    double maxw = numeric_limits<double>::min();
    for (size_t i = 0; i < n; ++i) {
        minw = min(minw, (double)ws[i]);
        maxw = max(maxw, (double)ws[i]);
    }
    if (maxw == minw) return;

    for (size_t i = 0; i < n; ++i)
        ws[i] = ((double)ws[i] - minw)/(maxw - minw);
}

static double proportional_loss(float_t* target, float_t* prediction, size_t n) {
    double total = 0;
    double mean_target = 0;
    double mean_prediction = 0;
    for (size_t i = 0; i < n; ++i) {
        mean_target += target[i];
        mean_prediction += prediction[i];
    }
    mean_target /= (double)n;
    mean_prediction /= (double)n;
    for (size_t i = 0; i < n; ++i) {
        double diff = 1. - (prediction[i]/(mean_prediction + 1.) + 1.)/(target[i]/(mean_target + 1.) + 1.);
        total += diff*diff;
    }
    return total;
}

static double kl_divergence(float_t* target, float_t* prediction, size_t n) {
    double total = 0;
    for (size_t i = 0; i < n; ++i)
        total += target[i]*log((target[i] + .0001)/(prediction[i] + .0001));
    return total;
}

static double mean_absolute_percentage_error(float_t* target, float_t* prediction, size_t n) {
    double total = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = target[i] - prediction[i];
        total += abs(diff);
    }
    return total*(100./(double)n);
}

static double mean_squared_error(float_t* target, float_t* prediction, size_t n) {
    double total = 0;
    for (size_t i = 0; i < n; ++i) {
        double diff = target[i] - prediction[i];
        total += diff*diff;
    }
    return total/(double)n;
}

struct metric {
    double active_fraction = 0.;
    double proportional_loss = 0.;
    double kl_divergence = 0.;
    double mean_absolute_percentage_error = 0.;
    double mean_squared_error = 0.;

    friend ostream &operator<<(ostream &os, const metric &metric) {
        os << "af: " << metric.active_fraction << " proportional: " << metric.proportional_loss
           << " KL-D: " << metric.kl_divergence << " MAPE: "
           << metric.mean_absolute_percentage_error << " MSE: " << metric.mean_squared_error;
        return os;
    }

    static metric mean(metric* ms, size_t n) {
        metric result;
        for (size_t i = 0; i < n; ++i) {
            result.active_fraction += ms[i].active_fraction;
            result.proportional_loss += ms[i].proportional_loss;
            result.kl_divergence += ms[i].kl_divergence;
            result.mean_absolute_percentage_error += ms[i].mean_absolute_percentage_error;
            result.mean_squared_error += ms[i].mean_squared_error;
        }
        result.active_fraction /= (double)n;
        result.proportional_loss /= (double)n;
        result.kl_divergence /= (double)n;
        result.mean_absolute_percentage_error /= (double)n;
        result.mean_squared_error /= (double)n;

        return result;
    }
};
void test_representative(size_t domain = 21, size_t image = 100) {
    bhv::rng.seed(107);
    std::normal_distribution<float_t> sampler(0.);
    word_t** hvs = (word_t**)malloc(sizeof(word_t*)*domain);
    float_t* weights = (float_t*)malloc(sizeof(float_t)*domain);
    float_t* recovered_weights = (float_t*)malloc(sizeof(float_t)*domain);
    word_t** rhvs = (word_t**)malloc(sizeof(word_t*)*image);
    metric* metrics = (metric*)malloc(sizeof(metric)*image);

    for (size_t i = 0; i < domain; ++i) {
        hvs[i] = bhv::empty();
#if false
        bhv::hadamard_encode_into(i + 1, hvs[i]);
#else
        bhv::rand_into(hvs[i]);
#endif
        weights[i] = 1.; //sampler(bhv::rng);
    }
    weights[0] = 0.;
    normalize(weights, domain);
    for (size_t i = 0; i < domain; ++i) cout << weights[i] << ", ";
    cout << endl;

    for (size_t i = 0; i < image; ++i) {
        rhvs[i] = bhv::empty();
#if true
//        bhv::representative_into_reference(hvs, domain, rhvs[i]);
        bhv::weighted_representative_into_reference(hvs, weights, domain, rhvs[i]);
#elif false
        double t = 0.;
        for (size_t j = 0; j < domain; ++j)
            t += weights[j];
//        bhv::weighted_threshold_into_naive<float_t>(hvs, weights, domain, t/2, rhvs[i]);
        bhv::true_majority_into(hvs, domain, rhvs[i]);
#else
        bhv::rand_into(rhvs[i]);
        bhv::opt::independent<double>(rhvs[i], [&weights, &recovered_weights, &hvs, domain](word_t* x) {
            for (size_t j = 0; j < domain; ++j)
                recovered_weights[j] = 1. - (float_t) bhv::hamming(hvs[j], x) / (float_t) BITS;
//            normalize(recovered_weights, domain);
            return mean_absolute_percentage_error(weights, recovered_weights, domain);
        });
#endif
        for (size_t j = 0; j < domain; ++j)
            recovered_weights[j] = 1. - (float_t)bhv::hamming(hvs[j], rhvs[i])/(float_t)BITS;
        normalize(recovered_weights, domain);

        for (size_t j = 0; j < domain; ++j) cout << recovered_weights[j] << ", ";
        cout << endl;

        metric m{
            bhv::active(rhvs[i])/(double)BITS,
            proportional_loss(weights, recovered_weights, domain),
            kl_divergence(weights, recovered_weights, domain),
            mean_absolute_percentage_error(weights, recovered_weights, domain),
            mean_squared_error(weights, recovered_weights, domain),
        };

        cout << m << endl;

        metrics[i] = m;
    }

    metric avg = metric::mean(metrics, image);
    cout << "summary: " << avg << endl;

    for (size_t i = 0; i < domain; ++i) free(hvs[i]);
    for (size_t i = 0; i < image; ++i) free(rhvs[i]);
    free(hvs);
    free(weights);
    free(recovered_weights);
    free(rhvs);
    free(metrics);
}

int main() {
//    test_easy_search();
//    test_harder_search();
//    test_opt_centroid();
//    test_hadamard();
//    test_hadamard_components();
//    test_levels();
//    test_hadamard_codes();
    test_representative();
}
