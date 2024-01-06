#include <iostream>
#include <chrono>
#include <functional>

#include "core.h"

using namespace std;

#define DO_VALIDATION true
#define SLOPPY_VALIDATION false

#define MAJ_INPUT_HYPERVECTOR_COUNT 1000001
#define INPUT_HYPERVECTOR_COUNT 100

//#define WITHIN
//#define TOP
//#define CLOSEST
//#define REPRESENTATIVE
//#define THRESHOLD
//#define WEIGHTED_THRESHOLD
//#define MAJ
//#define PARITY
//#define RAND
//#define RAND2
//#define RANDOM
//#define PERMUTE
#define ROLL
//#define ACTIVE
//#define HAMMING
//#define INVERT
//#define EVEN_ODD
//#define REHASH
//#define AND
//#define OR
//#define XOR
//#define SELECT
//#define MAJ3
//#define TERNARY

uint64_t hash_combine(uint64_t h, uint64_t k) {
    static constexpr uint64_t kM = 0xc6a4a7935bd1e995ULL;
    static constexpr int kR = 47;

    k *= kM;
    k ^= k >> kR;
    k *= kM;

    h ^= k;
    h *= kM;

    h += 0xe6546b64;

    return h;
}

#if SLOPPY_VALIDATION
#define summary(m) ((m)[0] + 3 * (m)[4] + 5 * (m)[WORDS / 2] + 7 * (m)[WORDS - 1])
#define integrate(t, h) ((t) ^ (h))
#else
#define summary(m) bhv::hash(m)
#define integrate(t, h) hash_combine((t), (h))
#endif

template <vector<size_t> F(word_t **, size_t, word_t *, bit_iter_t)>
void within_benchmark(size_t n, size_t k, bit_iter_t d, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    word_t **queries = (word_t**)malloc(sizeof(word_t*) * input_output_count);
    size_t **targets = (size_t**)malloc(sizeof(size_t*) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        size_t *target = (size_t *) malloc(sizeof(size_t) * k);
        vector<size_t> range(n);
        std::iota(range.begin(), range.end(), 0);
        std::sample(range.begin(), range.end(), target, k, bhv::rng);

        word_t *t = bhv::rand();
        for (size_t j = 0; j < n; ++j)
            rs[j] = bhv::rand();

        for (size_t j = 0; j < k; ++j) {
            bhv::permute_into(t, 0, rs[target[j]]);
            for (bit_iter_t b = 0; b < d/2; ++b)
                bhv::toggle(rs[target[j]], b);
        }

        free(t);

        inputs[i] = rs;
        targets[i] = target;
        queries[i] = rs[target[0]];
    }

    bool correct = true;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        vector<size_t> res = F(inputs[io_buf_idx], n, queries[io_buf_idx], d);

        correct &= std::is_permutation(targets[io_buf_idx], targets[io_buf_idx] + k, res.begin(), res.end());
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char *validation_status = (correct ? "correct: v" : "correct: x");

    double mean_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) test_count;
    if (display)
        cout << n << " hypervectors, " << validation_status << ", k: " << k << ", in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0
             << "µs, normalized: " << mean_test_time / (double) n << "ns/vec" << endl;

    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j)
            free(rs[j]);
        free(rs);
        free(targets[i]);
    }
    free(inputs);
    free(targets);
    free(queries);
}

template <void F(word_t **, size_t, word_t *, size_t, size_t *)>
void top_benchmark(size_t n, size_t k, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    word_t **queries = (word_t**)malloc(sizeof(word_t*) * input_output_count);
    size_t **targets = (size_t**)malloc(sizeof(size_t*) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        size_t *target = (size_t *) malloc(sizeof(size_t) * k);
        vector<size_t> range(n);
        std::iota(range.begin(), range.end(), 0);
        std::sample(range.begin(), range.end(), target, k, bhv::rng);

        word_t *t = bhv::rand();
        for (size_t j = 0; j < n; ++j)
            rs[j] = bhv::rand();

        for (size_t j = 0; j < k; ++j) {
//            bhv::permute_into(t, 0, rs[target[j]]);
//            for (bit_iter_t b = 0; b < BITS*(float)j/(1.1*(float)k); ++b)
//                bhv::toggle(rs[target[j]], b);
            word_t *r = bhv::random((float)j/(float)k);
            bhv::select_into(r, rs[target[j]], t, rs[target[j]]);
            free(r);
        }

        free(t);

        inputs[i] = rs;
        targets[i] = target;
        queries[i] = rs[target[0]];
    }

    bool correct = true;
    size_t *result_buffer = (size_t *) malloc(sizeof(size_t) * input_output_count * k);

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        F(inputs[io_buf_idx], n, queries[io_buf_idx], k, result_buffer + io_buf_idx*k);

        correct &= std::equal(targets[io_buf_idx], targets[io_buf_idx] + k, result_buffer + io_buf_idx*k);
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char *validation_status = (correct ? "correct: v" : "correct: x");

    double mean_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) test_count;
    if (display)
        cout << n << " hypervectors, " << validation_status << ", k: " << k << " , in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0
             << "µs, normalized: " << mean_test_time / (double) n << "ns/vec" << endl;

    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j)
            free(rs[j]);
        free(rs);
        free(targets[i]);
    }
    free(inputs);
    free(targets);
    free(queries);
    free(result_buffer);
}


template <size_t F(word_t **, size_t, word_t *)>
void closest_benchmark(size_t n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    std::uniform_int_distribution<size_t> target_gen(0, n-1);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    word_t **queries = (word_t**)malloc(sizeof(word_t*) * input_output_count);
    size_t *target = (size_t*)malloc(sizeof(size_t) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        for (size_t j = 0; j < n; ++j)
            rs[j] = bhv::rand();
        inputs[i] = rs;
        target[i] = target_gen(bhv::rng);
        queries[i] = rs[target[i]];
    }

    bool correct = true;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        correct &= target[io_buf_idx] == F(inputs[io_buf_idx], n, queries[io_buf_idx]);
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char *validation_status = (correct ? "correct: v, " : "correct: x, ");

    double mean_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) test_count;
    if (display)
        cout << n << " hypervectors, " << validation_status << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0
             << "µs, normalized: " << mean_test_time / (double) n << "ns/vec" << endl;

    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j)
            free(rs[j]);
        free(rs);
    }
    free(inputs);
    free(target);
    free(queries);
}


float threshold_benchmark(size_t n, size_t threshold, float af, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        for (size_t j = 0; j < n; ++j) {
            rs[j] = bhv::random(af);
        }
        inputs[i] = rs;
    }

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    volatile uint64_t checksum = 0;
    volatile uint64_t val_checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t **rs = inputs[io_buf_idx];

        bhv::threshold_into(rs, n, threshold, m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char* validation_status;
    if (DO_VALIDATION) {
        for (size_t i = 0; i < test_count; i++) {
            const size_t io_buf_idx = (keep_in_cache ? 0 : i);

            word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
            word_t **rs = inputs[io_buf_idx];

            bhv::threshold_into_reference(rs, n, threshold, m);

            val_checksum = integrate(val_checksum, summary(m));
        }
        validation_status = ((checksum == val_checksum) ? "equiv: v, " : "equiv: x, ");
    } else {
        validation_status = "";
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << n << " hypervectors, " << threshold << " threshold, " << validation_status << "in_cache: " << keep_in_cache
            << ", total: " << mean_test_time / 1000.0 << "µs, normalized: " << mean_test_time / (float) n << "ns/vec" << endl;

    //Clean up our mess
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j) {
            free(rs[j]);
        }
        free(rs);
    }
    free(result_buffer);
    free(inputs);

    return mean_test_time;
}

template <void F(word_t **, size_t, word_t *), void FC(word_t **, size_t, word_t *)>
float nary_benchmark(size_t n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        for (size_t j = 0; j < n; ++j) {
            rs[j] = bhv::rand();
        }
        inputs[i] = rs;
    }

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile uint64_t checksum = 0;
    volatile uint64_t val_checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t **rs = inputs[io_buf_idx];

        F(rs, n, m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char* validation_status;
    if (DO_VALIDATION) {
        for (size_t i = 0; i < test_count; i++) {
            const size_t io_buf_idx = (keep_in_cache ? 0 : i);

            word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
            word_t **rs = inputs[io_buf_idx];

            FC(rs, n, m);

            val_checksum = integrate(val_checksum, summary(m));
        }
        validation_status = ((checksum == val_checksum) ? "equiv: v, " : "equiv: x, ");
    } else {
        validation_status = "";
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << n << " hypervectors, " << validation_status << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0
             << "µs, normalized: " << mean_test_time / (float) n << "ns/vec" << endl;

    //Clean up our mess
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j) {
            free(rs[j]);
        }
        free(rs);
    }
    free(result_buffer);
    free(inputs);

    return mean_test_time;
}


float rand_benchmark(bool display, bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    volatile uint64_t checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand_into(m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}

float rand2_benchmark(bool display, bool keep_in_cache, int pow) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);
    volatile int p = pow;

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    double observed_frac[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand2_into(m, p);

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double) bhv::active(m) / (double) BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed =
            std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>()) / (double) test_count;
    mean_observed = mean_observed > .5 ? 1 + std::log2(1 - mean_observed) : -1 - std::log2(mean_observed);
    if (display)
        cout << pow << "pow, observed: " << mean_observed << ", in_cache: " << keep_in_cache << ", total: "
             << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}


float random_benchmark(bool display, bool keep_in_cache, float base_frac, bool randomize = false) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);
    volatile double n = randomize ? ((double) (rand() - RAND_MAX / 2) / (double) (RAND_MAX)) / 1000. : 0;
    float p = base_frac + n;

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    double observed_frac[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

//        bhv::random_into_reference(m, p);
        bhv::random_into(m, p);
//        bhv::random_into_buffer_avx2(m, p);

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double) bhv::active(m) / (double) BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed_frac =
            std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>()) / (double) test_count - n;
    if (display)
        cout << base_frac << "frac, observed error: " << abs(base_frac - mean_observed_frac) << ", in_cache: "
             << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}

template<void F(word_t*, int64_t, word_t*), bool in_place, int different_permutations>
pair<float, float> permute_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT * different_permutations * 2;

    bool correct = true;
    double permute_test_time;
    double unpermute_test_time;
    int64_t perms [different_permutations];
    for (size_t i = 0; i < different_permutations; ++i)
        perms[i] = i;

    if constexpr (in_place) {
        word_t *initial [INPUT_HYPERVECTOR_COUNT];
        word_t *final [INPUT_HYPERVECTOR_COUNT];

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            initial[i] = bhv::rand();
            final[i] = bhv::permute(initial[i], 0); // TODO do we need a copy utility instead of abusing permute?
        }

        auto t1 = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            for (size_t j = 0; j < different_permutations; ++j)
                F(final[i], perms[j], final[i]);

        auto t2 = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            for (size_t j = different_permutations; j > 0; --j)
                F(final[i], -perms[j - 1], final[i]);

        auto t3 = chrono::high_resolution_clock::now();

        permute_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
        unpermute_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            if (!bhv::eq(final[i], initial[i]))
                std::cout << "failed " << i << ", " << perms[i] << std::endl;
            correct &= bhv::eq(final[i], initial[i]);
        }

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            free(initial[i]);
            free(final[i]);
        }
    } else {
        word_t *forward [INPUT_HYPERVECTOR_COUNT][different_permutations + 1];
        word_t *backward [INPUT_HYPERVECTOR_COUNT][different_permutations + 1];
        // different_permutations=3
        // forward:  R                              p0(R)                     p1(p0(R))            p2(p1(p0(R)))
        //           | hopefully equal                                                  set equal  |
        // backward: p-0(p-1(p-2(p2(p1(p0(R))))))   p-1(p-2(p2(p1(p0(R)))))   p-2(p2(p1(p0(R))))   p2(p1(p0(R)))

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            forward[i][0] = bhv::rand();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            for (size_t j = 0; j < different_permutations; ++j)
                forward[i][j + 1] = bhv::empty();

            for (size_t j = different_permutations; j > 0; --j)
                backward[i][j - 1] = bhv::empty();
        }

        auto t1 = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            for (size_t j = 0; j < different_permutations; ++j)
                F(forward[i][j], perms[j], forward[i][j + 1]);

        auto t2 = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            backward[i][different_permutations] = forward[i][different_permutations];

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i)
            for (size_t j = different_permutations; j > 0; --j)
                F(backward[i][j], -perms[j - 1], backward[i][j - 1]);

        auto t3 = chrono::high_resolution_clock::now();

        permute_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
        unpermute_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            if (!bhv::eq(forward[i][0], backward[i][0]))
                std::cout << "failed " << i << ", " << perms[i] << std::endl;
            correct &= bhv::eq(forward[i][0], backward[i][0]);
        }
        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            for (size_t j = 0; j < different_permutations; ++j) free(forward[i][j]);
            for (size_t j = 0; j < different_permutations - 1; ++j) free(backward[i][j]); // -1, don't double free
        }
    }

    double mean_permute_test_time = permute_test_time / (double) test_count;
    double mean_unpermute_test_time = unpermute_test_time / (double) test_count;
    if (display)
        cout << "correctly inverted: " << (correct ? "v" : "x") << ", in_place: " << in_place
        << ", permuting: " << mean_permute_test_time / 1000.0 << "µs"
        << ", unpermuting: " << mean_unpermute_test_time / 1000.0 << "µs" << endl;

    return {mean_permute_test_time, mean_unpermute_test_time};
}

float active_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;

    word_t *hvs [test_count];

    for (size_t i = 0; i < test_count; ++i) {
        hvs[i] = bhv::random((float)i/(float)test_count);
    }

    uint32_t observed_active[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = hvs[i];
        observed_active[i] = bhv::active(m);
        // observed_active[i] = bhv::active_avx512(m);
        // observed_active[i] = bhv::active_adder_avx2(m);
        // observed_active[i] = bhv::active_reference(m);
    }
    auto t2 = chrono::high_resolution_clock::now();

    bool correct = true;
    for (size_t i = 0; i < test_count; ++i) {
        correct &= observed_active[i] == bhv::active_reference(hvs[i]);
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "correct " << (correct ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

float hamming_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT/2;

    word_t *as [test_count];
    word_t *bs [test_count];

    for (size_t i = 0; i < test_count; ++i) {
        as[i] = bhv::random((float)i/(float)test_count);
        bs[i] = bhv::random((float)(test_count - i)/(float)test_count);
    }

    uint32_t observed_distance[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        observed_distance[i] = bhv::hamming(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_reference(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_adder_avx2(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_avx512(as[i], bs[i]);
    }
    auto t2 = chrono::high_resolution_clock::now();

    bool correct = true;
    for (size_t i = 0; i < test_count; ++i) {
        correct &= observed_distance[i] == bhv::hamming_reference(as[i], bs[i]);
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "correct " << (correct ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    for (size_t i = 0; i < test_count; ++i) {
        free(as[i]);
        free(bs[i]);
    }
    return mean_test_time;
}

template <void F(word_t*, word_t*), void FC(word_t*, word_t*)>
float unary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs[i] = bhv::random((float)i/(float)test_count);

    volatile uint64_t checksum = 0;
    volatile uint64_t val_checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs[i], m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs[i], m);

        val_checksum = integrate(val_checksum, summary(m));
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((checksum == val_checksum) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    free(result_buffer);

    return mean_test_time;
}

template <void F(word_t*, word_t*, word_t*), void FC(word_t*, word_t*, word_t*)>
float binary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs0 [test_count];
    word_t *hvs1 [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs0[i] = bhv::random((float)i/(float)test_count);

    memcpy(hvs1, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs1, hvs1 + test_count, bhv::rng);

    volatile uint64_t checksum = 0;
    volatile uint64_t val_checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs0[i], hvs1[i], m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs0[i], hvs1[i], m);

        val_checksum = integrate(val_checksum, summary(m));
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((checksum == val_checksum) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    free(result_buffer);

    return mean_test_time;
}

typedef void(*TernaryFunc)(word_t*, word_t*, word_t*, word_t*);

template <TernaryFunc F, TernaryFunc FC>
float ternary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs0 [test_count];
    word_t *hvs1 [test_count];
    word_t *hvs2 [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs0[i] = bhv::random((float)i/(float)test_count);

    memcpy(hvs1, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs1, hvs1 + test_count, bhv::rng);
    memcpy(hvs2, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs2, hvs2 + test_count, bhv::rng);

    volatile uint64_t checksum = 0;
    volatile uint64_t val_checksum = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs0[i], hvs1[i], hvs2[i], m);

        checksum = integrate(checksum, summary(m));
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs0[i], hvs1[i], hvs2[i], m);

        val_checksum = integrate(val_checksum, summary(m));
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((checksum == val_checksum) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    free(result_buffer);

    return mean_test_time;
}


template<int32_t o, void roll(word_t*, int32_t, word_t*)>
void fixed_roll(word_t *x, word_t *target) {
    return roll(x, o, target);
}

template<void F(word_t*, int32_t, word_t*)>
void roll_as_permute(word_t *x, int64_t p, word_t *target) {
    return F(x, (int32_t)p, target);
}

inline void simulated_select(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0xca);
}

inline void simulated_maj3(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0xe8);
}

inline void simulated_any(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0b11111110);
}

void __attribute__ ((noinline)) any_via_threshold(word_t *x, word_t *y, word_t *z, word_t *target) {
    word_t *xs [3] = {x, y, z};
    bhv::threshold_into(xs, 3, 0, target);
}

inline void simulated_majority(word_t ** xs, size_t size, word_t *target) {
    bhv::threshold_into_reference<uint32_t>(xs, size, size/2, target);
}

template <typename N, void F(word_t **, N *, size_t, N, word_t *)>
inline void simulated_majority_iota_factors(word_t **xs, size_t size, word_t *target) {
    std::vector<N> factors(size);
    std::iota(factors.begin(), factors.end(), (N)0);
    F(xs, factors.data(), size, size*(size-1)/2, target);
}

template <typename N, void F(word_t **, N *, size_t, N, word_t *)>
inline void simulated_majority_uniform_distribution(word_t **xs, size_t size, word_t *target) {
    std::vector<N> weights(size);
    for (size_t i = 0; i < size; ++i)
        weights[i] = (N)(1./(double_t)size);
    F(xs, weights.data(), size, (N)size/2, target);
}


template <void F(word_t*)>
void nondestructive_unary(word_t *x, word_t *target) {
    memcpy(target, x, BYTES);
    F(target);
}


int main() {
    cout << "*-= WARMUP =-*" << endl;
    // burn some cycles to get the OS's attention
    volatile uint64_t x = 0x7834d688d8827099ULL;
    for (size_t i = 0; i < 50000000; ++i)
        x = x + (x % 7);

    cout << "*-= STARTING (" << x << ") =-*" << endl;

#ifdef TERNARY
    ternary_benchmark<simulated_select, bhv::select_into_reference>(false, true);

    cout << "*-= TERNARY =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<simulated_select, bhv::select_into_reference>(true, true);
    ternary_benchmark<simulated_maj3, bhv::majority3_into>(true, true);
    ternary_benchmark<simulated_any, any_via_threshold>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<simulated_select, bhv::select_into_reference>(true, false);
    ternary_benchmark<simulated_maj3, bhv::majority3_into>(true, false);
    ternary_benchmark<simulated_any, any_via_threshold>(true, false);
#endif
#ifdef MAJ3
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(false, true);

    cout << "*-= MAJ3 =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
#endif
#ifdef SELECT
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(false, true);

    cout << "*-= SELECT =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
#endif
#ifdef XOR
    binary_benchmark<bhv::xor_into, bhv::xor_into>(false, true);

    cout << "*-= XOR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
#endif
#ifdef XOR
    binary_benchmark<bhv::xor_into, bhv::xor_into>(false, true);

    cout << "*-= XOR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
#endif
#ifdef OR
    binary_benchmark<bhv::or_into, bhv::or_into>(false, true);

    cout << "*-= OR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
#endif
#ifdef AND
    binary_benchmark<bhv::and_into, bhv::and_into>(false, true);

    cout << "*-= AND =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
#endif
#ifdef REHASH
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(false, true);

    cout << "*-= REHASH =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
#endif
#ifdef EVEN_ODD
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(false, true);

    cout << "*-= SWAP_HALVES =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, true);
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, true);
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, false);
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, false);
    unary_benchmark<bhv::swap_even_odd_into, bhv::swap_even_odd_into_reference>(true, false);
#endif
#ifdef INVERT
    unary_benchmark<bhv::invert_into, bhv::invert_into>(false, true);

    cout << "*-= INVERT =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
#endif
#ifdef HAMMING
    hamming_benchmark(false);

    cout << "*-= HAMMING =-*" << endl;
    hamming_benchmark(true);
    hamming_benchmark(true);
    hamming_benchmark(true);
#endif
#ifdef ACTIVE
    active_benchmark(false);

    cout << "*-= ACTIVE =-*" << endl;
    active_benchmark(true);
    active_benchmark(true);
    active_benchmark(true);
#endif
#ifdef PERMUTE
    permute_benchmark<bhv::permute_into, false, 100>(false);

    cout << "*-= PERMUTE =-*" << endl;
    // FIXME permute_into doesn't work in place yet!
//    cout << "*-= IN CACHE TESTS =-*" << endl;
//    permute_benchmark<bhv::permute_into, true, 100>(true);
//    permute_benchmark<bhv::permute_into, true, 100>(true);
//    permute_benchmark<bhv::permute_into, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_into, false, 100>(true);
    permute_benchmark<bhv::permute_into, false, 100>(true);
    permute_benchmark<bhv::permute_into, false, 100>(true);

    cout << "*-= WORD PERMUTE =-*" << endl;
//    cout << "*-= IN CACHE TESTS =-*" << endl;
//    permute_benchmark<bhv::permute_words_into, true, 100>(true);
//    permute_benchmark<bhv::permute_words_into, true, 100>(true);
//    permute_benchmark<bhv::permute_words_into, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_words_into, false, 100>(true);
    permute_benchmark<bhv::permute_words_into, false, 100>(true);
    permute_benchmark<bhv::permute_words_into, false, 100>(true);

    cout << "*-= BYTE PERMUTE =-*" << endl;
//    cout << "*-= IN CACHE TESTS =-*" << endl;
//    permute_benchmark<bhv::permute_bytes_into, true, 100>(true);
//    permute_benchmark<bhv::permute_bytes_into, true, 100>(true);
//    permute_benchmark<bhv::permute_bytes_into, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_bytes_into, false, 100>(true);
    permute_benchmark<bhv::permute_bytes_into, false, 100>(true);
    permute_benchmark<bhv::permute_bytes_into, false, 100>(true);

    cout << "*-= BYTE BITS PERMUTE =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_byte_bits_into, true, 100>(true);
    permute_benchmark<bhv::permute_byte_bits_into, true, 100>(true);
    permute_benchmark<bhv::permute_byte_bits_into, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_byte_bits_into, false, 100>(true);
    permute_benchmark<bhv::permute_byte_bits_into, false, 100>(true);
    permute_benchmark<bhv::permute_byte_bits_into, false, 100>(true);

#ifdef __AVX512BW__
    cout << "*-= WORD BITS PERMUTE =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_word_bits_into, true, 100>(true);
    permute_benchmark<bhv::permute_word_bits_into, true, 100>(true);
    permute_benchmark<bhv::permute_word_bits_into, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<bhv::permute_word_bits_into, false, 100>(true);
    permute_benchmark<bhv::permute_word_bits_into, false, 100>(true);
    permute_benchmark<bhv::permute_word_bits_into, false, 100>(true);
#endif
#endif
#ifdef ROLL
    permute_benchmark<roll_as_permute<bhv::roll_words_into>, false, 100>(false);

    cout << "*-= ROLL WORDS =-*" << endl;
    // FIXME roll_words_into doesn't work in place yet!
//    cout << "*-= IN CACHE TESTS =-*" << endl;
//    permute_benchmark<roll_as_permute<bhv::roll_words_into>, true, 100>(true);
//    permute_benchmark<roll_as_permute<bhv::roll_words_into>, true, 100>(true);
//    permute_benchmark<roll_as_permute<bhv::roll_words_into>, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_words_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_words_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_words_into>, false, 100>(true);

    cout << "*-= ROLL BYTES =-*" << endl;
//    cout << "*-= IN CACHE TESTS =-*" << endl;
//    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, true, 100>(true);
//    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, true, 100>(true);
//    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_bytes_into>, false, 100>(true);

    cout << "*-= ROLL WORD BITS =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, true, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, true, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_word_bits_into>, false, 100>(true);

    cout << "*-= ROLL BITS =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_bits_into_reference>, true, 100>(true);
//    permute_benchmark<roll_as_permute<bhv::roll_bits_into_single_pass>, true, 100>(true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    permute_benchmark<roll_as_permute<bhv::roll_bits_into_reference>, false, 100>(true);
    permute_benchmark<roll_as_permute<bhv::roll_bits_into_single_pass>, false, 100>(true);
#endif
#ifdef RAND
    rand_benchmark(false, false);

    cout << "*-= RAND =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    rand_benchmark(true, true);
    rand_benchmark(true, true);
    rand_benchmark(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    rand_benchmark(true, false);
    rand_benchmark(true, false);
    rand_benchmark(true, false);
#endif
#ifdef RAND2
    int8_t pws[33];
    for (int8_t i = -16; i < 17; ++i)
        pws[i + 16] = i;

    rand2_benchmark(false, false, 4);
    cout << "*-= RAND2 =-*" << endl;

    cout << "*-= IN CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, true, p);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, false, p);
#endif
#ifdef RANDOM
    random_benchmark(false, false, .1);

    cout << "*-= RANDOM =-*" << endl;
    cout << "*-= COMMON =-*" << endl;
    float common[13] = {.001, .01, .04, .2, .26, .48, .5, .52, .74,.8, .95, .99, .999};
    cout << "*-= IN CACHE TESTS =-*" << endl;
    double total = 0.;
    for (float p: common)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: common)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;

    cout << "*-= SMALL =-*" << endl;
    float small[9] = {1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2};
    cout << "*-= IN CACHE TESTS =-*" << endl;
    total = 0.;
    for (float p: small)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: small)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;

    cout << "*-= PERCENTAGES =-*" << endl;
    float perc[99];
    for (size_t i = 1; i < 100; ++i)
        perc[i - 1] = (float) i / 100.f;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    total = 0.;
    for (float p: perc)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: perc)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;
#endif
#ifdef PARITY
    //Run one throw-away test to make sure the OS is ready to give us full resource

    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(3, false, false);

    cout << "*-= PARITY =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(3, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(5, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(7, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(9, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(11, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(15, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(17, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(19, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(21, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(23, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(25, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(27, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(39, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(47, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(55, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(63, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(73, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(77, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(79, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(81, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(85, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(89, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(91, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(109, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(175, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(201, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(255, true, true);

    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(257, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(385, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(511, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(667, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(881, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(945, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(1021, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(2001, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(5001, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(9999, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(10003, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(20001, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(200001, true, true);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(1000001, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(3, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(5, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(7, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(9, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(11, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(27, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(39, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(47, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(55, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(63, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(73, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(77, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(79, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(81, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(85, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(89, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(91, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(109, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(175, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(201, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(255, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(257, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(313, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(385, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(511, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(667, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(881, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(945, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(1021, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(2001, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(5001, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(9999, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(10003, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(20001, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(200001, true, false);
    nary_benchmark<bhv::parity_into, bhv::parity_into_reference>(1000001, true, false);
#endif
#ifdef MAJ
    nary_benchmark<bhv::true_majority_into, simulated_majority>(3, false, false);

    cout << "*-= MAJ =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    nary_benchmark<bhv::true_majority_into, simulated_majority>(3, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(5, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(7, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(9, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(11, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(15, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(17, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(19, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(21, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(23, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(25, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(27, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(39, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(47, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(55, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(63, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(73, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(77, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(79, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(81, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(85, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(89, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(91, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(109, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(175, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(201, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(255, true, true);

    nary_benchmark<bhv::true_majority_into, simulated_majority>(257, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(385, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(511, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(667, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(881, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(945, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(1021, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(2001, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(5001, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(9999, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(10003, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(20001, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(200001, true, true);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(1000001, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    nary_benchmark<bhv::true_majority_into, simulated_majority>(3, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(5, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(7, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(9, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(11, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(27, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(39, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(47, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(55, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(63, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(73, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(77, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(79, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(81, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(85, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(89, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(91, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(109, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(175, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(201, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(255, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(257, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(313, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(385, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(511, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(667, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(881, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(945, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(1021, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(2001, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(5001, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(9999, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(10003, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(20001, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(200001, true, false);
    nary_benchmark<bhv::true_majority_into, simulated_majority>(1000001, true, false);
#endif
#ifdef WEIGHTED_THRESHOLD
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_naive<uint8_t>>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(3, false, false);
    // FIXME silly slow
    cout << "*-= WEIGHTED_THRESHOLD =-*" << endl;
    cout << "*-= FACTORS =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_naive<uint8_t>>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(3, true, true);
//    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(3, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(3, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512_transpose>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(3, true, true);
//
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_naive<uint8_t>>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(10, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(10, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512_transpose>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(10, true, true);
//
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_naive<uint8_t>>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(20, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(20, true, true);
//    nary_benchmark<simulated_majority_iota_factors<uint8_t, bhv::factor_threshold_into_byte_avx512_transpose>, simulated_majority_iota_factors<uint8_t, bhv::weighted_threshold_into_reference<uint8_t>>>(20, true, true);
    cout << "*-= DISTRIBUTION =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(3, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(3, true, true);

    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(10, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(10, true, true);

    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(20, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(20, true, true);

    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(100, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(100, true, true);

    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(1000, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(1000, true, true);

    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_naive<float_t>>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(10000, true, true);
    nary_benchmark<simulated_majority_uniform_distribution<float_t, bhv::distribution_threshold_into>, simulated_majority_uniform_distribution<float_t, bhv::weighted_threshold_into_reference<float_t>>>(10000, true, true);
#endif
#ifdef THRESHOLD
    threshold_benchmark(3, 0, .5, false, false);

    cout << "*-= THRESHOLD =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    threshold_benchmark(3, 0, .5, true, true);
    threshold_benchmark(10, 2, .3, true, true);
    threshold_benchmark(30, 20, .7, true, true);


    threshold_benchmark(100, 48, .5, true, true);
    threshold_benchmark(200, 50, .25, true, true);

    threshold_benchmark(3000, 1502, .5, true, true);
    threshold_benchmark(4000, 1000, .25, true, true);

    threshold_benchmark(200001, 0, .5, true, true);
    threshold_benchmark(200001, 200000, .5, true, true);

    threshold_benchmark(1000001, 498384, .5, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    threshold_benchmark(3, 0, .5, true, false);
    threshold_benchmark(10, 2, .3, true, false);
    threshold_benchmark(30, 20, .7, true, false);


    threshold_benchmark(100, 48, .5, true, false);
    threshold_benchmark(200, 50, .25, true, false);

    threshold_benchmark(3000, 1502, .5, true, false);
    threshold_benchmark(4000, 1000, .25, true, false);

    threshold_benchmark(200001, 0, .5, true, false);
    threshold_benchmark(200001, 200000, .5, true, false);

    threshold_benchmark(1000001, 498384, .5, true, false);

    cout << "*-= CART =-*" << endl;
    for (uint8_t i = 0; i < 12; ++i) for (uint8_t j = 0; j < i; ++j) {
        threshold_benchmark(i, j, .5, true, true);
    }
#endif
#ifdef REPRESENTATIVE
    nary_benchmark<bhv::representative_into, bhv::representative_into>(3, false, false);

    cout << "*-= REPRESENTATIVE =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    for (size_t i = 1; i < 100000; i = i < 10 ? i+1 : (size_t)(i*1.111))
        nary_benchmark<bhv::representative_into, bhv::representative_into>(i, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (size_t i = 1; i < 100000; i = i < 10 ? i+1 : (size_t)(i*1.111))
        nary_benchmark<bhv::representative_into, bhv::representative_into>(i, true, false);
#endif
#ifdef CLOSEST
    cout << "*-= CLOSEST =-*" << endl;
    cout << "*-= SERIAL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    closest_benchmark<bhv::closest>(10000, false, true);
    for (size_t i = 1; i <= 1000000; i *= 10)
        closest_benchmark<bhv::closest>(i, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    closest_benchmark<bhv::closest>(10000, false, false);
    for (size_t i = 1; i <= 1000000; i *= 10)
        closest_benchmark<bhv::closest>(i, true, false);
    cout << "*-= PARALLEL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    closest_benchmark<bhv::closest_parallel>(100000, false, true);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        closest_benchmark<bhv::closest_parallel>(i, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    closest_benchmark<bhv::closest_parallel>(100000, false, false);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        closest_benchmark<bhv::closest_parallel>(i, true, false);
#endif
#ifdef TOP
    cout << "*-= TOP =-*" << endl;
    cout << "*-= SERIAL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    top_benchmark<bhv::top_into>(10000, 5, false, true);
    for (size_t i = 10; i <= 1000000; i *= 10)
        top_benchmark<bhv::top_into>(i, 5, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    top_benchmark<bhv::top_into>(10000, 5, false, false);
    for (size_t i = 10; i <= 1000000; i *= 10)
        top_benchmark<bhv::top_into>(i, 5, true, false);
    cout << "*-= PARALLEL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    top_benchmark<bhv::top_into_parallel>(100000, 5, false, true);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        top_benchmark<bhv::top_into_parallel>(i, 5, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    top_benchmark<bhv::top_into_parallel>(100000, 5, false, false);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        top_benchmark<bhv::top_into_parallel>(i, 5, true, false);
#endif
#ifdef WITHIN
    cout << "*-= WITHIN =-*" << endl;
    cout << "*-= SERIAL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    within_benchmark<bhv::within>(10000, 50, BITS/4, false, true);
    for (size_t i = 100; i <= 1000000; i *= 10)
        within_benchmark<bhv::within>(i, 50, BITS/4, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    within_benchmark<bhv::within>(10000, 50, BITS/4, false, false);
    for (size_t i = 100; i <= 1000000; i *= 10)
        within_benchmark<bhv::within>(i, 50, BITS/4, true, false);
    cout << "*-= PARALLEL =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    within_benchmark<bhv::within_parallel>(100000, 50, BITS/4, false, true);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        within_benchmark<bhv::within_parallel>(i, 50, BITS/4, true, true);
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    within_benchmark<bhv::within_parallel>(100000, 50, BITS/4, false, false);
    for (size_t i = 1000; i <= 1000000; i *= 10)
        within_benchmark<bhv::within_parallel>(i, 50, BITS/4, true, false);
#endif
}
