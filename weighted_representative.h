#if __AVX2__
__m256 randomf_range(float_t range) {
    __m256i bits = avx2_pcg32_random_r(&avx2_key);
    // Convert random bits into FP32 number in [ 1 .. 2 ) interval
    __m256i mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
    __m256i mantissa = _mm256_and_si256(bits, mantissa_mask);
    __m256 one = _mm256_set1_ps(1);
    __m256 val = _mm256_or_ps(_mm256_castsi256_ps(mantissa), one);

    // Scale the number from [ 1 .. 2 ) into [ 0 .. range ), via ( val * range ) - range
    __m256 rf = _mm256_set1_ps(range);
    val = _mm256_fmsub_ps(val, rf, rf);

    return val;
}
#endif

// DRAFT
/*void weighted_representative_into_avx2_binsearch(word_t **xs, float_t *cum_weights, size_t size, word_t *target) {
    // roughly BYTES*log2(size) gather's and BYTE pcg32_random's
    assert(false);
    // xs           0 1 2
    // cum_weights: 3 4 6
    // at iter i random([0, 6)) = 4.3
    // out[i] = xs[1][i]
    float_t limit = cum_weights[size - 1];
    float_t rands [BITS];

    for (size_t i = 0; i < BYTES; ++i) {
        __m256i ids = _mm256_set1_epi32(size/2);

        __m256 rand = randomf_range(limit);
        _mm256_storeu_ps(rands + i*8, rand);

        __m256 ws = _mm256_i32gather_ps(cum_weights, ids, 1);
        __m256i gt = _mm256_cmp_ps(ws, rand, 0);

        // ids<prev_lt> -= gt;
        // ids<prev_gt> += lt;
        // repeated for the max depth of binary search

        // gather values of binary search result
        __m256 vs = _mm256_i32gather_epi32(xs, ids, 1);

        // compress them into a result byte to update the target with
        ((uint8_t *)target)[byte_id] = result;
    }
}*/


void weighted_representative_into_reference(word_t **xs, float_t *weights, size_t size, word_t *target) {
    float_t* cum_weights = (float_t*)malloc(size*sizeof(float_t));
    memset(target, 0, BYTES);

    float_t total = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float_t w = weights[i];
        total += w;
        cum_weights[i] = total;
    }

    std::uniform_real_distribution<float_t> uniform(0., total);

    for (bit_iter_t i = 0; i < BITS; ++i) {
        float_t value = uniform(rng);
        size_t index = std::distance(cum_weights, std::lower_bound(cum_weights, cum_weights + size, value));
        if (get(xs[index], i)) update(target, i);
    }

    free(cum_weights);
}

#if __AVX2__
void weighted_representative_into_avx2(word_t **xs, float_t *weights, size_t size, word_t *target) {
    // roughly size*BYTES fmadd's and BYTE pcg32_random's
    __m256 totals[BYTES];
    memset(totals, 0, BYTES*sizeof(__m256));

    float_t total = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float_t w = weights[i];
        total += w;
        __m256 vw = _mm256_set1_ps(w);
        uint8_t *x = (uint8_t *)xs[i];

        for (bit_iter_t byte_id = 0; byte_id < BYTES; ++byte_id)
            totals[byte_id] = _mm256_fmadd_ps(factors[x[byte_id]], vw, totals[byte_id]);
    }

    for (bit_iter_t byte_id = 0; byte_id < BYTES; ++byte_id) {
        __m256 samples = randomf_range(total);
        uint8_t result = _mm256_movemask_ps(_mm256_cmp_ps(totals[byte_id], samples, _CMP_GT_OS));
        ((uint8_t *)target)[byte_id] = result;
    }
}
#endif

#if __AVX2__
#define weighted_representative_into weighted_representative_into_avx2
#else
#define weighted_representative_into weighted_representative_into_reference
#endif
