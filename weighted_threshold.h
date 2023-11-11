/// @brief A generic implementation for (exclusive) weighted_threshold_into, that can use any weight type, using unpack/pack
template<typename N>
void weighted_threshold_into_reference(word_t **xs, N *weights, size_t size, N threshold, word_t *target) {
    N totals[BITS];
    bool decision[BITS];
    memset(totals, (N)0, BITS*sizeof(N));

    for (size_t i = 0; i < size; ++i) {
        N w = weights[i];
        bool bit_buffer[BITS];
        unpack_into(xs[i], bit_buffer);

        for (size_t j = 0; j < BITS; ++j)
            if (bit_buffer[j])
                totals[j] += w;
    }

    for (size_t j = 0; j < BITS; ++j)
        decision[j] = threshold < totals[j];

    pack_into(decision, target);
}

/// @brief A generic implementation for (exclusive) weighted_threshold_into, that can use any weight type
template<typename N>
void weighted_threshold_into_naive(word_t **xs, N *weights, size_t size, N threshold, word_t *target) {
    N totals[BITS];
    memset(totals, (N)0, BITS*sizeof(N));

    for (size_t i = 0; i < size; ++i) {
        N w = weights[i];
        word_t *x = xs[i];

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = x[word_id];
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                totals[offset + bit_id] += w*((word >> bit_id) & 1);
            }
        }
    }

    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        bit_iter_t offset = word_id * BITS_PER_WORD;
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if (threshold < totals[offset + bit_id])
                word |= 1UL << bit_id;
        }
        target[word_id] = word;
    }
}

#if __AVX512BW__
void factor_threshold_into_byte_avx512(word_t **xs, uint8_t *weights, size_t size, uint8_t threshold, word_t *target) {
    __m512i totals[WORDS];
    memset(totals, 0, WORDS*sizeof(__m512i));

    for (size_t i = 0; i < size; ++i) {
        uint8_t w = weights[i];
        __m512i vw = _mm512_set1_epi8(w);
        word_t *x = xs[i];

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id)
            totals[word_id] = _mm512_mask_add_epi8(totals[word_id], x[word_id], totals[word_id], vw);
    }

    __m512i vthreshold = _mm512_set1_epi8(threshold);
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id)
        target[word_id] = _mm512_cmpgt_epu8_mask(totals[word_id], vthreshold);
}

void factor_threshold_into_byte_avx512_transpose(word_t **xs, uint8_t *weights, size_t size, uint8_t threshold, word_t *target) {
    __m512i vthreshold = _mm512_set1_epi8(threshold);
    std::vector<__m512i> vweights(size);

    for (size_t i = 0; i < size; ++i)
        vweights[i] = _mm512_set1_epi8(weights[i]);

    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        __m512i total = _mm512_setzero_si512();
        for (size_t i = 0; i < size; ++i)
            total = _mm512_mask_add_epi8(total, xs[i][word_id], total, vweights[i]);
        target[word_id] = _mm512_cmpgt_epu8_mask(total, vthreshold);
    }
}
#endif

void factor_threshold_into(word_t **xs, uint32_t *weights, size_t size, uint64_t threshold, word_t *target) {
    uint64_t max_result = 0;
    for (size_t i = 0; i < size; ++i)
        max_result += weights[i];

    assert(threshold <= max_result);

    if (max_result < 256)
        weighted_threshold_into_naive<uint8_t>(xs, (uint8_t*)weights, size, threshold, target);
    else if (max_result < 65536)
        weighted_threshold_into_naive<uint16_t>(xs, (uint16_t*)weights, size, threshold, target);
    else if (max_result < 4294967296)
        weighted_threshold_into_naive<uint32_t>(xs, (uint32_t*)weights, size, threshold, target);
    else
        weighted_threshold_into_naive<uint64_t>(xs, (uint64_t*)weights, size, threshold, target);
}

#define distribution_threshold_into weighted_threshold_into_naive<double_t>
