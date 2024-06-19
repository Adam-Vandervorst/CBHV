void parity_into_reference(word_t ** xs, size_t size, word_t *target) {
    memset(target, 0, BYTES);
    for (size_t i = 0; i < size; ++i)
        bhv::xor_into(xs[i], target, target);
}

// for some reason, auto-vectorization doesn't work on this one
void parity_into_wordlevel(word_t **xs, size_t size, word_t *target) {
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        word_t parity = xs[0][word_id];

        for (size_t i = 1; i < size; ++i)
            parity ^= xs[i][word_id];

        target[word_id] = parity;
    }
}

#if __AVX2__
void parity_into_avx2(word_t **xs, size_t size, word_t *target) {
    if (unlikely(size*BYTES > 128*1024*1024)) { // outside of L3 (say 128MiB), it's more efficient to load one after the other
        parity_into_reference(xs, size, target); // don't worry, the reference one is auto-vectorized
        return;
    }

    for (word_iter_t word_id = 0; word_id < BITS/256; ++word_id) {
        __m256i parity = _mm256_loadu_si256((__m256i*)xs[0] + word_id);

        for (size_t i = 1; i < size; ++i)
            parity ^= _mm256_loadu_si256((__m256i*)xs[i] + word_id);

        _mm256_storeu_si256((__m256i*)target + word_id, parity);
    }
}
#endif

#if __AVX512BW__
void parity_into_avx512(word_t **xs, size_t size, word_t *target) {
    if (unlikely(size*BYTES > 256*1024*1024)) { // outside of L3 (say 256MiB), it's more efficient to load one after the other
        parity_into_reference(xs, size, target); // don't worry, the reference one is auto-vectorized
        return;
    }

    for (word_iter_t word_id = 0; word_id < BITS/512; ++word_id) {
        __m512i parity = _mm512_loadu_si512((__m512i*)xs[0] + word_id);

        for (size_t i = 1; i < size; ++i)
            parity ^= _mm512_loadu_si512((__m512i*)xs[i] + word_id);

        _mm512_storeu_si512((__m512i*)target + word_id, parity);
    }
}
#endif

extern "C" void parity_into(word_t ** xs, size_t size, word_t *target) {
#if __AVX512BW__
    parity_into_avx512(xs, size, target);
#elif __AVX2__
    parity_into_avx2(xs, size, target);
#else
    parity_into_reference(xs, size, target);
#endif //#if __AVX512BW__
}