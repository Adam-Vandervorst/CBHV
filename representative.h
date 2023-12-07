/// @brief Plain C implementation of representative sampling
void representative_into_bitcount_reference(word_t **xs, size_t size, word_t *target) {
    std::uniform_int_distribution<size_t> gen(0, size - 1);
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            size_t x_id = gen(rng);
            if ((xs[x_id][word_id] >> bit_id) & 1)
                word |= 1UL << bit_id;
        }
        target[word_id] = word;
    }
}

#ifdef __AVX2__
constexpr uint8_t reachable_binary(uint8_t p, uint8_t d, uint8_t circ, uint8_t n) {
    if (d > circ)
        return 0;

    uint8_t power_factor = 1 << (circ - d);
    uint8_t lower = p * power_factor;
    uint8_t upper = (lower + power_factor < n) ? lower + power_factor : n;

    uint8_t diff = upper - lower;
    return (lower < n) ? diff : 0;
}

// Recursion is slow? Not if you do it compiletime! (now compiletime is slow)
template<uint8_t p, uint8_t d, uint8_t circ, uint8_t n>
word_t* representative_into_recursive_avx2_inner(word_t **vs, word_t **rs) {
    constexpr uint8_t L0 = reachable_binary(p << 1, d + 1, circ, n);
    constexpr uint8_t L1 = reachable_binary(1 | (p << 1), d + 1, circ, n);

    if constexpr (L0 != 0 and L1 != 0) {
        static word_t node [WORDS];
        word_t * cond;
        if constexpr (L0 == L1) cond = rs[d];
        else { random_into(node, L1 / (L0 + L1)); cond = node; }

        select_into(cond,
            representative_into_recursive_avx2_inner<1 | (p << 1), d + 1, circ, n>(vs, rs),
            representative_into_recursive_avx2_inner<p << 1, d + 1, circ, n>(vs, rs), node);
        return node;
    } else if constexpr (L0 != 0) {
        return representative_into_recursive_avx2_inner<p << 1, d + 1, circ, n>(vs, rs);
    } else if constexpr (L1 != 0) {
        return representative_into_recursive_avx2_inner<1 | (p << 1), d + 1, circ, n>(vs, rs);
    } else {
        return vs[p];
    }
}

template <uint8_t n>
void representative_into_recursive_avx2(word_t **xs, word_t *target) {
    constexpr uint8_t circumscribed = 64 - __builtin_clzll(n - 1);

    word_t rs_buf [circumscribed][WORDS];
    word_t *rs [circumscribed];

    for (size_t i = 0; i < circumscribed; ++i) {
        rand_into(rs_buf[i]);
        rs[i] = rs_buf[i];
    }

    word_t* res = representative_into_recursive_avx2_inner<0, 0, circumscribed, n>(xs, rs);

    memcpy(target, res, BYTES);
}

// Looks fancy but is quite slow
template <size_t circumscribed>
void representative_into_btree_array_avx2(word_t **xs, size_t n, word_t *target) {
    constexpr size_t size = 2 << circumscribed;
    __m256i arr [size];
    __m256i rs [circumscribed];
    size_t lows [size];
    size_t highs [size];
    uint8_t to;
    float correction;

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        for (size_t i = 0; i < circumscribed; ++i)
            rs[i] = avx2_pcg32_random_r(&avx2_key);
        for (size_t i = size - 1; i > 0; --i) {
            if (size - n <= i) {
                arr[i] = _mm256_loadu_si256((__m256i *) (xs[size - i - 1] + word_id));
                lows[i] = size - i - 1;
                highs[i] = size - i;
            } else if (2*i + 1 < size) {
                if (lows[2*i] != 0) {
                    // Not sure if we should intermix this with the AVX logic or not, likely dependent on `n`
                    size_t da = highs[2*i] - lows[2*i];
                    size_t db = highs[2*i + 1] - lows[2*i + 1];

                    lows[i] = lows[2*i + 1];
                    highs[i] = highs[2*i];

                    uint64_t instr = instruction_upto((float)db/(float)(da + db), &to, &correction, 0.002);

                    size_t sp = 62 - __builtin_clzll(2*i + 1);
                    __m256i cond = rs[sp];

                    for (uint8_t i = to - 1; i < to; --i) {
                        if ((instr & (1ULL << i)) >> i)
                            cond = _mm256_or_si256(cond, avx2_pcg32_random_r(&avx2_key));
                        else
                            cond = _mm256_and_si256(cond, avx2_pcg32_random_r(&avx2_key));
                    }

                    __m256i when1 = arr[2*i + 1];
                    __m256i when0 = arr[2*i];
                    arr[i] = when0 ^ (cond & (when0 ^ when1));
                } else {
                    lows[i] = lows[2*i + 1];
                    highs[i] = highs[2*i + 1];
                    arr[i] = arr[2*i + 1];
                }
            }
        }
        _mm256_storeu_si256((__m256i *) (target + word_id), arr[1]);
    }
}

__m256i random32_range(int range) {
    __m256i bits = avx2_pcg32_random_r(&avx2_key);
    // Convert random bits into FP32 number in [ 1 .. 2 ) interval
    __m256i mantissa_mask = _mm256_set1_epi32(0x7FFFFF);
    __m256i mantissa = _mm256_and_si256(bits, mantissa_mask);
    __m256 one = _mm256_set1_ps(1);
    __m256 val = _mm256_or_ps(_mm256_castsi256_ps(mantissa), one);

    // Scale the number from [ 1 .. 2 ) into [ 0 .. range ), via ( val * range ) - range
    __m256 rf = _mm256_set1_ps((float_t)range);
    val = _mm256_fmsub_ps(val, rf, rf);

    return _mm256_cvttps_epi32(val);
}

void representative_into_counters_avx2(word_t ** xs, uint32_t size, word_t* dst) {
    const __m256i signed_compare_adjustment = _mm256_set1_epi32(0x80000000);
    uint8_t* dst_bytes = (uint8_t*)dst;
    uint32_t counters[BITS];

    //Clear out the counters
    memset(counters, 0, BITS * sizeof(uint32_t));

    //Loop over all input vectors, 255 at a time
    for (size_t i = 0; i < size; i += 255) {
        size_t num_inputs = (i<size-254)? 255: size-i;

        size_t cur_counters = 0;
        for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 32) {

            //Call (inline) the function to load half a cache line of input bits from each input hypervector
            __m256i scrambled_counts[8];
            count_half_cacheline_for_255_input_hypervectors_avx2(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m256i out_counts[8];
            unscramble_byte_counters_avx2(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 32-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m128i converted0 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[0]);
                __m128i converted1 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[1]);
                __m128i converted2 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[2]);
                __m128i converted3 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[3]);

                __m256i increment0 = _mm256_cvtepu8_epi32((__m128i) converted0);
                __m256i increment1 = _mm256_cvtepu8_epi32((__m128i) converted1);
                __m256i increment2 = _mm256_cvtepu8_epi32((__m128i) converted2);
                __m256i increment3 = _mm256_cvtepu8_epi32((__m128i) converted3);
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 0]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 0]), increment0));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 8]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 8]), increment1));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 16]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 16]), increment2));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 24]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 24]), increment3));
                cur_counters += 32;
            }
        }
    }

    //Now do thresholding, and output the final bits
    for (size_t i = 0; i < BITS/8; i++) {
        __m256i threshold_simd = _mm256_sub_epi32(random32_range(size), signed_compare_adjustment);
        __m256i adjusted_counters = _mm256_sub_epi32(*(__m256i*)(&counters[i * 8]), signed_compare_adjustment);
        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi32(adjusted_counters, threshold_simd);
        uint8_t maj_bytes;
        maj_bytes = (uint8_t)_pext_u64(maj_words[0], 0x0000000100000001) |
            (uint8_t)_pext_u64(maj_words[1], 0x0000000100000001) << 2 |
            (uint8_t)_pext_u64(maj_words[2], 0x0000000100000001) << 4 |
            (uint8_t)_pext_u64(maj_words[3], 0x0000000100000001) << 6;

        *((uint16_t*)(dst_bytes + i)) = maj_bytes;
    }
}

// uselessly biased for low number of range
__m256i random8_fastrange_avx2(uint8_t range) {
    __m256i r = _mm256_set1_epi16(range);

    __m256i sample = avx2_pcg32_random_r(&avx2_key);

    __m128i l = _mm256_extractf128_si256(sample, 0);
    __m128i h = _mm256_extractf128_si256(sample, 1);

    __m256i lm = _mm256_cvtepu8_epi16(l);
    __m256i hm = _mm256_cvtepu8_epi16(h);

    __m256i lr = _mm256_mullo_epi16(lm, r);
    __m256i hr = _mm256_mullo_epi16(hm, r);

    // lr = _mm256_srli_epi16(lr, 8); //d
    // hr = _mm256_srli_epi16(hr, 8);

    // __m128i lp = _mm256_cvtepi16_epi8(lr);
    // __m128i hp = _mm256_cvtepi16_epi8(hr);

    // __m256i out = _mm256_set_m128i(lp, hp);


    lr = _mm256_srli_epi16(lr, 8);
    __m256i m = _mm256_set_epi8(0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff);
    __m256i out = _mm256_blendv_epi8(hr, lr, m);

    return out;
}

// uselessly slow
__m256i random8_range_avx2(uint8_t range) {
    __m256i sample;

    uint8_t k = 64 - __builtin_clzll(range - 1);
    uint8_t M = (1 << k) - ((1 << k) % range);

    __m256i M_simd = _mm256_set1_epi8(M);
    __m256i onetwentyeight_simd = _mm256_set1_epi8(128);
    __m256i range_simd = _mm256_set1_epi8(range + 128);

    __m256i out = _mm256_set1_epi8(0);
    uint32_t mask = 0;

    while (mask != 0xffffffff) {
        sample = avx2_pcg32_random_r(&avx2_key);

        __m256i lt = _mm256_cmpgt_epi8(range_simd, _mm256_sub_epi8(sample, onetwentyeight_simd));

        mask |= _mm256_movemask_epi8(lt);

        // could be sped up a lot by using modulo rejection sampling, especially for very low range
        // __m256i valid = compiletime_rem_epi8_avx2(sample, range); // a lookup table of functions with fixed divisor
        // __m256i valid = _mm256_rem_epi8(sample, range_simd); // this is only provided by the Intel compiler and not even that much faster than the above

        // this is stupendously wasteful, because here the *position* matters, not just if it was successfully generated
        // maybe https://www.felixcloutier.com/x86/vpmultishiftqb to the rescue
        out = _mm256_blendv_epi8(out, sample, lt);
    }

    return out;
}

// quite slow
__m256i random8_range_emulated(uint8_t range) {
    static std::uniform_int_distribution<uint8_t> distr(0, range - 1);

    return _mm256_set_epi8(
        distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng),
        distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng),
        distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng),
        distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng), distr(rng));
}

void representative_into_byte_counters_avx2(word_t **xs, size_t size, word_t *target) {
    const __m256i one_twenty_eight = _mm256_set1_epi8((char)128);
    uint8_t* dst_bytes = (uint8_t*)target;

    for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 32) {

        //Call (inline) the function to load one cache line of input bits from each input hypervector
        __m256i scrambled_counts[8];
        count_half_cacheline_for_255_input_hypervectors_avx2(xs, byte_offset, size, scrambled_counts);

        //Unscramble the counters
        __m256i out_counts[8];
        unscramble_byte_counters_avx2(scrambled_counts, out_counts);

        //Do the threshold test, and compose out output bits
        __m256i out_bits;
        for (int i=0; i<8; i++) {
            __m256i rand_thresholds = random8_range_emulated(size);
            rand_thresholds = _mm256_sub_epi32(rand_thresholds, one_twenty_eight); // offset that used to be one threshold_simd

            __m256i adjusted_counts = _mm256_sub_epi8(out_counts[i], one_twenty_eight);
            __m256i maj_bits_vec = _mm256_cmpgt_epi8(adjusted_counts, rand_thresholds);
            __mmask32 maj_bits = _mm256_movemask_epi8(maj_bits_vec);
            ((uint32_t*)&out_bits)[i] = maj_bits;
        }

        //Store the results
        _mm256_storeu_si256((__m256i*)(dst_bytes + byte_offset), out_bits);
    }
}
#endif


//word_t * n_representatives_impl(word_t ** xs, size_t size) {
//    word_t * x = zero();
//
//    std::uniform_int_distribution<size_t> gen(0, size - 1);
//    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
//        word_t word = 0;
//        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
//            size_t x_id = gen(rng);
//            word |=  1UL << (xs[x_id][word_id] >> bit_id) & 1;
//        }
//        x[word_id] = word;
//    }
//
//    return x;
//}

/// @brief For each dimension, samples a bit from xs into target
void representative_into_reference(word_t **xs, size_t size, word_t *target) {
    if (size == 0) rand_into(target);
    else if (size == 1) memcpy(target, xs[0], BYTES);
    else if (size == 2) { word_t r[WORDS]; rand_into(r); select_into(r, xs[0], xs[1], target); }
    else representative_into_bitcount_reference(xs, size, target);
}


#if __AVX2__
void representative_into_avx2(word_t **xs, size_t size, word_t *target) {
    if (size == 0) rand_into(target);
    else if (size == 1) memcpy(target, xs[0], BYTES);
    else if (size == 2) { word_t r[WORDS]; rand_into(r); select_into(r, xs[0], xs[1], target); }
    else if (size == 3) representative_into_recursive_avx2<3>(xs, target);
    else if (size == 4) representative_into_recursive_avx2<4>(xs, target);
    else if (size == 5) representative_into_recursive_avx2<5>(xs, target);
    else if (size == 6) representative_into_recursive_avx2<6>(xs, target);
    else if (size == 7) representative_into_recursive_avx2<7>(xs, target);
    else if (size == 8) representative_into_recursive_avx2<8>(xs, target);
    else if (size == 9) representative_into_recursive_avx2<9>(xs, target);
    else if (size == 10) representative_into_recursive_avx2<10>(xs, target);
    else if (size == 11) representative_into_recursive_avx2<11>(xs, target);
    else if (size == 12) representative_into_recursive_avx2<12>(xs, target);
    else if (size == 13) representative_into_recursive_avx2<13>(xs, target);
    else if (size == 14) representative_into_recursive_avx2<14>(xs, target);
    else if (size == 15) representative_into_recursive_avx2<15>(xs, target);
    else if (size == 16) representative_into_recursive_avx2<16>(xs, target);
    else if (size == 17) representative_into_recursive_avx2<17>(xs, target);
    else if (size == 18) representative_into_recursive_avx2<18>(xs, target);
    else if (size == 19) representative_into_recursive_avx2<19>(xs, target);
    else if (size == 20) representative_into_recursive_avx2<20>(xs, target);
    else if (size == 21) representative_into_recursive_avx2<21>(xs, target);
    else if (size == 22) representative_into_recursive_avx2<22>(xs, target);
    else if (size == 23) representative_into_recursive_avx2<23>(xs, target);
    else if (size == 24) representative_into_recursive_avx2<24>(xs, target);
    else if (size == 25) representative_into_recursive_avx2<25>(xs, target);
    else if (size == 26) representative_into_recursive_avx2<26>(xs, target);
    else if (size == 27) representative_into_recursive_avx2<27>(xs, target);
    else if (size == 28) representative_into_recursive_avx2<28>(xs, target);
    else if (size == 29) representative_into_recursive_avx2<29>(xs, target);
    else if (size == 30) representative_into_recursive_avx2<30>(xs, target);
    else if (size == 31) representative_into_recursive_avx2<31>(xs, target);
    else if (size == 32) representative_into_recursive_avx2<32>(xs, target);
    else if (size <= 255) representative_into_byte_counters_avx2(xs, size, target);
    else if (size <= 2000) representative_into_counters_avx2(xs, size, target);
    else representative_into_bitcount_reference(xs, size, target);
}
#endif

#if __AVX2__
#define representative_into representative_into_avx2
#else
#define representative_into representative_into_reference
#endif
