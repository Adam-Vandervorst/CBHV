constexpr uint32_t smod(int32_t x, uint32_t m) {
    return ((x % m) + m) % m;
}

void swap_even_odd_into_reference(word_t* x, word_t* target) {
    for (size_t i = 0; i < WORDS; ++i)
        target[i] = ((x[i] & EVEN_WORD) << 1) | ((x[i] & ODD_WORD) >> 1);
}

// TODO can be sped up drastically using GFNI with an exchange matrix
void swap_bits_of_byte_order_into_reference(word_t* x, word_t* target) {
    uint8_t* x_bytes = (uint8_t*)x;
    uint8_t* target_bytes = (uint8_t*)target;

    static uint8_t table[] = {
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
        0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
        0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
        0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
        0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
        0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
        0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
        0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
        0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
        0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
        0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
        0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
        0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
        0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
        0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
        0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
        0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
        0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
        0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
        0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
        0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
        0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
        0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
        0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
        0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
        0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
        0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
        0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
        0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
        0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
        0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
    };

    for (bit_iter_t i = 0; i < BYTES; ++i)
        target_bytes[i] = table[x_bytes[i]];
}

// TODO this and the following can be sped up using AVX-2 shuffle epi8 or using AVX-512 permutexvar epi8
void swap_byte_order_inplace(word_t* x) {
    uint8_t* x_bytes = (uint8_t*)x;

    for (bit_iter_t i = 0; i < BYTES / 2; ++i) {
        uint8_t tmp = x_bytes[i];
        x_bytes[i] = x_bytes[BYTES - i - 1];
        x_bytes[BYTES - i - 1] = tmp;
    }
}

void swap_byte_order_into(word_t* x, word_t* target) {
    if (x == target)
        return swap_byte_order_inplace(x);

    uint8_t* x_bytes = (uint8_t*)x;
    uint8_t* target_bytes = (uint8_t*)target;

    for (bit_iter_t i = 0; i < BYTES; ++i)
        target_bytes[i] = x_bytes[BYTES - 1 - i];
}

#if __GFNI__
void swap_even_odd_into_gfni(word_t* x, word_t* target) {
    for (size_t i = 0; i < DIMENSION/512; ++i) {
        __m512i matrix = _mm512_set1_epi64(0b0000001000000001000010000000010000100000000100001000000001000000);
        __m512i res = _mm512_gf2p8affine_epi64_epi8(_mm512_loadu_si512((__m512i*)x + i), matrix, 0);
        _mm512_storeu_si512((__m512i*)target + i, res);
    }
}
#endif

#if __GFNI__
#define swap_even_odd_into swap_even_odd_into_gfni
#else
#define swap_even_odd_into swap_even_odd_into_reference
#endif

void roll_bytes_into(word_t *x, int32_t d, word_t *target) {
    assert(x != target);
    if (d == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    uint8_t *x_bytes = (uint8_t *)x;
    uint8_t *target_bytes = (uint8_t *)target;
    int32_t offset = smod(d, BYTES);

    memcpy(target_bytes, x_bytes + offset, BYTES - offset);
    memcpy(target_bytes + BYTES - offset, x_bytes, offset);
}

void roll_words_into(word_t *x, int32_t d, word_t *target) {
    assert(x != target);
    int32_t offset = smod(d, WORDS);

    memcpy(target, x + offset, (WORDS - offset) * sizeof(word_t));
    memcpy(target + WORDS - offset, x, offset * sizeof(word_t));
}

void roll_word_bits_into(word_t *x, int32_t d, word_t *target) {
    int32_t offset = smod(d, BITS_PER_WORD);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[i] = std::rotl(x[i], offset);
    }
}

void roll_bits_into_reference(word_t *x, int32_t o, word_t *target) {
    int32_t offset = smod(o, BITS);

    bool cx [BITS];
    bool ctarget [BITS];

    unpack_into(x, cx);

    memcpy(ctarget, cx + offset, BITS - offset);
    memcpy(ctarget + BITS - offset, cx, offset);

    pack_into(ctarget, target);
}

void roll_bits_into_single_pass(word_t *x, int32_t o, word_t *target) {
    assert(x != target);
    if (o < 0) {
        roll_bits_into_single_pass(x, BITS + o, target);
        return;
    }
    if (o == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    // say BITS_PER_WORD=4 and WORDS=3
    // roll 5 would look like
    // x  = abcd|efgh|ijkl|
    // l  = ijkl|abcd|efgh|   roll 1 word
    // h  = efgh|ijkl|abcd|   roll 2 words
    // lo = -ijk|-abc|-efg|   shift right 1 bit
    // ho = h---|l---|d---|   shift left 4-1 bits
    // res  hijk|labc|defg|   union intermediates

    // TODO smod can be optimized out by splitting the loop on b

    int32_t b = o / BITS_PER_WORD;
    int32_t e = o % BITS_PER_WORD;
    word_t hmask = ~(0xffffffffffffffff >> e);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        word_t l = x[smod(b + i, WORDS)];
        word_t h = x[smod(b + i + 1, WORDS)];
        word_t lo = l >> e;
        word_t ho = h << (BITS_PER_WORD - e);
        word_t res = lo | (ho & hmask);
        target[i] = res;
    }
}

#define roll_bits_into roll_bits_into_single_pass

uint8_t permute_single_byte_bits(uint8_t x, uint64_t p) {
    uint64_t w = _pdep_u64(x, 0x0101010101010101);
    uint64_t res = (uint64_t) _mm_shuffle_pi8(_mm_cvtsi64_m64(w), _mm_cvtsi64_m64(p));
    return _pext_u64(res, 0x0101010101010101);
}

uint64_t byte_bits_permutation_invert(uint64_t p) {
    uint64_t r = 0;

    for (uint64_t i = 0; i < 8; ++i)
        r |= i << (((p >> (i * 8)) & 0x07) * 8);

    return r;
}

uint64_t rand_byte_bits_permutation(int64_t seed) {
    std::mt19937_64 perm_rng(seed);

    uint8_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::shuffle(p, p + 8, perm_rng);

    return *((uint64_t *) p);
}

void permute_byte_bits_into_shuffle(word_t *x, int64_t perm, word_t *target) {
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    uint8_t *x_bytes = (uint8_t *) x;
    uint8_t *target_bytes = (uint8_t *) target;

    uint64_t byte_perm = rand_byte_bits_permutation(abs(perm));
    if (perm < 0) byte_perm = byte_bits_permutation_invert(byte_perm);

    for (byte_iter_t i = 0; i < BYTES; ++i)
        target_bytes[i] = permute_single_byte_bits(x_bytes[i], byte_perm);
}

uint64_t byte_bits_permutation_matrix(uint64_t packed_indices) {
    uint64_t r = 0;

    for (uint8_t i = 0; i < 64; i += 8)
        r |= 1ULL << ((56 - i) + ((packed_indices >> i) & 0x07));

    return r;
}

#if __GFNI__
void permute_byte_bits_into_gfni(word_t *x, int64_t perm, word_t *target) {
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    __m512i *x_vec = (__m512i *) x;
    __m512i *target_vec = (__m512i *) target;

    uint64_t byte_perm = rand_byte_bits_permutation(abs(perm));
    if (perm < 0) byte_perm = byte_bits_permutation_invert(byte_perm);
    uint64_t byte_perm_matrix = byte_bits_permutation_matrix(byte_perm);
    __m512i byte_perm_matrices = _mm512_set1_epi64(byte_perm_matrix);

    for (word_iter_t i = 0; i < BITS/512; ++i) {
        __m512i vec = _mm512_loadu_si512(x_vec + i);
        __m512i permuted_vec = _mm512_gf2p8affine_epi64_epi8(vec, byte_perm_matrices, 0);
        _mm512_storeu_si512(target_vec + i, permuted_vec);
    }
}
#endif

#if __GFNI__
#define permute_byte_bits_into permute_byte_bits_into_gfni
#else
#define permute_byte_bits_into permute_byte_bits_into_shuffle
#endif

#if __AVX512BW__
uint64_t permute_single_word_bits(uint64_t x, __m512i p) {
    __m512i x_simd = _mm512_set1_epi64(x);
    __mmask64 permuted_bits = _mm512_bitshuffle_epi64_mask(x_simd, p);
    return _cvtmask64_u64(permuted_bits);
}

__m512i word_bits_permutation_invert(__m512i p) {
    uint8_t p_array [64];
    uint8_t r_array [64];
    _mm512_storeu_si512(p_array, p);

    for (uint8_t i = 0; i < 64; ++i)
        r_array[p_array[i] & 0x3f] = i;

    return _mm512_loadu_si512(r_array);
}

__m512i rand_word_bits_permutation(int64_t seed) {
    std::mt19937_64 perm_rng(seed);

    uint8_t p [64];
    for (uint8_t i = 0; i < 64; ++i)
        p[i] = i;

    std::shuffle(p, p + 64, perm_rng);

    return _mm512_loadu_si512(p);
}

void permute_word_bits_into(word_t * x, int64_t perm, word_t * target) {
    if (perm == 0) {memcpy(target, x, BYTES); return;}

    __m512i word_perm = rand_word_bits_permutation(abs(perm));
    if (perm < 0) word_perm = word_bits_permutation_invert(word_perm);

    for (word_iter_t i = 0; i < WORDS; ++i)
        target[i] = permute_single_word_bits(x[i], word_perm);
}
#endif

template<bool inverse>
void apply_word_permutation_into(word_t *x, word_iter_t *word_permutation, word_t *target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        if constexpr (inverse)
            target[word_permutation[i]] = x[i];
        else
            target[i] = x[word_permutation[i]];
    }
}

void rand_word_permutation_into(int64_t seed, word_iter_t *p) {
    std::mt19937_64 perm_rng(seed);

    for (word_iter_t i = 0; i < WORDS; ++i)
        p[i] = i;

    std::shuffle(p, p + WORDS, perm_rng);
}

void permute_words_into(word_t *x, int64_t perm, word_t *target) {
    assert(x != target);
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    word_iter_t p[WORDS];
    rand_word_permutation_into(abs(perm), p);

    if (perm > 0) apply_word_permutation_into<false>(x, p, target);
    else apply_word_permutation_into<true>(x, p, target);
}


template<bool inverse>
void apply_byte_permutation_into(word_t *x, byte_iter_t *byte_permutation, word_t *target) {
    uint8_t *x_bytes = (uint8_t *) x;
    uint8_t *target_bytes = (uint8_t *) target;

    for (byte_iter_t i = 0; i < BYTES; ++i) {
        if constexpr (inverse)
            target_bytes[byte_permutation[i]] = x_bytes[i];
        else
            target_bytes[i] = x_bytes[byte_permutation[i]];
    }
}

void rand_byte_permutation_into(int64_t seed, byte_iter_t *p) {
    std::mt19937_64 perm_rng(seed);

    for (byte_iter_t i = 0; i < BYTES; ++i)
        p[i] = i;

    std::shuffle(p, p + BYTES, perm_rng);
}

void permute_bytes_into(word_t *x, int64_t perm, word_t *target) {
    assert(x != target);
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    byte_iter_t p[BYTES];
    rand_byte_permutation_into(abs(perm), p);

    if (perm > 0) apply_byte_permutation_into<false>(x, p, target);
    else apply_byte_permutation_into<true>(x, p, target);
}

void permute_into(word_t *x, int64_t perm, word_t *target) {
#if __AVX512BW__
    permute_words_into(x, perm, target);
    permute_word_bits_into(target, perm, target);
#else
    permute_bytes_into(x, perm, target);
    permute_byte_bits_into(target, perm, target);
#endif
}

word_t * permute(word_t *x, int64_t perm) {
    word_t *r = empty();
#if __AVX512BW__
    permute_words_into(x, perm, r);
    permute_word_bits_into(r, perm, r);
#else
    permute_bytes_into(x, perm, r);
    permute_byte_bits_into(r, perm, r);
#endif
    return r;
}
