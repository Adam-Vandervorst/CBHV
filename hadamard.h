constexpr word_t HADAMARD_WORD [6] = {
        0xaaaaaaaaaaaaaaaa,
        0xcccccccccccccccc,
        0xf0f0f0f0f0f0f0f0,
        0xff00ff00ff00ff00,
        0xffff0000ffff0000,
        0xffffffff00000000,
};

template<uint32_t f, word_t one, word_t zero>
word_t *const_word_freq() {
    static word_t x[WORDS];
    for (size_t i = 0; i < WORDS; ++i)
        x[i] = (i % (2 << f)) < (1 << f) ? one : zero;
    return x;
}

word_t *walsh_basis[13] = {
        const_bhv<bhv::HADAMARD_WORD[0]>(),
        const_bhv<bhv::HADAMARD_WORD[1]>(),
        const_bhv<bhv::HADAMARD_WORD[2]>(),
        const_bhv<bhv::HADAMARD_WORD[3]>(),
        const_bhv<bhv::HADAMARD_WORD[4]>(),
        const_bhv<bhv::HADAMARD_WORD[5]>(),

        const_word_freq<0, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<1, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<2, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<3, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<4, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<5, 0ULL, bhv::ONE_WORD>(),
        const_word_freq<6, 0ULL, bhv::ONE_WORD>()
};

void hadamard_encode_into_reference(uint32_t w, word_t *target) {
    bool bits [BITS];

    for (uint32_t i = 0; i < BITS; ++i) {
        bits[i] = std::popcount(w & i) % 2 == 0;
    }

    pack_into(bits, target);
}

void hadamard_encode_into_basis(uint32_t w, word_t *target) {
    if (w == 0) {
        memset(target, -1, BYTES);
        return;
    }

    word_t *to_combine[13];
    uint32_t gathered = 0;
    for (uint32_t i = 0; i < 13; ++i)
        if (w & (1 << i))
            to_combine[gathered++] = walsh_basis[i];

    parity_into(to_combine, gathered, target);
    invert_into(target, target);
}

#define hadamard_encode_into hadamard_encode_into_basis

void hadamard_transform_into(word_t *x, uint32_t *factors) {
    for (uint32_t i = 0; i < BITS; ++i) {
        word_t tmp [WORDS];
        hadamard_encode_into(i, tmp);
        factors[i] = hamming(tmp, x);
    }
}

uint32_t hadamard_decode_reference(word_t *x) {
    uint32_t walsh_spectrum [BITS];

    hadamard_transform_into(x, walsh_spectrum);

    return std::distance(walsh_spectrum, std::min_element(walsh_spectrum, walsh_spectrum + BITS));
}

#define hadamard_decode hadamard_decode_reference