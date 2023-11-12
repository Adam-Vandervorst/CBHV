size_t closest_reference(word_t **xs, size_t size, word_t *x) {
    bit_iter_t distance = BITS + 1;
    size_t index = -1;

    for (size_t i = 0; i < size; ++i) {
        bit_iter_t d = hamming(xs[i], x);
        if (d < distance) {
            distance = d;
            index = i;
        }
    }

    return index;
}

