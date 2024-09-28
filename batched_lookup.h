template<size_t y_size>
void batched_closest_reference(word_t **xs, size_t size, word_t **ys, size_t *indices) {
    bit_iter_t distances[y_size];
    std::fill_n(distances, y_size, BITS + 1);

    for (size_t i = 0; i < size; ++i) {
        word_t *x = xs[i];
        for (size_t j = 0; j < y_size; ++j) {
            bit_iter_t d = hamming(x, ys[j]);
            if (d < distances[j]) {
                distances[j] = d;
                indices[j] = i;
            }
        }
    }
}

extern "C" void batched_closest(word_t **xs, size_t size, word_t **ys, size_t *indices, size_t batch_size) {
    switch (batch_size) {
        case 1: batched_closest_reference<1>(xs, size, ys, indices); break;
        case 2: batched_closest_reference<2>(xs, size, ys, indices); break;
        case 4: batched_closest_reference<4>(xs, size, ys, indices); break;
        case 8: batched_closest_reference<8>(xs, size, ys, indices); break;
        case 16: batched_closest_reference<16>(xs, size, ys, indices); break;
        case 32: batched_closest_reference<32>(xs, size, ys, indices); break;
        case 64: batched_closest_reference<64>(xs, size, ys, indices); break;
        case 128: batched_closest_reference<128>(xs, size, ys, indices); break;
        case 256: batched_closest_reference<256>(xs, size, ys, indices); break;
    }
}
