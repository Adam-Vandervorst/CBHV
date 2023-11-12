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


#ifdef _OPENMP
size_t closest_omp(word_t **xs, size_t size, word_t *x) {
    bit_iter_t distance = BITS + 1;
    size_t index = -1;

#pragma omp parallel
    {
        bit_iter_t local_distance = BITS + 1;
        size_t local_index = -1;

#pragma omp for nowait
        for (size_t i = 0; i < size; ++i) {
            bit_iter_t d = hamming(xs[i], x);
            if (d < local_distance) {
                local_distance = d;
                local_index = i;
            }
        }

#pragma omp critical
        {
            if (local_distance < distance) {
                distance = local_distance;
                index = local_index;
            }
        }
    }

    return index;
}
#else
size_t closest_execution(word_t **xs, size_t size, word_t *x) {
    bit_iter_t *distances = (bit_iter_t *)malloc(sizeof(bit_iter_t) * size);

    std::transform(std::execution::par_unseq, xs, xs + size, distances, [&x](word_t * x_) { return hamming(x, x_); });
    size_t res = std::distance(distances, std::min_element(std::execution::par_unseq, distances, distances + size));

    free(distances);

    return res;
}
#endif

size_t closest(word_t **xs, size_t size, word_t *x) {
    if (size < 1000)
        return closest_reference(xs, size, x);
    else
#ifdef _OPENMP
        return closest_omp(xs, size, x);
#else
        return closest_execution(xs, size, x);
#endif
}
