void window_into_reference(word_t **xs, size_t size, size_t b, size_t t, word_t *target) {
    word_t b_hv [WORDS];
    word_t t_hv [WORDS];

    threshold_into(xs, size, b - 1, b_hv);
    threshold_into(xs, size, t, t_hv);
    invert_into(t_hv, t_hv);

    and_into(b_hv, t_hv, target);
}

#define window_into window_into_reference