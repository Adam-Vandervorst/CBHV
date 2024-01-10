word_t** load_pbm(FILE* file, size_t* n_elements) {
    char header[2];
    fread(header, 1, 2, file);
    assert(header[0] == 'P' && header[1] == '4');

    int dimension, n;
    fscanf(file, "%d %d", &dimension, &n);
    assert(dimension == BITS);

    *n_elements = n;
    word_t **hvs = (word_t **)malloc(n*sizeof(word_t *));

    for (int i = 0; i < n; ++i) {
        hvs[i] = empty();
        fread(hvs[i], 1, BYTES, file);
    }

    return hvs;
}


void save_pbm(FILE* file, word_t** data, size_t n_elements) {
    // white are off bits
    fwrite("P4", 1, 2, file);

    fprintf(file, "\n%d %zu\n", BITS, n_elements);

    for (size_t i = 0; i < n_elements; ++i)
        fwrite(data[i], 1, BYTES, file);
}


std::string to_string(uint64_t *x) {
    char viz [DIMENSION + 1];
    viz[DIMENSION] = '\0';
    unpack_into(x, (bool*)viz);
    for (size_t i = 0; i < DIMENSION; ++i)
        viz[i] += '0';
    return std::string(viz);
}
