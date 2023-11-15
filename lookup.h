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
    if (size < 2000)
        return closest_reference(xs, size, x);
    else
#ifdef _OPENMP
        return closest_omp(xs, size, x);
#else
        return closest_execution(xs, size, x);
#endif
}

#include "pq.h"

struct Element {
    size_t index;
    bit_iter_t distance;

    bool operator<(const Element &rhs) const {
        return distance > rhs.distance;
    }

    bool operator>(const Element &rhs) const {
        return rhs < *this;
    }

    bool operator<=(const Element &rhs) const {
        return !(rhs < *this);
    }

    bool operator>=(const Element &rhs) const {
        return !(*this < rhs);
    }
};

void top_into_reference(word_t **xs, size_t size, word_t *x, size_t k, size_t* target) {
    assert(size >= k);

    fixed_size_priority_queue<Element> heap(k);

    for (size_t i = 0; i < size; ++i)
        heap.push(Element{i, hamming(xs[i], x)});

    for (size_t i = 0; i < k; ++i) {
        target[i] = heap.top().index;
        heap.pop();
    }
}

#ifdef _OPENMP
void top_into_omp(word_t **xs, size_t size, word_t *x, size_t k, size_t* target) {
    assert(size >= k);

    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::vector<fixed_size_priority_queue<Element>> heaps;
    for (size_t t = 0; t < num_threads; ++t)
        heaps.push_back(fixed_size_priority_queue<Element>(k));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        fixed_size_priority_queue<Element>& heap = heaps[thread_id];

        #pragma omp for nowait
        for (size_t i = 0; i < size; ++i) {
            int dist = hamming(xs[i], x);
            heap.push(Element{i, dist});
        }
    }

    // Merge heaps
    fixed_size_priority_queue<Element> final_heap(k);
    for (auto& heap : heaps) {
        while (!heap.empty()) {
            final_heap.push(heap.top());
            heap.pop();
        }
    }

    // Extract final top k elements
    for (size_t i = 0; i < k; ++i) {
        target[i] = final_heap.top().index;
        final_heap.pop();
    }
}
#else
void top_into_execution_sort(word_t **xs, size_t size, word_t *x, size_t k, size_t* target) {
    assert(size >= k);

    bit_iter_t *distances = (bit_iter_t *)malloc(sizeof(bit_iter_t) * size);
    size_t *indices = (size_t *)malloc(sizeof(size_t) * size);

    std::iota(indices, indices + size, 0);

    std::transform(std::execution::par_unseq, xs, xs + size, distances, [x](word_t * x_) { return hamming(x, x_); });

    std::partial_sort_copy(indices, indices + size, target, target + k, [&distances](size_t i, size_t j) { return distances[i] < distances[j]; });
}
#endif

void top_into(word_t **xs, size_t size, word_t *x, size_t k, size_t* target) {
    if (size < 2000)
        return top_into_reference(xs, size, x, k, target);
    else
#ifdef _OPENMP
        return top_into_omp(xs, size, x, k, target);
#else
        return top_into_execution_sort(xs, size, x, k, target);
#endif
}

std::vector<size_t> within_reference(word_t **xs, size_t size, word_t *x, bit_iter_t d) {
    std::vector<size_t> ys;

    for (size_t i = 0; i < size; ++i)
        if (hamming(xs[i], x) <= d)
            ys.push_back(i);

    return ys;
}

#ifdef _OPENMP
std::vector<size_t> within_omp(word_t **xs, size_t size, word_t *x, bit_iter_t d) {
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::vector<std::vector<size_t>> ys_private(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < size; ++i) {
            if (hamming(xs[i], x) <= d) {
                ys_private[thread_id].push_back(i);
            }
        }
    }

    // Calculate total size to reserve memory
    size_t total_size = 0;
    for (const auto& vec : ys_private) {
        total_size += vec.size();
    }

    std::vector<size_t> ys;
    ys.reserve(total_size);

    // Merge all vectors into one
    for (const auto& vec : ys_private) {
        ys.insert(ys.end(), vec.begin(), vec.end());
    }

    return ys;
}
#else
std::vector<size_t> within_execution(word_t **xs, size_t size, word_t *x, bit_iter_t d) {
    std::vector<size_t> indices(size);

    std::iota(indices.begin(), indices.end(), 0);

    indices.erase(std::remove_if(std::execution::par_unseq, indices.begin(), indices.end(), [xs, x, d](size_t i){ return hamming(x, xs[i]) > d; }),
                  indices.end());

    return indices;
}
#endif

std::vector<size_t> within(word_t **xs, size_t size, word_t *x, bit_iter_t d) {
    if (size < 2000)
        return within_reference(xs, size, x, d);
    else
#ifdef _OPENMP
        return within_omp(xs, size, x, d);
#else
        return within_execution(xs, size, x, d);
#endif
}
