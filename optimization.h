namespace opt {

template <typename N>
N independent(word_t* x, std::function<N (word_t*)> loss) {
    N best_l = loss(x);
    for (bit_iter_t i = 0; i < BITS; ++i) {
        toggle(x, i);
        N l = loss(x);
        if (l > best_l)
            toggle(x, i);
        else
            best_l = l;
    }
    return best_l;
}

template <typename N, double U, double L, uint64_t half_time>
float_t time_based_decay(uint64_t time_step, N loss) {
    double decay_rate = log((U - L)/(U + L))/(double)half_time;
    return L + (U - L)*exp(decay_rate*(double)time_step);
}

const double tau = 6.283185307179586;

template <typename N, double U, double L, uint64_t period>
float_t cyclic(uint64_t time_step, N loss) {
    return L + (U - L)*(sin(time_step*tau/(double)period)/2 + .5);
}

#define DEBUG 0
// WIP
template <typename N, double initial, double a, double ai, double b>
float_t success_based_narrowing(uint64_t time_step, N loss) {
    static N last_loss;
    static double UB_improvement;
    static double LB_improvement;
    static double M_improvement;
    static double UB;
    static double LB;
    static double M;

    double c = .5;
    double delta = (double)last_loss - (double)loss;
    double log_delta = 1./(1. + std::exp(-delta));

    switch (time_step % 3) {
        case 0:
            if (time_step < 3) {
                M = initial;
                M_improvement = 1.;
            } else {
                M_improvement = c*(ai*M_improvement + (1. - ai)*log_delta);
            }
            break;
        case 1:
            if (time_step < 3) {
                UB = initial + b;
                UB_improvement = 1.;
            } else {
                UB_improvement = c*(ai*UB_improvement + (1. - ai)*log_delta);
            }
            break;
        case 2:
            if (time_step < 3) {
                LB = initial - b;
                LB_improvement = 1.;
            } else {
                LB_improvement = c*(ai*LB_improvement + (1. - ai)*log_delta);
            }
            break;
    }

    last_loss = loss;

    M = (UB + LB)/2.;

    double HV_EPS = 1. / (double) BITS;
    if (time_step % 3 == 0 and time_step != 0) {
        if (M_improvement > UB_improvement and M_improvement > LB_improvement and (UB - LB) > 2*HV_EPS) {
            if constexpr (DEBUG) std::cout << "N" << std::endl; // good, narrow
            UB = a*UB + (1 - a)*((UB + M)/2 - HV_EPS);
            LB = a*LB + (1 - a)*((LB + M)/2 + HV_EPS);
        } else if (UB_improvement > M_improvement and UB_improvement > LB_improvement) {
            if constexpr (DEBUG) std::cout << "U" << std::endl; // needs to be higher
            UB = a*UB + (1 - a)*(UB + (UB - M)/2 + HV_EPS);
        } else if (LB_improvement > M_improvement and LB_improvement > UB_improvement) {
            if constexpr (DEBUG) std::cout << "L" << std::endl; // needs to be lower
            LB = a*LB + (1 - a)*(LB - (M - LB)/2 - HV_EPS);
        } else {
            if constexpr (DEBUG) std::cout << "W" << std::endl; // bad, widen
            UB = a*UB + (1 - a)*(UB + (UB - M)/2 + HV_EPS);
            LB = a*LB + (1 - a)*(LB - (M - LB)/2 - HV_EPS);
        }

        UB = std::min(UB, .5);
        LB = std::max(LB, 0.);
        if (UB < LB) {
//            assert(false);
            double tmp = UB;
            UB = LB;
            LB = tmp;
        }
    }

    if constexpr (DEBUG) std::cout << "UB: " << UB << ", M: " << (UB + LB)/2. << ", LB: " << LB << std::endl;
    if constexpr (DEBUG) std::cout << "UBi: " << UB_improvement << ", Mi: " << M_improvement << ", LBi: " << LB_improvement << std::endl;
    if constexpr (DEBUG) std::cout << "D: " << delta << ", LD: " << log_delta << std::endl;

    switch (time_step % 3) {
        case 0: return UB;
        case 1: return LB;
        case 2: return (UB + LB)/2.;
    }
}

template <typename N>
N linear_search(word_t* x, std::function<N (word_t*)> loss,
                    uint64_t max_iter = 1000000,
                    N stagnant = 2*BITS,
                    std::function<float_t (uint64_t, N)> update = time_based_decay<N, 0.001, 0.0001, 100000>) {
    N best_l = loss(x);
    N l;
    word_t buf [WORDS];
    bool x_best = true;
    float_t delta = update(0, best_l);
    uint64_t last_improvement = 0;

    for (uint64_t it = 1; it <= max_iter; ++it) {
        if (x_best) {
            sparse_random_switch_into<-1>(x, delta, buf);
            l = loss(buf);
        } else {
            sparse_random_switch_into<-1>(buf, delta, x);
            l = loss(x);
        }

        if (l < best_l) {
            best_l = l;
            last_improvement = it;
            x_best = not x_best;
        } else if (it - last_improvement > stagnant) {
            break;
        }

        if (it % (max_iter/100) == 0) {
//            std::cout << std::endl << it << ": " << best_l << " (" << delta << ")" << std::endl;
        }

        delta = update(it, l);
    }

    if (not x_best)
        memcpy(x, buf, BYTES);

    return best_l;
}

// greater variability in learning rate -> create a new scheduler that does that
template <typename N, size_t K, size_t T = K/2 - 1>
N evolutionary_search(word_t* x, std::function<N (word_t*)> loss, uint64_t generations = 1000, uint64_t stripe = 50000) {
    static word_t candidates [K][WORDS];
    static N scores [K];
    static size_t indices [K];
    static word_t* top [T];

    std::iota(indices, indices + K, 0);

    for (size_t i = 0; i < K; ++i) {
        random_into(candidates[i], .1);
        xor_into(candidates[i], x, candidates[i]);
    }

    for (uint64_t gen = 0; gen < generations; ++gen) {
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c < K; ++c) {
            scores[c] = linear_search<N>(candidates[c], loss, stripe, stripe, cyclic<N, 5./(double)BITS, 1./(double)BITS, 3>);
        }

        std::sort(indices, indices + K, [](size_t i, size_t j){ return scores[i] < scores[j]; });

        std::cout << gen << " - best: " << scores[indices[0]] << ", worst: " << scores[indices[K - 1]] << std::endl;

        for (size_t i = 0; i < T; ++i) {
            top[i] = candidates[indices[i]];
        }

        for (size_t i = T; i < K; ++i) {
            representative_into(top, T, candidates[indices[i]]);
        }
    }

    memcpy(x, candidates[indices[0]], BYTES);

    return scores[indices[0]];
}

/*
        N best_l = loss(c);
        std::mt19937_64 perm_rng(gen*K + i);

        bit_iter_t p[BITS];
        std::iota(p, p + BITS, 0);
        std::shuffle(p, p + BITS, perm_rng);

        for (bit_iter_t i : p) {
            toggle(c, i);
            N l = loss(c);
            if (l > best_l)
                toggle(c, i);
            else
                best_l = l;
        }
 */

}