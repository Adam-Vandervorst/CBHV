#include <chrono>
#include <iostream>
#include <random>
#include <bitset>
#include <functional>
#include "core.h"

using namespace std;

const double tau = 6.283185307179586;

double index_to_angle(size_t i) { return tau*i/BITS; }
size_t angle_to_index(double a) { return (size_t)(BITS*a/tau); }

double angle_difference(double a1, double a2) {
    return 2*abs(sin((a1 - a2)/2.));
}

double phase_difference(double a1, double a2) {
    double ad = angle_difference(a1, a2);
    return (ad*ad)/4;
}

/*
FILE* ifile = fopen("pdm_circle.pbm", "rb");
size_t elements;
word_t** hvs = bhv::load_pbm(ifile, &elements);
fclose(ifile);

for (size_t i = 0; i < elements; ++i) {
    word_t tmp [WORDS];
    bhv::roll_bits_into(hvs[i], -i, tmp);
    cout << i << ": " << loss(tmp) << endl;
}
 */

void search_kernel(size_t desired [BITS], word_t* target) {
    bhv::rand_into(target);

    auto loss = [&desired](word_t *x) {
        word_t tmp [WORDS];
        uint64_t total_error = 0;
        for (size_t i = 0; i < BITS; ++i) {
            bhv::roll_bits_into(x, i, tmp);
            uint64_t s = (desired[i] - (size_t)bhv::hamming(x, tmp));
//            total_error += s*s;
            total_error += std::abs((long)s);
        }
        return total_error;
    };

    auto t1 = chrono::high_resolution_clock::now();
//    for (size_t i = 0; i < 2; ++i) {
//        cout << i << ": " << loss(target) << " loss" << endl;
//        bhv::opt::independent<uint64_t>(target, loss);
//    }
//    bhv::opt::linear_search<uint64_t>(target, loss, 1000000, 5*BITS);
//    bhv::opt::evolutionary_search<uint64_t, 16>(target, loss);
    bhv::opt::evolutionary_search<uint64_t, 64, 8>(target, loss, 12);
    bhv::opt::independent<uint64_t>(target, loss);
    auto t2 = chrono::high_resolution_clock::now();

    cout << loss(target) << " loss" << endl;
    double test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) (1000000000);
    cout << test_time << " seconds" << endl;
}


int main() {
    word_t v [WORDS];

    // angle_to_index(phase_difference(0., index_to_angle(i))/2)
    // total absolute error 1643365

    size_t linear_lookup [BITS];
    for (size_t i = 0; i < BITS; ++i)
        linear_lookup[i] = angle_to_index(angle_difference(0., index_to_angle(i))/2);
//        linear_lookup[i] = min(i, BITS - i);

    cout << "initialized" << endl;
    search_kernel(linear_lookup, v);
    cout << "result" << endl;
    cout << bhv::to_string(v) << endl;

    word_t* v_array [1] = {v};
    FILE* ofile = fopen("phase_difference.pbm", "wb");
    bhv::save_pbm(ofile, v_array, 1);
    fclose(ofile);
}
