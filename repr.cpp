#include "iostream"
#include "core.h"


using namespace std;


template <void F(word_t**, size_t, word_t*)>
void analyze(int arity) {
    cout << "arity: " << arity << endl;

    vector<word_t*> hvs;
    for (int i = 0; i < arity; ++i) {
        hvs.push_back(bhv::rand());
    }

    word_t* r = bhv::empty();

    F(hvs.data(), arity, r);

    cout << arity << endl;
    for (int i = 0; i < min(arity, 50); ++i) {
        cout << "d(x" << i + 1 << ", r) = " << (float)bhv::hamming(hvs[i], r) / (float)DIMENSION << endl;
    }

    for (word_t* hv : hvs) {
        free(hv);
    }

    free(r);
}

int main() {
    cout << "rec" << endl;
    // analyze<bhv::representative_into_avx2>(3);
    // analyze<bhv::representative_into_avx2>(4);
    // analyze<bhv::representative_into_avx2>(5);
    // analyze<bhv::representative_into_avx2>(8);
    analyze<bhv::representative_into_avx2>(133);
    cout << "byte counter" << endl;
    // analyze<bhv::representative_into_byte_counters_avx2>(3);
    // analyze<bhv::representative_into_byte_counters_avx2>(4);
    // analyze<bhv::representative_into_byte_counters_avx2>(5);
    // analyze<bhv::representative_into_byte_counters_avx2>(8);
    analyze<bhv::representative_into_byte_counters_avx2>(133);
    cout << "reference" << endl;
    // analyze<bhv::representative_into_bitcount_reference>(3);
    // analyze<bhv::representative_into_bitcount_reference>(4);
    // analyze<bhv::representative_into_bitcount_reference>(5);
    // analyze<bhv::representative_into_bitcount_reference>(8);
    analyze<bhv::representative_into_bitcount_reference>(133);
    // analyze<bhv::representative_into_counters_avx2>(260);
    // cout << "counters32" << endl;
    // analyze<bhv::representative_into_counters_avx2>(260);
}
