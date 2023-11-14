set -e
g++ benchmark.cpp TurboSHAKE_opt/TurboSHAKE.cpp TurboSHAKE_opt/KeccakP-1600-opt64.cpp \
  TurboSHAKE_AVX512/TurboSHAKE.cpp TurboSHAKE_AVX512/KeccakP-1600-AVX512.cpp \
  -D DIMENSION=8192 -O3 -std=c++20 -march=native -Wall -Wpedantic -Wextra -g -ffast-math -fopenmp
# add -fopenmp if you want to use that instead of TBB for parallelism; without it -ltbb is used with the std library
./a.out
#rm a.out
