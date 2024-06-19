g++ -c lib.cpp -fPIC  -o cbhv.o \
  -D DIMENSION=8192 -O3 -std=c++20 -march=native -Wall -Wpedantic -Wextra -g -ffast-math -fopenmp -pipe
# add -fopenmp if you want to use that instead of TBB for parallelism; without it -ltbb is used with the std library
# if you don't want to build with parallelism use -D NOPARALLELISM
gcc -fdata-sections -ffunction-sections -Wl,-gc-sections -shared -o libbhv.so cbhv.o
