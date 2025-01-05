#ifndef BHV_CONSTANTS_H
#define BHV_CONSTANTS_H


using word_t = uint64_t;

#if DIMENSION/64 >= 65536
using word_iter_t = uint32_t;
#elif DIMENSION/64 >= 256
using word_iter_t = uint16_t;
#else
using word_iter_t = uint8_t;
#endif

#if DIMENSION/8 >= 65536
using byte_iter_t = uint32_t;
#elif DIMENSION/8 >= 256
using byte_iter_t = uint16_t;
#else
using byte_iter_t = uint8_t;
#endif

#if DIMENSION >= 65536
using bit_iter_t = uint32_t;
#else
using bit_iter_t = uint16_t;
#endif

using bit_word_iter_t = uint8_t;

#define BITS_PER_WORD 64
#define BITS DIMENSION

#define BYTES (BITS / 8)
#define WORDS (BITS / BITS_PER_WORD)

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
    (static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

template<auto n, typename F>
constexpr void for_sequence(F f) {
    for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

#endif //BHV_CONSTANTS_H
