#pragma once
#include <cstdio>
#include <cmath>
#include <string>

namespace test {

inline int& failures() { static int f = 0; return f; }
inline int& checks()   { static int c = 0; return c; }

inline void report_and_reset(const char* suite) {
    std::printf("[%s] %d checks, %d failures\n", suite, checks(), failures());
}

} // namespace test

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++::test::checks();                                                    \
        if (!(cond)) {                                                         \
            ++::test::failures();                                              \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__, #cond);\
        }                                                                      \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        ++::test::checks();                                                    \
        if (!((a) == (b))) {                                                   \
            ++::test::failures();                                              \
            std::printf("  FAIL %s:%d  CHECK_EQ(%s, %s)\n",                    \
                        __FILE__, __LINE__, #a, #b);                           \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        ++::test::checks();                                                    \
        if (std::fabs(double(a) - double(b)) > (eps)) {                        \
            ++::test::failures();                                              \
            std::printf("  FAIL %s:%d  CHECK_NEAR(%s, %s)\n",                  \
                        __FILE__, __LINE__, #a, #b);                           \
        }                                                                      \
    } while (0)
