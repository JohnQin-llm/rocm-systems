// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for scalar floating-point arithmetic and selection.

#include "rj_arith/detail/rj_arith_scalar.h"
#include "scalar/fp_helpers.h"
#include <algorithm>
#include <limits>

// Check scalar FP results and the expected semantics/status policy.
TEST(s_add_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_add_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x + y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_sub_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_sub_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x - y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_mul_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_mul_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(x * y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_min_num_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_min_num_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_max_num_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_max_num_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_minimum_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_minimum_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_maximum_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_maximum_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmac_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_fmac_f32(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(y), float(z)));
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_add_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_add_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x + y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_sub_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_sub_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x - y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_mul_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_mul_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(x * y);
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_min_num_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_min_num_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_max_num_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_max_num_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? (std::isnan(x) ? y : x)
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_minimum_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_minimum_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) || std::signbit(y)) ? -1.0 : 1.0)
                : std::min(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_maximum_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_maximum_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(
            (std::isnan(x) || std::isnan(y)) ? std::numeric_limits<double>::quiet_NaN()
            : x == 0 && y == 0
                ? std::copysign(0.0, (std::signbit(x) && std::signbit(y)) ? -1.0 : 1.0)
                : std::max(x, y));
      },
      false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmac_f16, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<true>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        (void)b;
        (void)c;
        return rj_arith_s_fmac_f16(f, m, d, a, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<true>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<true>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<true>(c, flush);
        (void)y;
        (void)z;
        return rocjitsu::arith::test::scalar_fp_bits<true>(std::fma(x, y, z));
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmaak_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmaak_f32(f, m, d, a, b, c, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(y), float(z)));
      },
      true, 0, false);
}

// Check scalar FP results and the expected semantics/status policy.
TEST(s_fmamk_f32, Arithmetic) {
  rocjitsu::arith::test::check_scalar_fp<false>(
      [](auto f, auto m, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmamk_f32(f, m, d, a, c, b, status);
      },
      [](auto a, auto b, auto c, bool flush) {
        double x = rocjitsu::arith::test::scalar_fp_value<false>(a, flush),
               y = rocjitsu::arith::test::scalar_fp_value<false>(b, flush),
               z = rocjitsu::arith::test::scalar_fp_value<false>(c, flush);
        return rocjitsu::arith::test::scalar_fp_bits<false>(std::fma(float(x), float(z), float(y)));
      },
      true, 0, false);
}

// gfx1201 captures: special, denormal, overflowing and inexact inputs in default MODE
// 0xf0 and in flushing, directed and FP16_OVFL MODEs; inputs whose result changes with
// the rounding direction in all four directions; and one input under MODE 0xf3/0xfc,
// whose F32 and F16 rounding fields differ. c is the FMAC accumulator or the
// FMAAK/FMAMK literal.
const rocjitsu::arith::test::ScalarCapture s_add_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0x0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x80eba725, 0x11a17a4, 0x0, 0x0},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x800000, 0x1, 0x0, 0x800001},
    {0x10, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0x15, 0x7f7fffff, 0x1, 0x0, 0x7f800000},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf5, 0x3f800000, 0x1, 0x0, 0x3f800001},
    {0xfa, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xff, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf5, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xfa, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xff, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xf0, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf5, 0x3f000000, 0x1, 0x0, 0x3f000001},
    {0xfa, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xff, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf3, 0xbf800000, 0x1, 0x0, 0xbf7fffff},
    {0xfc, 0xbf800000, 0x1, 0x0, 0xbf800000},
};

TEST(s_add_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_add_f32(f, 0, d, a, b, status);
      },
      s_add_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_sub_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0xf0, 0xff7fffff, 0x7f7fffff, 0x0, 0xff800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800001, 0x0, 0x0, 0x7fc00001},
    {0x0, 0x3f800000, 0x800000, 0x0, 0x3f800000},
    {0x0, 0xff7fffff, 0x7f7fffff, 0x0, 0xff800000},
    {0x0, 0x80d8feb1, 0x811cab1c, 0x0, 0x0},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0x15, 0x7f7fffff, 0x807fffff, 0x0, 0x7f800000},
    {0xf0, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf5, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xfa, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xff, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xf0, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf5, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xfa, 0xbf800000, 0x1, 0x0, 0xbf800001},
    {0xff, 0xbf800000, 0x1, 0x0, 0xbf800000},
    {0xf0, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xf5, 0x3f000000, 0x1, 0x0, 0x3f000000},
    {0xfa, 0x3f000000, 0x1, 0x0, 0x3effffff},
    {0xff, 0x3f000000, 0x1, 0x0, 0x3effffff},
    {0xf3, 0x3f800000, 0x1, 0x0, 0x3f7fffff},
    {0xfc, 0x3f800000, 0x1, 0x0, 0x3f800000},
    {0xf0, 0x0, 0x7fc00000, 0x0, 0xffc00000},
};

TEST(s_sub_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_sub_f32(f, 0, d, a, b, status);
      },
      s_sub_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_mul_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x800000, 0x800000, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0xf0, 0x3effffff, 0x7f7fffff, 0x0, 0x7efffffe},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x3effffff, 0x7f7fffff, 0x0, 0x7efffffe},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf5, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xfa, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xff, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xf0, 0xbfc00000, 0x1, 0x0, 0x80000002},
    {0xf5, 0xbfc00000, 0x1, 0x0, 0x80000001},
    {0xfa, 0xbfc00000, 0x1, 0x0, 0x80000002},
    {0xff, 0xbfc00000, 0x1, 0x0, 0x80000001},
    {0xf0, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xf5, 0x40490fdb, 0x1, 0x0, 0x4},
    {0xfa, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xff, 0x40490fdb, 0x1, 0x0, 0x3},
    {0xf3, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xfc, 0x3fc00000, 0x1, 0x0, 0x2},
};

TEST(s_mul_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_mul_f32(f, 0, d, a, b, status);
      },
      s_mul_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_add_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},         {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7c01, 0x0, 0x0, 0x7e01},   {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf0, 0x3c00, 0x400, 0x0, 0x3c00}, {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x0, 0x0, 0x0, 0x0},          {0x0, 0x7c01, 0x0, 0x0, 0x7e01},
    {0x0, 0x3c00, 0x400, 0x0, 0x3c00},  {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x47b, 0x8751, 0x0, 0x8000},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x400, 0x1, 0x0, 0x401},     {0x40, 0x3c00, 0x1, 0x0, 0x3c00},
    {0x45, 0x7bff, 0x1, 0x0, 0x7c00},   {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf5, 0x3c00, 0x1, 0x0, 0x3c01},   {0xfa, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xff, 0x3c00, 0x1, 0x0, 0x3c00},   {0xf0, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xf5, 0xbc00, 0x1, 0x0, 0xbbff},   {0xfa, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xff, 0xbc00, 0x1, 0x0, 0xbbff},   {0xf0, 0x3800, 0x1, 0x0, 0x3800},
    {0xf5, 0x3800, 0x1, 0x0, 0x3801},   {0xfa, 0x3800, 0x1, 0x0, 0x3800},
    {0xff, 0x3800, 0x1, 0x0, 0x3800},   {0xf3, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xfc, 0xbc00, 0x1, 0x0, 0xbbff},
};

TEST(s_add_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_add_f16(f, 0, d, a, b, status);
      },
      s_add_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_sub_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x1},
    {0xf0, 0x7c01, 0x0, 0x0, 0x7e01},    {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf0, 0x3c00, 0x400, 0x0, 0x3c00},  {0xf0, 0xfbff, 0x7bff, 0x0, 0xfc00},
    {0x0, 0x0, 0x0, 0x0, 0x0},           {0x0, 0x7c01, 0x0, 0x0, 0x7e01},
    {0x0, 0x3c00, 0x400, 0x0, 0x3c00},   {0x0, 0xfbff, 0x7bff, 0x0, 0xfc00},
    {0x0, 0x868b, 0x85d8, 0x0, 0x8000},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x1, 0x1, 0x0, 0x0},          {0x40, 0x3c00, 0x1, 0x0, 0x3c00},
    {0x45, 0x7bff, 0x83ff, 0x0, 0x7c00}, {0xf0, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xf5, 0x3c00, 0x1, 0x0, 0x3c00},    {0xfa, 0x3c00, 0x1, 0x0, 0x3bff},
    {0xff, 0x3c00, 0x1, 0x0, 0x3bff},    {0xf0, 0xbc00, 0x1, 0x0, 0xbc00},
    {0xf5, 0xbc00, 0x1, 0x0, 0xbc00},    {0xfa, 0xbc00, 0x1, 0x0, 0xbc01},
    {0xff, 0xbc00, 0x1, 0x0, 0xbc00},    {0xf0, 0x3800, 0x1, 0x0, 0x3800},
    {0xf5, 0x3800, 0x1, 0x0, 0x3800},    {0xfa, 0x3800, 0x1, 0x0, 0x37ff},
    {0xff, 0x3800, 0x1, 0x0, 0x37ff},    {0xf3, 0x3c00, 0x1, 0x0, 0x3c00},
    {0xfc, 0x3c00, 0x1, 0x0, 0x3bff},    {0xf0, 0x0, 0x7e00, 0x0, 0xfe00},
};

TEST(s_sub_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_sub_f16(f, 0, d, a, b, status);
      },
      s_sub_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_mul_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7c00, 0x0, 0x0, 0xfe00},    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf0, 0x400, 0x400, 0x0, 0x0},
    {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00}, {0xf0, 0x3bff, 0x7bff, 0x0, 0x7bfe},
    {0x0, 0x0, 0x0, 0x0, 0x0},           {0x0, 0x7c00, 0x0, 0x0, 0xfe00},
    {0x0, 0x400, 0x400, 0x0, 0x0},       {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},
    {0x0, 0x3bff, 0x7bff, 0x0, 0x7bfe},  {0x40, 0x1, 0x0, 0x0, 0x0},
    {0x40, 0x1, 0x1, 0x0, 0x0},          {0x40, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xf0, 0x3e00, 0x1, 0x0, 0x2},       {0xf5, 0x3e00, 0x1, 0x0, 0x2},
    {0xfa, 0x3e00, 0x1, 0x0, 0x1},       {0xff, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0xbe00, 0x1, 0x0, 0x8002},    {0xf5, 0xbe00, 0x1, 0x0, 0x8001},
    {0xfa, 0xbe00, 0x1, 0x0, 0x8002},    {0xff, 0xbe00, 0x1, 0x0, 0x8001},
    {0xf0, 0x4248, 0x1, 0x0, 0x3},       {0xf5, 0x4248, 0x1, 0x0, 0x4},
    {0xfa, 0x4248, 0x1, 0x0, 0x3},       {0xff, 0x4248, 0x1, 0x0, 0x3},
    {0xf3, 0x3e00, 0x1, 0x0, 0x2},       {0xfc, 0x3e00, 0x1, 0x0, 0x1},
};

TEST(s_mul_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_mul_f16(f, 0, d, a, b, status);
      },
      s_mul_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_fmac_f16_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},          {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7c00, 0x0, 0x0, 0xfe00},    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf0, 0x400, 0x400, 0x0, 0x0},
    {0xf0, 0x7bff, 0x7bff, 0x0, 0x7c00}, {0xf0, 0x7bff, 0x7bff, 0x1, 0x7c00},
    {0xf0, 0x400, 0x400, 0x400, 0x400},  {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7c00, 0x0, 0x0, 0xfe00},     {0x0, 0x400, 0x400, 0x0, 0x0},
    {0x0, 0x7bff, 0x7bff, 0x0, 0x7c00},  {0x0, 0x400, 0x400, 0x400, 0x400},
    {0x40, 0x1, 0x0, 0x0, 0x0},          {0x40, 0x1, 0x1, 0x0, 0x0},
    {0x40, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0x40, 0x7bff, 0x7bff, 0x1, 0x7c00},
    {0xf0, 0x3e00, 0x1, 0x0, 0x2},       {0xf5, 0x3e00, 0x1, 0x0, 0x2},
    {0xfa, 0x3e00, 0x1, 0x0, 0x1},       {0xff, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0x3800, 0x83ff, 0x0, 0x8200}, {0xf5, 0x3800, 0x83ff, 0x0, 0x81ff},
    {0xfa, 0x3800, 0x83ff, 0x0, 0x8200}, {0xff, 0x3800, 0x83ff, 0x0, 0x81ff},
    {0xf0, 0x7bff, 0x83ff, 0x0, 0xc3fd}, {0xf5, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xfa, 0x7bff, 0x83ff, 0x0, 0xc3fe}, {0xff, 0x7bff, 0x83ff, 0x0, 0xc3fd},
    {0xf3, 0x3e00, 0x1, 0x0, 0x2},       {0xfc, 0x3e00, 0x1, 0x0, 0x1},
    {0xf0, 0x7c00, 0x0, 0x7e00, 0xfe00},
};

TEST(s_fmac_f16, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_fmac_f16(f, 0, d, a, b, status);
      },
      s_fmac_f16_captures, 16, 16);
}

const rocjitsu::arith::test::ScalarCapture s_fmac_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x0, 0x0},
    {0xf0, 0x1, 0x0, 0x0, 0x0},
    {0xf0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf0, 0x800000, 0x800000, 0x0, 0x0},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x1, 0x7f800000},
    {0xf0, 0x800000, 0x800000, 0x800000, 0x800000},
    {0x0, 0x0, 0x0, 0x0, 0x0},
    {0x0, 0x7f800000, 0x0, 0x0, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x0, 0x0},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x0, 0x7f800000},
    {0x0, 0x800000, 0x800000, 0x800000, 0x800000},
    {0x10, 0x1, 0x0, 0x0, 0x0},
    {0x10, 0x1, 0x1, 0x0, 0x0},
    {0x10, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0x10, 0x7f7fffff, 0x7f7fffff, 0x1, 0x7f800000},
    {0xf0, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf5, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xfa, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xff, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xf0, 0x3f000000, 0x807fffff, 0x0, 0x80400000},
    {0xf5, 0x3f000000, 0x807fffff, 0x0, 0x803fffff},
    {0xfa, 0x3f000000, 0x807fffff, 0x0, 0x80400000},
    {0xff, 0x3f000000, 0x807fffff, 0x0, 0x803fffff},
    {0xf0, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf5, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xfa, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffe},
    {0xff, 0x7f7fffff, 0x807fffff, 0x0, 0xc07ffffd},
    {0xf3, 0x3fc00000, 0x1, 0x0, 0x1},
    {0xfc, 0x3fc00000, 0x1, 0x0, 0x2},
    {0xf0, 0x7f800000, 0x0, 0x7fc00000, 0xffc00000},
};

TEST(s_fmac_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto, auto status) {
        return rj_arith_s_fmac_f32(f, 0, d, a, b, status);
      },
      s_fmac_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_fmaak_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x7f800000, 0x0, 0x40490fdb, 0xffc00000},
    {0xf0, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x800000, 0x800000, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x7f7fffff, 0x7f7fffff, 0x40490fdb, 0x7f800000},
    {0x0, 0x0, 0x0, 0x40490fdb, 0x40490fdb},
    {0x0, 0x7f800000, 0x0, 0x40490fdb, 0xffc00000},
    {0x0, 0x800000, 0x800000, 0x40490fdb, 0x40490fdb},
    {0x0, 0x7f7fffff, 0x7f7fffff, 0x40490fdb, 0x7f800000},
    {0x10, 0x1, 0x0, 0x40490fdb, 0x40490fdb},
    {0x10, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x1, 0x1, 0x40490fdb, 0x40490fdc},
    {0xfa, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xff, 0x1, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf0, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
    {0xfa, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xff, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xf0, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf5, 0x800000, 0x1, 0x40490fdb, 0x40490fdc},
    {0xfa, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xff, 0x800000, 0x1, 0x40490fdb, 0x40490fdb},
    {0xf3, 0x807fffff, 0x1, 0x40490fdb, 0x40490fda},
    {0xfc, 0x807fffff, 0x1, 0x40490fdb, 0x40490fdb},
};

TEST(s_fmaak_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmaak_f32(f, 0, d, a, b, c, status);
      },
      s_fmaak_f32_captures, 32, 32);
}

const rocjitsu::arith::test::ScalarCapture s_fmamk_f32_captures[] = {
    {0xf0, 0x0, 0x0, 0x40490fdb, 0x0},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf0, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xf0, 0x7f800001, 0x0, 0x40490fdb, 0x7fc00001},
    {0xf0, 0x3fc00000, 0x0, 0x40490fdb, 0x4096cbe4},
    {0xf0, 0x400000, 0x0, 0x40490fdb, 0xc90fdb},
    {0xf0, 0x7f7fffff, 0x1, 0x40490fdb, 0x7f800000},
    {0xf0, 0x9e81b6, 0x81508a70, 0x40490fdb, 0x50e1c4},
    {0x0, 0x0, 0x0, 0x40490fdb, 0x0},
    {0x0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0x0, 0x7f800001, 0x0, 0x40490fdb, 0x7fc00001},
    {0x0, 0x3fc00000, 0x0, 0x40490fdb, 0x4096cbe4},
    {0x0, 0x9e81b6, 0x81508a70, 0x40490fdb, 0x0},
    {0x10, 0x1, 0x0, 0x40490fdb, 0x0},
    {0x10, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0x10, 0x400000, 0x0, 0x40490fdb, 0xc90fdb},
    {0x10, 0x7f7fffff, 0x1, 0x40490fdb, 0x7f800000},
    {0xf0, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf5, 0x1, 0x0, 0x40490fdb, 0x4},
    {0xfa, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xff, 0x1, 0x0, 0x40490fdb, 0x3},
    {0xf0, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf5, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xfa, 0x807fffff, 0x0, 0x40490fdb, 0x81490fda},
    {0xff, 0x807fffff, 0x0, 0x40490fdb, 0x81490fd9},
    {0xf0, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xf5, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
    {0xfa, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xff, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xf3, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f7fffff},
    {0xfc, 0x7f7fffff, 0x0, 0x40490fdb, 0x7f800000},
};

TEST(s_fmamk_f32, Gfx1201Captures) {
  rocjitsu::arith::test::check_scalar_captures(
      [](auto f, auto d, auto a, auto b, auto c, auto status) {
        return rj_arith_s_fmamk_f32(f, 0, d, a, c, b, status);
      },
      s_fmamk_f32_captures, 32, 32);
}
