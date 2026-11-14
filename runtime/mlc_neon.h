// Vector math for generated code on arm64 NEON: exp and tanh on four floats
// at a time. The C library only offers these one float at a time, and a
// vectorized loop that had to call it per lane would not be worth vectorizing.
#pragma once

#include <arm_neon.h>

// exp(x) = 2^n * exp(r), with n = round(x / ln 2) and r = x - n ln 2, so that
// |r| <= ln(2)/2 = 0.35. On that short interval exp(r) is a degree-6
// polynomial to within float rounding (coefficients from Cephes' expf), and
// 2^n is built directly by writing n into the exponent bits of a float.
//
// Accuracy: relative error below 2.5e-7 over the whole range (tested).
// Range: x is clamped to [-87.3, 88.0] first, so the result is never 0, never
// a denormal, and never inf: exp(-1000) gives 1.2e-38 and exp(1000) gives
// 1.7e38. NaN stays NaN. For softmax (arguments <= 0) and tanh (below) the
// clamps are never reached in a way that matters.
static inline float32x4_t mlc_exp_f32x4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.0f));
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));  // round to nearest
  // r = x - n * ln2, with ln2 split in two parts so the product n * ln2_hi is exact.
  float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
  r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
  // exp(r) = 1 + r + r^2 * (c5 + c4 r + c3 r^2 + c2 r^3 + c1 r^4 + c0 r^5), by Horner's rule.
  float32x4_t poly = vdupq_n_f32(1.9875691500e-4f);
  poly = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), poly, r);
  poly = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), poly, r);
  const float32x4_t y = vaddq_f32(vfmaq_f32(r, poly, vmulq_f32(r, r)), vdupq_n_f32(1.0f));
  // 2^n: a float with exponent field n + 127 and a zero fraction.
  const int32x4_t bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(bits));
}

// tanh(x) = 1 - 2 / (exp(2x) + 1). x is clamped to [-10, 10], beyond which
// tanh is 1 or -1 to float precision. Absolute error is below 2.5e-7 everywhere;
// relative error is poor for |x| < 1e-3, where the result is close to x and
// the subtraction from 1 cancels. GELU multiplies by x again, so that does
// not matter here; a general-purpose tanh would switch to a polynomial there.
static inline float32x4_t mlc_tanh_f32x4(float32x4_t x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-10.0f)), vdupq_n_f32(10.0f));
  const float32x4_t e = mlc_exp_f32x4(vaddq_f32(x, x));
  return vsubq_f32(vdupq_n_f32(1.0f), vdivq_f32(vdupq_n_f32(2.0f), vaddq_f32(e, vdupq_n_f32(1.0f))));
}
