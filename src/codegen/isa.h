// The instruction-set layer: the only place that knows how a vector operation
// is spelled. The emitter asks for "add two vectors" and gets back the name of
// the intrinsic. Supporting AVX2 means adding a second table here, and a
// second header of vector math next to runtime/mlc_neon.h; the loop IR and
// the emitter stay as they are.
#pragma once

namespace mlc::codegen {

struct Isa {
  const char* name;
  int width;             // f32 lanes per vector
  const char* includes;  // what generated code must include to use it
  const char* type;      // the vector-of-f32 type
  // Each is a function name; the comment gives the call's meaning.
  const char* add;       // add(a, b)
  const char* sub;       // sub(a, b)      a - b
  const char* mul;       // mul(a, b)
  const char* div;       // div(a, b)      a / b
  const char* max;       // max(a, b)
  const char* fma;       // fma(acc, a, b) acc + a * b, rounded once
  const char* fms;       // fms(acc, a, b) acc - a * b
  const char* dup;       // dup(scalar)    the scalar in every lane
  const char* load;      // load(pointer)  `width` consecutive floats
  const char* store;     // store(pointer, v)
  const char* sum;       // sum(v)         add the lanes together, giving a scalar
  const char* hmax;      // hmax(v)        the largest lane
  const char* sqrt;
  const char* exp;       // from the vector math header
  const char* tanh;
};

inline const Isa& neon() {
  static const Isa isa = {
      "neon",      4,           "#include <arm_neon.h>\n\n#include \"mlc_neon.h\"\n", "float32x4_t",
      "vaddq_f32", "vsubq_f32", "vmulq_f32", "vdivq_f32", "vmaxq_f32", "vfmaq_f32", "vfmsq_f32", "vdupq_n_f32",
      "vld1q_f32", "vst1q_f32", "vaddvq_f32", "vmaxvq_f32", "vsqrtq_f32", "mlc_exp_f32x4", "mlc_tanh_f32x4",
  };
  return isa;
}

}  // namespace mlc::codegen
