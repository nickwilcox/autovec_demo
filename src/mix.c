#include <stdint.h>
#include <assert.h>
#include <emmintrin.h>

void mix_mono_to_stereo_intrinsics(uint32_t num_samples, float *dst, float *src, float gain_l, float gain_r)
{
    // the number of samples to mix must be a multiple of 4
    assert(num_samples % 4 == 0);

    // create SIMD variables with each element set to the same value
    // mul_l = |  gain_l |  gain_l |  gain_l |  gain_l |
    // mul_r = |  gain_r |  gain_r |  gain_r |  gain_r |
    __m128 mul_l = _mm_set1_ps(gain_l);
    __m128 mul_r = _mm_set1_ps(gain_r);

    // process the source samples in blocks of four
    for (uint32_t i = 0; i < num_samples; i += 4)
    {
        // load 4 of our source samples
        // in = | src(i + 0) | src(i + 1) | src(i + 2) | src(i + 3) |
        __m128 in = _mm_loadu_ps(src + i);

        // multiply each of the four input values by the left and right volumes
        // we now have two variables containing four output values is each
        // out_l = | src(i + 0) * gain_l | src(i + 1) * gain_l | src(i + 2) * gain_l | src(i + 3) * gain_l |
        // out_r = | src(i + 0) * gain_r | src(i + 1) * gain_r | src(i + 2) * gain_r | src(i + 3) * gain_r |
        __m128 out_l = _mm_mul_ps(in, mul_l);
        __m128 out_r = _mm_mul_ps(in, mul_r);

        // re-arrange the output values so that each left-right pair is next to each other
        // out_lo = | src(i + 0) * gain_l | src(i + 0) * gain_r | src(i + 1) * gain_l | src(i + 1) * gain_r |
        // out_hi = | src(i + 2) * gain_l | src(i + 2) * gain_r | src(i + 3) * gain_l | src(i + 3) * gain_r |
        __m128 out_lo = _mm_unpacklo_ps(out_l, out_r);
        __m128 out_hi = _mm_unpackhi_ps(out_l, out_r);

        // write the four output samples (8 values) to our destination memory
        _mm_storeu_ps(dst + (2 * i + 0), out_lo);
        _mm_storeu_ps(dst + (2 * i + 4), out_hi);
    }
}
