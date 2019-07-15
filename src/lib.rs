#![allow(dead_code)]

use core::arch::x86_64::*;

pub fn mix_mono_to_stereo_intrinsics_rust(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    assert_eq!(src.len() % 4, 0);
    assert_eq!(dst.len(), src.len() * 2);
    unsafe {
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();

        // create SIMD variables with each element set to the same value
        // mul_l = |  gain_l |  gain_l |  gain_l |  gain_l |
        // mul_r = |  gain_r |  gain_r |  gain_r |  gain_r |
        let mul_l = _mm_set1_ps(gain_l);
        let mul_r = _mm_set1_ps(gain_r);

        // process the source samples in blocks of four
        let mut i = 0;
        while i < src.len() {
            // load 4 of our source samples
            // input = | src(i + 0) | src(i + 1) | src(i + 2) | src(i + 3) |
            let input = _mm_loadu_ps(src_ptr.add(i));

            // multiply each of the four input values by the left and right volumes
            // we now have two variables containing four output values is each
            // out_l = | src(i + 0) * gain_l | src(i + 1) * gain_l | src(i + 2) * gain_l | src(i + 3) * gain_l |
            // out_r = | src(i + 0) * gain_r | src(i + 1) * gain_r | src(i + 2) * gain_r | src(i + 3) * gain_r |
            let out_l = _mm_mul_ps(input, mul_l);
            let out_r = _mm_mul_ps(input, mul_r);

            // re-arrange the output values so that each left-right pair is next to each other
            // out_lo = | src(i + 0) * gain_l | src(i + 0) * gain_r | src(i + 1) * gain_l | src(i + 1) * gain_r |
            // out_hi = | src(i + 2) * gain_l | src(i + 2) * gain_r | src(i + 3) * gain_l | src(i + 3) * gain_r |
            let out_lo = _mm_unpacklo_ps(out_l, out_r);
            let out_hi = _mm_unpackhi_ps(out_l, out_r);

            // write the four output samples (8 f32 values) to our destination memory
            _mm_storeu_ps(dst_ptr.add(2 * i + 0), out_lo);
            _mm_storeu_ps(dst_ptr.add(2 * i + 4), out_hi);

            i += 4;
        }
    }
}

extern "C" {
    fn mix_mono_to_stereo_intrinsics(
        samples: i32,
        dst: *mut f32,
        src: *const f32,
        gain_l: f32,
        gain_r: f32,
    );
}

/// Wrap the version written in C intrinsics in a safe rust wrapper
pub fn mix_mono_to_stereo_intrinsics_safe(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    unsafe {
        mix_mono_to_stereo_intrinsics(
            src.len() as i32,
            dst.as_mut_ptr(),
            src.as_ptr(),
            gain_l,
            gain_r,
        )
    };
}

/// First attempt to produce an auto-vectorized loop.
/// As seen at https://godbolt.org/z/Das0yY it is unsuccessful
pub fn mix_mono_to_stereo_1(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    for i in 0..src.len() {
        dst[i * 2 + 0] = src[i] * gain_l;
        dst[i * 2 + 1] = src[i] * gain_r;
    }
}

/// Second attempt to produce an auto-vectorized loop by proving the range bounds to the compiler
/// As seen at https://godbolt.org/z/UPUWL1 it is also unsuccessful
pub fn mix_mono_to_stereo_2(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    let dst_known_bounds = &mut dst[0..src.len() * 2];
    for i in 0..src.len() {
        dst_known_bounds[i * 2 + 0] = src[i] * gain_l;
        dst_known_bounds[i * 2 + 1] = src[i] * gain_r;
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct StereoSample {
    pub l: f32,
    pub r: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct MonoSample(pub f32);

/// Third attempt that is able to produce an auto-vectorized loop.
/// See https://godbolt.org/z/UPUWL1 for compiler output
pub fn mix_mono_to_stereo_3(
    dst: &mut [StereoSample],
    src: &[MonoSample],
    gain_l: f32,
    gain_r: f32,
) {
    let dst_known_bounds = &mut dst[0..src.len()];
    for i in 0..src.len() {
        dst_known_bounds[i].l = src[i].0 * gain_l;
        dst_known_bounds[i].r = src[i].0 * gain_r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intrinsics() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0; src.len() * 2];
        mix_mono_to_stereo_intrinsics_safe(&mut dst, &src, 0.25, 2.0);
        assert_eq!(
            dst,
            vec![
                0.25, 2.0, 0.5, 4.0, 0.75, 6.0, 1.0, 8.0, 1.25, 10.0, 1.5, 12.0, 1.75, 14.0, 2.0,
                16.0
            ]
        );
    }

    #[test]
    fn intrinsics_rust() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0; src.len() * 2];
        mix_mono_to_stereo_intrinsics_rust(&mut dst, &src, 0.25, 2.0);
        assert_eq!(
            dst,
            vec![
                0.25, 2.0, 0.5, 4.0, 0.75, 6.0, 1.0, 8.0, 1.25, 10.0, 1.5, 12.0, 1.75, 14.0, 2.0,
                16.0
            ]
        );
    }
    #[test]
    fn vectorized() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .map(|x| MonoSample(*x))
            .collect::<Vec<_>>();
        let mut dst = vec![StereoSample { l: 0.0, r: 0.0 }; src.len()];
        mix_mono_to_stereo_3(&mut dst, &src, 0.25, 2.0);
        assert_eq!(
            dst,
            vec![
                StereoSample { l: 0.25, r: 2.0 },
                StereoSample { l: 0.5, r: 4.0 },
                StereoSample { l: 0.75, r: 6.0 },
                StereoSample { l: 1.0, r: 8.0 },
                StereoSample { l: 1.25, r: 10.0 },
                StereoSample { l: 1.5, r: 12.0 },
                StereoSample { l: 1.75, r: 14.0 },
                StereoSample { l: 2.0, r: 16.0 },
            ]
        );
    }
}
