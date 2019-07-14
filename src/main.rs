#![allow(dead_code)]

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
    l: f32,
    r: f32,
}

#[repr(transparent)]
pub struct MonoSample(f32);

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

fn main() {
    println!("Auto Vectorization Tutorial");
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
