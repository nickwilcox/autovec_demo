use autovec_demo::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const BENCHMARK_SAMPLES: usize = 100_000;

fn criterion_benchmark(c: &mut Criterion) {
    let src = vec![0.0; BENCHMARK_SAMPLES];
    let mut dst = vec![0.0; BENCHMARK_SAMPLES * 2];
    c.bench_function("attempt 1", move |b| {
        b.iter(|| mix_mono_to_stereo_1(&mut dst, &src, 1.0, 1.0))
    });
    let src = vec![MonoSample(0.0); BENCHMARK_SAMPLES];
    let mut dst = vec![StereoSample { l: 0.0, r: 0.0 }; BENCHMARK_SAMPLES * 2];
    c.bench_function("attempt 3", move |b| {
        b.iter(|| mix_mono_to_stereo_3(&mut dst, &src, 1.0, 1.0))
    });
    let src = vec![0.0; BENCHMARK_SAMPLES];
    let mut dst = vec![0.0; BENCHMARK_SAMPLES * 2];
    c.bench_function("rust intrinsics", move |b| {
        b.iter(|| mix_mono_to_stereo_intrinsics_rust(&mut dst, &src, 1.0, 1.0))
    });
    let src = vec![0.0; BENCHMARK_SAMPLES];
    let mut dst = vec![0.0; BENCHMARK_SAMPLES * 2];
    c.bench_function("C intrinsics", move |b| {
        b.iter(|| mix_mono_to_stereo_intrinsics_safe(&mut dst, &src, 1.0, 1.0))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
