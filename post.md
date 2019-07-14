# Taking Advantage of Auto-Vectorization in Rust
Recently I wrote some audio processing code in Rust. In the past I've worked on a lot of audio processing code in C++ where performance was critical and spent a lot of time making sure the code could process audio samples a fast as possible.

We're going to take one small piece of audio processing code and take a look at how we can optimize the Rust version.

## Mixing Mono to Stereo
The function we're going to optimize is taking a mono audio signal and mixing it into a stereo signal containing both a left and right component.

The pseudo-code is
```
for each input sample
    output sample left = input sample * left gain
    output sample right = input sample * right gain
```
The processing is very simple, but audio signals typically contain 48000 samples per second. So we are going to want this loop to be able to run as quickly as possible.

We are also going to require that the output be interleaved samples - this means that we alternate samples from the left and right signal.

## SIMD for Optimization
One of the tools available when optimizing are Single Instruction Multiple Data (SIMD) CPU instructions. They differ from typical CPU instructions by the fact they are able to operate on multiple values at once, in the same time it takes the corresponding scalar instruction to operate on a single value.

Each sample we process in our mixing loop is operated on in an identical way to all other samples. This makes our mixing loop a perfect candidate for optimization using SIMD instructions. We will be able to process four samples at once for the same CPU cost as a single sample.

## SIMD in C
Generating SIMD by hand in C/C++ is well established and no more or less safe than any other code using pointers.

Targeting the SIMD instructions on Intel's X64 architecture, known as SSE, we use intrinsics functions provided by
the compiler. Each instruction function is documented to map to a single assembler instructions.

```C
#include <assert.h>
#include <stdint.h>
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
```

This is only a limited implementation of the loop as it can only handle lengths that are a multiple of four. It also shows that despite the original algorithm being very simple the intrinsic version can quickly become complicated.

## SIMD In Rust
Rust includes a crate in core `core::arch::x86_64` that provides intrinsics the will be familiar to anyone who's worked with intrinsics in C.

All the intrinsic functions are marked **unsafe** as they work outside the memory safety provided by default in Rust.

Is it possible to write safe Rust code and still gain the performance advantages of SIMD?

## Auto-Vectorization
Auto-Vectorization refers to the compiler being able to take a loop, and generate code that uses SIMD instructions to process multiple iterations of the loop at once.

Not every loop is able to be vectorized. There may not be a way to express the code in the loop using the available SIMD instructions on a particular target. Also the compiler has to be able to prove that the SIMD version has exactly the same behavior as the scalar version. In Rust this includes obeying all the type and memory safety requirements. However those same type and memory requirements of Rust can also aid the compiler in being able to auto-vectorize.

### First Attempt at Taking Advantage of Auto-Vectorization in Rust
If we take the signature of our C function and switch from raw pointers to slices, we can generate a version of the loop that processes one sample at a time.

```rust
pub fn mix_mono_to_stereo_1(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    for i in 0..src.len() {
        dst[i * 2 + 0] = src[i] * gain_l;
        dst[i * 2 + 1] = src[i] * gain_r;
    }
}
```

We can then use the very helpful Godbolt compiler explorer site to [preview](https://godbolt.org/z/Das0yY) what the assembler output of this Rust code would be.

```assembly
example::mix_mono_to_stereo_1:
        push    rax
        test    rcx, rcx
        je      .LBB0_5
        mov     r8, rsi
        xor     eax, eax
.LBB0_2:
        lea     rsi, [rax + rax]
        cmp     rsi, r8
        jae     .LBB0_6
        movss   xmm2, dword ptr [rdx + 4*rax]
        movaps  xmm3, xmm2
        mulss   xmm3, xmm0
        movss   dword ptr [rdi + 4*rsi], xmm3
        or      rsi, 1
        cmp     rsi, r8
        jae     .LBB0_8
        add     rax, 1
        mulss   xmm2, xmm1
        movss   dword ptr [rdi + 4*rsi], xmm2
        cmp     rax, rcx
        jb      .LBB0_2
.LBB0_5:
        pop     rax
        ret
.LBB0_6:
        lea     rdi, [rip + .L__unnamed_1]
        jmp     .LBB0_7
.LBB0_8:
        lea     rdi, [rip + .L__unnamed_2]
.LBB0_7:
        mov     rdx, r8
        call    qword ptr [rip + core::panicking::panic_bounds_check@GOTPCREL]
        ud2
```

Looking through the disassembly we can see six sections, marked by the labels `example::mix_mono_to_stereo_1`, `.LBB0_2`, and `.LBB0_5`, to `.LBB0_8`. 

The first section has a label with the name of our function, and is called the function prelude. It contains code to deal with our function arguments and setting up the stack. It also contains a check that if the length of the `src` slice is zero we can exit the function straight away.

`.LBBO_2` is the contents of our loop. You can see the last two instructions are a `cmp` which is a comparison of two number, followed by `jb` which is a jump to the label `.LBB0_2` based on the results of the comparison. This is how we can spot a loop in assembly.

If we examine the rest of the instructions in the `.LBBO_2` block we can see instructions that show the compiler has only generated code to process a single input value each iteration of the loop. In particular we can see the instruction `mulss` which is short for **Mul**tiply **S**calar **S**ingle-precision-float. When we wrote our SIMD version by hand we use the `_mm_mul_ps` instrinsics which generates the `mulps` instruction which is short for **Mul**tiply **P**acked **S**ingle-precision-float. As a naming convention SSE instructions that operate all four elements at once end in `ps` and those that only operate on the first element end in `ss`.

There is some more code in the loop body that gives us a hint about is stopping the compiler from generating code to process four values at once. On lines 9 and 16 we see a `cmp` followed by a `jae` which is a another jump instructions. The labels for the jump are `.LBB0_6` and `.LBB0_8` which we haven't looked at yet. If we look there, along with label `.LBB0_7` which contains a `call` to the function `core::panicking::panic_bounds_check` we can determine that this is code to panic if we write outside the bounds of our destination slice.

What's stopping the compiler from vectorizing our loop is that it wants to check the slice bounds on every single write to the destination slice. So to do that it has to process only a single input value each iteration.

But we didn't see any bounds checks in the input slice. That's because in the Rust code our loop uses the range `i in 0..src.len()`. With that loop statement the compiler is able to known that we never read out of the bounds of our loop slice.

### Second Attempt
Can we prove to compiler that all our writes to the destination slice are within bounds? Our second attempt looks like
```rust
pub fn mix_mono_to_stereo_2(dst: &mut [f32], src: &[f32], gain_l: f32, gain_r: f32) {
    let dst_known_bounds = &mut dst[0..src.len() * 2];
    for i in 0..src.len() {
        dst_known_bounds[i * 2 + 0] = src[i] * gain_l;
        dst_known_bounds[i * 2 + 1] = src[i] * gain_r;
    }
}
```
Here we create a new slice with bounds that are known to the compiler when it's processing this function. Hopefully it will be able to prove based on this new slice being exactly twice as long as the source slice, no access using the index `i * 2 + 0` or `i * 2 + 1` could ever be out of bounds.

Unfortunately if we check the [assembly output](https://godbolt.org/z/UPUWL1) of the compiler it looks very similar to our first attempt. The loop body still uses the `mulss` to perform a single multiply at a time, and there is still a check on the bounds for each access of the destination slice.

### Third Attempt
The way we index the destination slice is too complicated for the compiler to prove that all accesses are in bounds.

It's worth taking a step back and thinking about what the destination indexing code is expressing: The destination slice is twice as long as the input slice, where even indexed value represent the left channel of the signal, and odd indexed values represent the right channel.

That's a lot of implicit assumptions about the structure of the slice. Let's use the type system to make it more explicit to both the compiler and any future readers of our code who aren't familiar with digital audio conventions.

```rust
#[repr(C)]
pub struct StereoSample {
    l: f32,
    r: f32,
}

#[repr(transparent)]
pub struct MonoSample(f32);

pub fn mix_mono_to_stereo_3(dst: &mut [StereoSample], src: &[MonoSample], gain_l: f32, gain_r: f32) {
    let dst_known_bounds = &mut dst[0..src.len()];
    for i in 0..src.len() {
        dst_known_bounds[i].l = src[i].0 * gain_l;
        dst_known_bounds[i].r = src[i].0 * gain_r;
    }
}
```

If we check the [assembler output](https://godbolt.org/z/3etzSu) we can immediately see there is a lot more assembly generated. So much that I won't break it all down. But scanning through we can see there are blocks containing the `mulps`, `unpcklps` and `unpckhps` instructions we used in the hand written instrinsic version. This shows that the compiler has been able to vectorize our loop to process four samples at a time using SIMD.

If we examine the structure of the assembler blocks we can get an rough idea of what it has actually generated. It goes beyond our simple hand written version, and includes multiple stages for processing all samples.

The stages go in order:
1. If the number of samples to process is odd, process a single sample
2. Process two samples at time until the number of samples remaining is a multiple of four
3. Process sixteen samples at time using SIMD until the number of samples remaining is less than sixteen
4. Process four samples at time using SIMD until no samples remain

## Conclusion

By taking advantage of the Rust type system and slice bounds safety we are able to write simple code that produces optimized SIMD output equal to that written by hand.

We can also check the [assembler output](https://godbolt.org/z/nlYe6N) for an AArch64 target and see that we're able to also produce code using Arm's SIMD instruction. This is another advantage over using hand written intrinsics which must be re-written for each target architecture. 