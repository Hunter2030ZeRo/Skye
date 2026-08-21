use half::f16;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps::<1>(v);

    let sum128 = _mm_add_ps(low, high);
    let high64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, high64);

    let lane1 = _mm_shuffle_ps::<0x55>(sum64, sum64);
    let total = _mm_add_ss(sum64, lane1);

    _mm_cvtss_f32(total)
}

#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn gemv_fp16_avx2(weights: &[f16], rows: usize, cols: usize, x: &[f16], y: &mut [f32]) {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);

    // MVP: LLM 차원은 대부분 8의 배수이므로 tail 처리 생략
    assert_eq!(cols % 8, 0);

    for row in 0..rows {
        let mut acc = _mm256_setzero_ps();

        let row_offset = row * cols;

        unsafe {
            for col in (0..cols).step_by(8) {
                let w_ptr = weights.as_ptr().add(row_offset + col);

                let x_ptr = x.as_ptr().add(col);

                // 8 × FP16 = 16 bytes
                let w_bits = _mm_loadu_si128(w_ptr as *const __m128i);

                let x_bits = _mm_loadu_si128(x_ptr as *const __m128i);

                // 8 × FP16 -> 8 × FP32
                let w = _mm256_cvtph_ps(w_bits);
                let xv = _mm256_cvtph_ps(x_bits);

                acc = _mm256_fmadd_ps(w, xv, acc);
            }

            y[row] = horizontal_sum_avx(acc);
        }
    }
}
