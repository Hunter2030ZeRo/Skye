use crate::format::NvFp4Matrix;
use half::f16;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn dot_block_avx2(
    packed_weights: *const u8,
    activations: *const f16,
    block_scale: f32,
) -> __m256 {
    unsafe {
        let packed = _mm_loadu_si64(packed_weights);

        let nibble_mask = _mm_set1_epi8(0x0f);

        let low = _mm_and_si128(packed, nibble_mask);

        let high = _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask);

        let codes = _mm_unpacklo_epi8(low, high);

        let lut128 = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, -0, -1, -2, -3, -4, -6, -8, -12);

        let lut256 = _mm256_broadcastsi128_si256(lut128);

        let codes256 = _mm256_broadcastsi128_si256(codes);

        let decoded256 = _mm256_shuffle_epi8(lut256, codes256);

        let decoded = _mm256_castsi256_si128(decoded256);

        let w0_i32 = _mm256_cvtepi8_epi32(decoded);

        let upper8 = _mm_srli_si128::<8>(decoded);

        let w1_i32 = _mm256_cvtepi8_epi32(upper8);

        let w0 = _mm256_cvtepi32_ps(w0_i32);

        let w1 = _mm256_cvtepi32_ps(w1_i32);

        let x0_bits = _mm_loadu_si128(activations as *const __m128i);

        let x1_bits = _mm_loadu_si128(activations.add(8) as *const __m128i);

        let x0 = _mm256_cvtph_ps(x0_bits);
        let x1 = _mm256_cvtph_ps(x1_bits);

        let products0 = _mm256_mul_ps(w0, x0);

        let products = _mm256_fmadd_ps(w1, x1, products0);

        let scale = _mm256_set1_ps(block_scale * 0.5);

        let result = _mm256_mul_ps(products, scale);

        result
    }
}

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

/// The host CPU must support AVX2, F16C and FMA instruction extensions.
#[inline]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn gemv_avx2(weights: &NvFp4Matrix, x: &[f16], y: &mut [f32]) {
    assert_eq!(x.len(), weights.cols);
    assert_eq!(y.len(), weights.rows);
    assert_eq!(weights.cols % 16, 0);

    let blocks_per_row = weights.cols / 16;

    for row in 0..weights.rows {
        let mut vec_acc = _mm256_setzero_ps();

        for block in 0..blocks_per_row {
            let col_start = block * 16;

            let weight_index = row * weights.cols + col_start;

            let packed_index = weight_index / 2;

            let scale_index = row * blocks_per_row + block;

            let block_scale =
                crate::kernel::scalar::decode_nvfp4_scale(weights.scales[scale_index]);

            let block_result = unsafe {
                crate::kernel::avx::dot_block_avx2(
                    weights.data.as_ptr().add(packed_index),
                    x.as_ptr().add(col_start),
                    block_scale,
                )
            };

            vec_acc = _mm256_add_ps(vec_acc, block_result);
        }

        unsafe {
            y[row] = horizontal_sum_avx(vec_acc) * weights.global_scale;
        }
    }
}
