use half::f16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use skye::format::NvFp4Matrix;

#[cfg(target_arch = "x86_64")]
use skye::kernel::avx::gemv_avx2;

use skye::kernel::scalar::gemv_scalar;

#[test]
fn test_decode_e2m1_scalar() {
    let expected = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    for (bits, expected) in expected.iter().enumerate() {
        assert_eq!(
            skye::kernel::scalar::decode_e2m1_scalar(bits as u8),
            *expected
        );
    }
}

#[test]
fn test_decode_nvfp4_scale() {
    use skye::kernel::scalar::decode_nvfp4_scale;

    assert_eq!(decode_nvfp4_scale(0x00), 0.0);
    assert_eq!(decode_nvfp4_scale(0x38), 1.0);
    assert_eq!(decode_nvfp4_scale(0x40), 2.0);
    assert_eq!(decode_nvfp4_scale(0x48), 4.0);

    assert!(decode_nvfp4_scale(0x7f).is_nan());
}

#[test]
fn test_gemv_scalar_single_block() {
    let data = vec![0x22; 8];

    let scales = vec![0x38];

    let weights = skye::format::NvFp4Matrix::new(1, 16, data, scales, 1.0);

    let x = vec![half::f16::from_f32(1.0); 16];

    let mut y = vec![0.0f32; 1];

    gemv_scalar(&weights, &x, &mut y);

    assert_eq!(y[0], 16.0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_matches_scalar_random() {
    if !std::arch::is_x86_feature_detected!("avx2")
        || !std::arch::is_x86_feature_detected!("f16c")
        || !std::arch::is_x86_feature_detected!("fma")
    {
        return;
    }

    let mut rng = StdRng::seed_from_u64(0x5A17_E);

    for &(rows, cols) in &[(1usize, 16usize), (8, 256), (32, 1024)] {
        let num_weights = rows * cols;

        // FP4 code 두 개를 한 byte에 packing
        let mut data = Vec::with_capacity(num_weights / 2);

        for _ in 0..num_weights / 2 {
            let low: u8 = rng.random_range(0..16);
            let high: u8 = rng.random_range(0..16);

            data.push(low | (high << 4));
        }

        // NaN 등이 없는, 알고 있는 E4M3 scale들
        let scale_codes = [
            0x30u8, // 0.5
            0x34,   // 0.75
            0x38,   // 1.0
            0x3c,   // 1.5
            0x40,   // 2.0
        ];

        let scales = (0..num_weights / 16)
            .map(|_| {
                let index = rng.random_range(0..scale_codes.len());
                scale_codes[index]
            })
            .collect();

        let weights = NvFp4Matrix::new(rows, cols, data, scales, 0.75);

        let x: Vec<f16> = (0..cols)
            .map(|_| f16::from_f32(rng.random_range(-2.0f32..2.0f32)))
            .collect();

        let mut scalar = vec![0.0f32; rows];
        let mut avx2 = vec![0.0f32; rows];

        gemv_scalar(&weights, &x, &mut scalar);

        unsafe {
            gemv_avx2(&weights, &x, &mut avx2);
        }

        for row in 0..rows {
            let a = scalar[row];
            let b = avx2[row];

            let error = (a - b).abs();

            // SIMD에서는 누산 순서가 달라지므로
            // bit-identical 결과를 요구하면 안 됨.
            let tolerance = 1e-3 + 1e-4 * a.abs();

            assert!(
                error <= tolerance,
                "shape={rows}x{cols}, row={row}, \
                scalar={a}, avx2={b}, error={error}"
            );
        }
    }
}
