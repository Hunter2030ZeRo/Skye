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
