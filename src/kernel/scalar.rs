use crate::format::NvFp4Matrix;
use half::f16;

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const E4M3_POW2: [f32; 15] = [
    1.0 / 64.0, // 2^-6, exponent field = 1
    1.0 / 32.0, // 2^-5
    1.0 / 16.0, // 2^-4
    1.0 / 8.0,  // 2^-3
    1.0 / 4.0,  // 2^-2
    1.0 / 2.0,  // 2^-1
    1.0,        // 2^0
    2.0,        // 2^1
    4.0,        // 2^2
    8.0,        // 2^3
    16.0,       // 2^4
    32.0,       // 2^5
    64.0,       // 2^6
    128.0,      // 2^7
    256.0,      // 2^8
];

const fn decode_e4m3_const(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;

    if exponent == 0 {
        mantissa as f32 / 512.0
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        let significand = 1.0 + mantissa as f32 / 8.0;

        significand * E4M3_POW2[(exponent - 1) as usize]
    }
}

const fn build_nvfp4_scale_lut() -> [f32; 128] {
    let mut lut = [0.0f32; 128];

    let mut i = 0;
    while i < 128 {
        lut[i] = decode_e4m3_const(i as u8);
        i += 1;
    }

    lut
}

const NVFP4_SCALE_LUT: [f32; 128] = build_nvfp4_scale_lut();

#[inline(always)]
pub fn decode_e2m1_scalar(bits: u8) -> f32 {
    E2M1_LUT[(bits & 0x0f) as usize]
}

#[inline(always)]
pub fn decode_nvfp4_scale(bits: u8) -> f32 {
    NVFP4_SCALE_LUT[(bits & 0x7f) as usize]
}

pub fn gemv_scalar(weights: &NvFp4Matrix, x: &[f16], y: &mut [f32]) {
    assert_eq!(x.len(), weights.cols);
    assert_eq!(y.len(), weights.rows);
    assert_eq!(weights.cols % 16, 0);

    let blocks_per_row = weights.cols / 16;

    for row in 0..weights.rows {
        let mut acc = 0.0f32;

        for block in 0..blocks_per_row {
            let col_start = block * 16;

            let weight_index = row * weights.cols + col_start;
            let packed_index = weight_index / 2;

            let scale_index = row * blocks_per_row + block;

            let mut block_acc = 0.0f32;

            for i in 0..16 {
                let byte = weights.data[packed_index + i / 2];

                let code = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };

                let w = decode_e2m1_scalar(code);
                let a = x[col_start + i].to_f32();

                block_acc += w * a;
            }

            let scale = decode_nvfp4_scale(weights.scales[scale_index]);

            acc += block_acc * scale;
        }

        y[row] = acc * weights.global_scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_e2m1() {
        let expected = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];

        for (bits, expected) in expected.iter().enumerate() {
            assert_eq!(decode_e2m1_scalar(bits as u8), *expected);
        }
    }
}
