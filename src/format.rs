#[derive(Debug, Clone)]
pub struct NvFp4Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,

    pub(crate) data: Vec<u8>,

    pub(crate) scales: Vec<u8>,

    pub(crate) global_scale: f32,
}

impl NvFp4Matrix {
    pub const BLOCK_SIZE: usize = 16;

    pub fn new(
        rows: usize,
        cols: usize,
        data: Vec<u8>,
        scales: Vec<u8>,
        global_scale: f32,
    ) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        assert!(global_scale.is_finite());
        assert!(global_scale >= 0.0);
        assert_eq!(cols % Self::BLOCK_SIZE, 0);

        let expected_data_len = rows
            .checked_mul(cols / 2)
            .expect("NVFP4 data dimensions overflow");

        let expected_scales_len = rows
            .checked_mul(cols / Self::BLOCK_SIZE)
            .expect("NVFP4 scale dimension overflow");

        assert_eq!(data.len(), expected_data_len);
        assert_eq!(scales.len(), expected_scales_len);
        Self {
            rows: rows,
            cols: cols,
            data: data,
            scales: scales,
            global_scale: global_scale,
        }
    }

    #[inline]
    pub fn packed_byte(&self, index: usize) -> u8 {
        self.data[index >> 1]
    }

    #[inline]
    pub fn nibble(&self, index: usize) -> u8 {
        let byte = self.packed_byte(index);

        if index & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    }
}
