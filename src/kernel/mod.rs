#[cfg(target_arch = "x86_64")]
pub mod avx;
#[cfg(target_arch = "x86_64")]
pub mod fp16;
pub mod scalar;
