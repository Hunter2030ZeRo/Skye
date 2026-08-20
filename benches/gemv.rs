use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use half::f16;

use skye::format::NvFp4Matrix;
use skye::kernel::scalar::gemv_scalar;

#[cfg(target_arch = "x86_64")]
use skye::kernel::avx::gemv_avx2;

fn make_weights(rows: usize, cols: usize) -> NvFp4Matrix {
    assert_eq!(cols % 16, 0);

    let num_weights = rows * cols;

    // E2M1 값 두 개 / byte.
    //
    // 0x23:
    // low  nibble = 0x3 = +1.5
    // high nibble = 0x2 = +1.0
    //
    // 데이터가 전부 0인 비현실적인 경우는 피한다.
    let data = vec![0x23u8; num_weights / 2];

    // E4M3 0x38 = 1.0
    let scales = vec![0x38u8; num_weights / 16];

    let weights = NvFp4Matrix::new(rows, cols, data, scales, 1.0);

    weights
}

fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("nvfp4_w4a16_gemv");

    // 오늘 빠르게 확인하려면 이 두 개면 충분.
    //
    // 최종 결과 낼 때 4096x4096 등을 더 추가.
    let shapes = [(1024usize, 1024usize), (4096usize, 4096usize)];

    for &(rows, cols) in &shapes {
        let weights = make_weights(rows, cols);

        let x = vec![f16::from_f32(1.0); cols];

        let mut y_scalar = vec![0.0f32; rows];
        let mut y_avx2 = vec![0.0f32; rows];

        // 처리한 logical weight 개수.
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let shape = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("scalar", &shape), &(), |b, _| {
            b.iter(|| {
                gemv_scalar(black_box(&weights), black_box(&x), black_box(&mut y_scalar));

                black_box(&y_scalar);
            });
        });

        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("fma")
        {
            group.bench_with_input(BenchmarkId::new("avx2_fused", &shape), &(), |b, _| {
                b.iter(|| unsafe {
                    gemv_avx2(black_box(&weights), black_box(&x), black_box(&mut y_avx2));

                    black_box(&y_avx2);
                });
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_gemv
}

criterion_main!(benches);
