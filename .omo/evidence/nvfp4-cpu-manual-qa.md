# Manual QA: NvFp4Matrix constructor and GEMV

Date: 2026-08-20
Surface: Rust library API, invoked through temporary Cargo harness at `qa-harness/src/main.rs` (path dependency to `C:\Users\ss_ch\nvfp4-cpu`).

## Evidence artifacts

- `A1`: `rtk cargo test` with isolated `CARGO_TARGET_DIR`; existing unit/integration tests: 4 passed.
- `A2`: debug harness invocation: `rtk pwsh -NoProfile -Command "... cargo run --manifest-path ...qa-harness/Cargo.toml"`; constructor matrix and scalar/AVX2 output.
- `A3`: release harness invocation: `rtk pwsh -NoProfile -Command "... cargo run --release --manifest-path ...qa-harness/Cargo.toml"`; release overflow output.

## surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| C1 | valid shape | `NvFp4Matrix::new(1,16,vec![0;8],vec![0x38],1.0)` | debug harness `cargo run` | PASS, accepted | A2 |
| C2 | bad data length | `new(1,16,vec![0;7],vec![0x38],1.0)` | debug harness `cargo run` | PASS, rejected by assertion at format.rs:28 | A2 |
| C3 | bad scale length | `new(1,16,vec![0;8],vec![],1.0)` | debug harness `cargo run` | PASS, rejected by assertion at format.rs:29 | A2 |
| C4 | cols not multiple of 16 | `new(1,15,vec![0;7],vec![],1.0)` | debug harness `cargo run` | PASS, rejected at format.rs:27 | A2 |
| C5 | zero rows | `new(0,16,vec![],vec![],1.0)` | debug harness `cargo run` | PASS, rejected at format.rs:23 | A2 |
| C6 | zero cols | `new(1,0,vec![],vec![],1.0)` | debug harness `cargo run` | PASS, rejected at format.rs:24 | A2 |
| C7 | NaN global scale | `new(1,16,vec![0;8],vec![0x38],f32::NAN)` | debug harness `cargo run` | FAIL, accepted (assert_ne NaN is ineffective) | A2, A3 |
| C8 | +Inf global scale | same with `f32::INFINITY` | debug harness `cargo run` | PASS, rejected at format.rs:26 | A2, A3 |
| C9 | -Inf global scale | same with `f32::NEG_INFINITY` | debug harness `cargo run` | FAIL, accepted | A2, A3 |
| C10 | negative global scale | same with `-1.0` | debug harness `cargo run` | PASS, accepted as requested behavior | A2, A3 |
| C11 | release overflow | `new(usize::MAX,16,vec![],vec![],1.0)` | release harness `cargo run --release` | PASS for observed safety outcome: multiplication wrapped (`left: 0`, expected `18446744073709551608`) and constructor rejected; no release arithmetic panic | A3 |
| C12 | scalar numerical | 1x16 packed `0x22`, scale `0x38`, x all `f16(1.0)` | debug harness `cargo run` | PASS, `y=[16.0]` | A2 |
| C13 | AVX2 numerical | same; `is_x86_feature_detected!(avx2,f16c,fma)` true then unsafe `gemv_avx2` | debug harness `cargo run` | PASS, `y=[16.0]`, matches scalar | A2 |

## adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| A-C7 | global scale finite | NaN input | reject non-finite scale | FAIL: accepted | A2 |
| A-C9 | global scale finite | negative infinity | reject non-finite scale | FAIL: accepted | A2 |
| A-C11 | overflow safety | release usize multiplication overflow | reject safely / no UB | PASS observed assertion rejection after wrapping; no unsafe acceptance | A3 |
| A-C13 | numerical parity | AVX2 vs scalar | equal result on AVX2/F16C/FMA host | PASS, both 16.0 | A2 |

## artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | test-log | Existing cargo unit/integration tests, 4 passed | command output (isolated target) |
| A2 | harness-log | Debug constructor and numerical harness output | `qa-harness` run output |
| A3 | harness-log | Release constructor overflow harness output | `qa-harness` release run output |
