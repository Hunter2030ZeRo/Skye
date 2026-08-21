# Security and safety review: NVFP4 AVX2 prototype

**Scope:** `src/format.rs`, `src/kernel/avx.rs`, `src/kernel/scalar.rs`, and current tests.

**Skill-perspective check:** Ran `omo:remove-ai-slops` and `omo:programming` plus its Rust and Rust-UB references. The diff has no deletion-only, tautological, implementation-mirroring, or prompt-text tests. It does violate the programming/unsafe perspective: the public unsafe API lacks a `# Safety` contract and the raw-load blocks have no specific `SAFETY:` justification. No needless production parsing or normalization was introduced.

## Findings

### HIGH

- **AVX2 unsafe contract is not fully documented** — `src/kernel/avx.rs:79-82` documents only CPU-feature support; the public `unsafe fn` has no rustdoc `# Safety` section. `src/kernel/avx.rs:12-13`, `src/kernel/avx.rs:43-45`, and `src/kernel/avx.rs:104-109` perform raw pointer loads/arithmetic with no nearby `SAFETY:` explanation. The intended preconditions need to explicitly cover AVX2/F16C/FMA availability and why a matrix constructed through the public API has at least eight packed bytes and sixteen `f16`s at every block. Clippy independently reports `missing_safety_doc` at line 82.

### MEDIUM

- **Release integer overflow weakens the matrix invariant that backs raw loads** — `src/format.rs:28-29` computes packed-data and scale lengths with unchecked `usize` multiplication; `src/kernel/avx.rs:95` and `99` repeat unchecked multiplication for block offsets. Debug builds panic on overflow, but release builds wrap. The constructor therefore proves equality only to wrapped lengths, not `rows * cols / 2` and `rows * cols / 16`. `src/kernel/avx.rs:106` then calls `.add(packed_index)` and line 13 reads eight bytes without an ordinary slice bound check. This is UB category 10/13 if the raw-load range is not valid. Most readily constructible malformed shapes fail safely earlier at the bounds-checked scale access on line 102; this is not itself a direct practical UB exploit on normal 64-bit machines because a shape that keeps scales in-bounds while wrapping data requires exabyte-scale `scales`. It remains an unsound proof and a denial-of-service allocation/panic risk, not a demonstrated current low-memory remote exploit.

- **`global_scale` finiteness checks are logically ineffective** — `src/format.rs:25` accepts NaN because IEEE NaN is unequal to every value, including itself; line 26 rejects only positive infinity and accepts negative infinity. `NaN`, `+inf`, and `-inf` propagate through `src/kernel/avx.rs:56,115` and scalar line 66. This is numerical-contract failure, not memory UB. Encoded per-block scale `0x7f` deliberately decodes to NaN (`src/kernel/scalar.rs:22-23` and `tests/correctness.rs:26`), so that behavior should be explicitly documented or rejected consistently at the API boundary.

### LOW

- **Unsafe behavior is untested** — all existing tests exercise scalar decoding/GEMV (`tests/correctness.rs:1-43`) and do not invoke `gemv_avx2` or validate its CPU-dispatch contract. This does not create UB alone, but it leaves the AVX2 raw-load equivalence and edge conditions unverified. Miri is not installed for the available nightly toolchain, and Miri cannot execute AVX intrinsics directly; an AVX2 hardware-gated differential test plus Miri-testable safe range validation would be appropriate.

### CRITICAL

None identified.

## Verification

- `cargo test`: PASS — 4 tests across 3 suites.
- `cargo test --release`: PASS — 4 tests across 3 suites; no AVX2 coverage.
- `cargo clippy --all-targets --all-features -- -D warnings`: FAIL — reports the missing public unsafe safety docs above (and non-safety style warnings).
- `cargo +nightly miri --version`: unavailable; `cargo-miri.exe` is not installed for `nightly-x86_64-pc-windows-msvc`.

## Verdict

`codeQualityStatus: BLOCK`

`recommendation: REQUEST_CHANGES`

**Blockers:** document every public/raw unsafe precondition with specific `# Safety` / `SAFETY:` contracts; make constructor and kernel size arithmetic overflow-safe (or explicitly bound dimensions before the unsafe path); correct the finiteness policy for `global_scale`; add an AVX2-gated regression/differential test.
