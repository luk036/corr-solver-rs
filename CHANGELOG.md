# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-18

### Added
- Initial Rust port of corr-solver-cpp
- LSQ correlation solver (LsqOracle implementing OracleOptim)
- MLE correlation solver (MleOracle implementing OracleOptim)
- QMI oracle with interior mutability (Cell/RefCell) for caching
- LMI0 oracle for linear matrix inequality constraints
- GMI oracle with HOmni trait
- Linear algebra utilities (trace, matmul, cholesky, sym_invert)
- Polynomial matrix construction helper
- Integration tests for LSQ and MLE solvers
- Benchmark binary for performance comparison
- GitHub Actions CI (test, rustfmt, clippy, docs)
- Code coverage with cargo-llvm-cov
- Security audit workflow

### Performance
- Reorder `matmul` to i-k-j so both operands are contiguous and the inner loop
  auto-vectorizes; the previous order walked the right operand one column at a
  time (stride n). Same change in `mle_oracle::ndarray_matmul`, which now
  normalizes to standard layout first — `invR.t().to_owned()` yields an F-order
  array, so `as_slice()` would otherwise fail.
  Measured on the 80-site problem: MLE 1.29 s -> 0.88 s (-31%), total
  2.42 s -> 2.01 s (-16%), iteration counts unchanged.
- `trace` no longer builds a temporary diagonal vector just to reduce it; added a
  `frob_inner` helper.

### Dependencies
- `ellalgo-rs` 0.1.8 -> 0.1.10
- `lmi-solver-rs` 0.1.2 -> 0.1.3
