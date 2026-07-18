# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
