# corr-solver-rs

[![CI](https://github.com/luk036/corr-solver-rs/workflows/CI/badge.svg)](https://github.com/luk036/corr-solver-rs/actions)
[![codecov](https://codecov.io/gh/luk036/corr-solver-rs/branch/main/graph/badge.svg?token=ZZ4KpPfmdF)](https://codecov.io/gh/luk036/corr-solver-rs)

Rust implementation of correlation matrix solvers using the ellipsoid method.

Solves correlation matrix identification problems by formulating them as semidefinite programs and solving via the cutting-plane / ellipsoid method. Two solvers are provided:

- **LSQ (Least Squares)**: Minimizes the Frobenius-norm error between a parameterized correlation model and observed data.
- **MLE (Maximum Likelihood)**: Maximizes the log-likelihood of observed data under a parameterized correlation model.

Both solvers enforce positive-definiteness of the resulting correlation matrix via quadratic matrix inequality (QMI) constraints.

## Quick Start

```rust
use corr_solver_rs::lsq_oracle::LsqOracle;
use corr_solver_rs::corr_helper::test_lsq_corr_fn;
use ellalgo_rs::cutting_plane::cutting_plane_optim;
use ellalgo_rs::ell::Ell;

let (x_best, num_iters) = test_lsq_corr_fn(42);
println!("Best solution: {:?}", x_best);
println!("Iterations: {}", num_iters);
```

Add to `Cargo.toml`:

```toml
[dependencies]
corr-solver-rs = "0.1"
```

## Installation

### Cargo

```shell
cargo install corr-solver-rs
```

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Related Projects

### Polyglot Implementations

- [**corr-solver-cpp**](https://github.com/luk036/corr-solver-cpp) - C++ version
- [**corr-solver**](https://github.com/luk036/corr-solver) - Python version

### Dependencies

- [**ellalgo-rs**](https://github.com/luk036/ellalgo-rs) - Ellipsoid method in Rust
- [**lmi-solver-rs**](https://github.com/luk036/lmi-solver-rs) - LMI solver in Rust
