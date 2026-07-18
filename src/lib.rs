//! # corr-solver-rs
//!
//! Correlation solver in Rust.
//!
//! This crate provides oracles for solving correlation estimation problems
//! using the ellipsoid method from [ellalgo-rs](https://github.com/luk036/ellalgo-rs)
//! and LMI oracles from [lmi-solver-rs](https://github.com/luk036/lmi-solver-rs).
//!
//! ## Modules
//!
//! - [`linalg`] - Linear algebra helpers (transpose, matmul, Cholesky, inverse, etc.)
//! - [`gmi_oracle`] - General Matrix Inequality oracle
//! - [`qmi_oracle`] - Quadratic Matrix Inequality oracle
//! - [`lmi0_oracle`] - LMI oracle with zero constant term (F(x) ⪰ 0)
//! - [`lsq_oracle`] - Least-squares correlation oracle
//! - [`mle_oracle`] - Maximum likelihood estimation oracle
//! - [`corr_helper`] - Helper functions (create sites, isotropic data, etc.)

#![allow(non_snake_case)]

pub mod corr_helper;
pub mod gmi_oracle;
pub mod linalg;
pub mod lmi0_oracle;
pub mod lsq_oracle;
pub mod mle_oracle;
pub mod qmi_oracle;
