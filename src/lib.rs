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
//! - [`linalg`] - Linear algebra helpers on [`ellalgo_rs::arr::Arr`]
//! - [`convert`] - Conversions between `Arr` and [`ndarray::Array2`]
//! - [`ndops`] - Matrix helpers on [`ndarray::Array2`]
//! - [`kernels`] - Radial covariance kernels
//! - [`geometry`] - Pairwise distance geometry
//! - [`sites`] - Site layouts and biased sample covariance generation
//! - [`layouts`] - Initial-guess strategies for the cutting-plane drivers
//! - [`gmi_oracle`] - General Matrix Inequality oracle
//! - [`qmi_oracle`] - Quadratic Matrix Inequality oracle
//! - [`lmi0_oracle`] - LMI oracle with zero constant term (F(x) ⪰ 0)
//! - [`lsq_oracle`] - Least-squares correlation oracle
//! - [`mle_oracle`] - Maximum likelihood estimation oracle
//! - [`mle_common`] - Scratch and cut helpers shared by the MLE oracles
//! - [`fitting`] - Public fitting drivers
//! - [`bspline`] - Quadratic B-spline basis and the monotone oracle
//! - [`eigen`] - Jacobi eigendecomposition and design conditioning
//! - [`halton`] - Halton low-discrepancy site generator
//! - [`corr_helper`] - Polynomial basis plus re-exported geometry/site helpers

#![allow(non_snake_case)]

pub mod bspline;
pub mod convert;
pub mod corr_helper;
pub mod eigen;
pub mod fitting;
pub mod geometry;
pub mod gmi_oracle;
pub mod halton;
pub mod kernels;
pub mod layouts;
pub mod linalg;
pub mod lmi0_oracle;
pub mod lsq_oracle;
pub mod mle_common;
pub mod mle_oracle;
pub mod ndops;
pub mod qmi_oracle;
pub mod sites;
