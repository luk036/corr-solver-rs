//! Polynomial basis plus re-exports of the geometry and site generators.

pub use crate::geometry::construct_distance_matrix;
pub use crate::sites::{create_2d_isotropic, create_2d_isotropic_with, create_2d_sites};

use ellalgo_rs::arr::Arr;
use ndarray::Array2;

/// Polynomial basis `sigma[k] = D.^k`, the Hadamard powers of the distance matrix.
pub fn construct_poly_matrix(site: &Arr, m: usize) -> Vec<Array2<f64>> {
    let n = site.rows();
    let d1 = construct_distance_matrix(site);
    let mut d = Array2::ones((n, n));
    let mut sig: Vec<Array2<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        if i > 0 {
            for r in 0..n {
                for c in 0..n {
                    d[[r, c]] *= d1.get(r, c);
                }
            }
        }
        sig.push(d.clone());
    }
    sig
}
