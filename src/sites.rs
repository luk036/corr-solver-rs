//! Site layouts and biased sample covariance generation.

use crate::kernels::exponential_kernel;
use crate::linalg;
use ellalgo_rs::arr::{linspace, Arr};
use rand::rngs::StdRng;

/// Uniform 2D grid of `nx` by `ny` sites over `[0, 10] x [0, 8]`.
pub fn create_2d_sites(nx: usize, ny: usize) -> Arr {
    let sx = linspace(0.0, 10.0, nx);
    let sy = linspace(0.0, 8.0, ny);
    let (xx, yy) = linalg::meshgrid(&sx, &sy);
    let fx = linalg::flatten(&xx);
    let fy = linalg::flatten(&yy);
    let st = linalg::stack(&fx, &fy);
    linalg::transpose(&st)
}

/// Biased sample covariance for an exponential kernel over site distances.
///
/// Uses the historical deterministic seed, so repeated calls reproduce the
/// reference data.
pub fn create_2d_isotropic(site: &Arr, n: usize) -> Arr {
    create_2d_isotropic_with(site, n, &mut linalg::seeded_rng(5))
}

/// As [`create_2d_isotropic`], drawing from an explicit RNG.
pub fn create_2d_isotropic_with(site: &Arr, n: usize, rng: &mut StdRng) -> Arr {
    let n_sites = site.rows();
    let sdkern = 0.3;
    let var = 2.0;
    let tau = 0.00001;
    let mut sig = Arr::zeros(n_sites, n_sites);
    for i in 0..n_sites {
        for j in i..n_sites {
            let mut d = 0.0;
            for k in 0..site.cols() {
                let diff = site.get(j, k) - site.get(i, k);
                d += diff * diff;
            }
            let val = exponential_kernel(d.sqrt(), sdkern);
            sig.set(i, j, val);
            sig.set(j, i, val);
        }
    }

    let a = linalg::cholesky(&sig);
    let mut y = Arr::zeros(n_sites, n_sites);
    for _ in 0..n {
        let x = var * linalg::randn_with(n_sites, rng);
        let mut y_tmp = Arr::new(n_sites);
        for i in 0..n_sites {
            let mut s = 0.0;
            for j in 0..n_sites {
                s += a.get(i, j) * x[j];
            }
            y_tmp[i] = s;
        }
        let noise = linalg::randn_with(n_sites, rng);
        for i in 0..n_sites {
            y_tmp[i] += tau * noise[i];
        }
        for i in 0..n_sites {
            for j in 0..n_sites {
                let v = y.get(i, j) + y_tmp[i] * y_tmp[j];
                y.set(i, j, v);
            }
        }
    }
    let nf = n as f64;
    for i in 0..n_sites {
        for j in 0..n_sites {
            y.set(i, j, y.get(i, j) / nf);
        }
    }
    y
}
