#![allow(non_snake_case)]
use corr_solver_rs::corr_helper::{construct_poly_matrix, create_2d_isotropic, create_2d_sites};
use corr_solver_rs::linalg;
use corr_solver_rs::lsq_oracle::LsqOracle;
use corr_solver_rs::mle_oracle::MleOracle;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options};
use ellalgo_rs::ell::Ell;
use ndarray::Array2;

fn lsq_corr_core2(Y: &Arr, m: usize, omega: &mut LsqOracle) -> (Arr, usize) {
    let norm_y = linalg::norm(Y);
    let norm_y2 = 32.0 * norm_y * norm_y;
    let mut val = vec![256.0; m + 1];
    val[m] = norm_y2 * norm_y2;
    let mut x = Arr::new(m + 1);
    x[0] = 4.0;
    x[m] = norm_y2 / 2.0;
    let mut ellip = Ell::new(Arr::from(val), x);
    let mut t = 1e100;
    let (x_best, num_iters) = cutting_plane_optim(omega, &mut ellip, &mut t, &Options::default());
    let mut a = Arr::new(m);
    if let Some(xb) = x_best {
        for i in 0..m {
            a[i] = xb[i];
        }
    }
    (a, num_iters)
}

fn lsq_corr_poly2(Y: &Arr, site: &Arr, m: usize) -> (Arr, usize) {
    let sig_vec = construct_poly_matrix(site, m);
    let f0 = arr_to_ndarray(Y);
    let mut omega = LsqOracle::new(Y.rows(), sig_vec, f0);
    lsq_corr_core2(Y, m, &mut omega)
}

fn mle_corr_core(m: usize, omega: &mut MleOracle) -> (Option<Arr>, usize) {
    let mut x = Arr::new(m);
    x[0] = 4.0;
    let mut ellip = Ell::new_with_scalar(500.0, x);
    let mut t = 1e100;
    let (x_best, num_iters) = cutting_plane_optim(omega, &mut ellip, &mut t, &Options::default());
    (x_best, num_iters)
}

fn mle_corr_poly(Y: &Arr, site: &Arr, m: usize) -> (Option<Arr>, usize) {
    let sig_vec = construct_poly_matrix(site, m);
    let f0 = arr_to_ndarray(Y);
    let mut omega = MleOracle::new(sig_vec, f0);
    mle_corr_core(m, &mut omega)
}

fn arr_to_ndarray(a: &Arr) -> Array2<f64> {
    let n = a.rows();
    let m = a.cols();
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a.get(i, j);
        }
    }
    out
}

#[test]
fn test_create_2d_isotropic() {
    let site = create_2d_sites(5, 4);
    let y = create_2d_isotropic(&site, 3000);
    assert!(y.rows() > 0);
}

#[test]
fn test_lsq_corr_fn() {
    let site = create_2d_sites(10, 8);
    let y = create_2d_isotropic(&site, 3000);
    let (coeffs, num_iters) = lsq_corr_poly2(&y, &site, 4);
    assert!(coeffs.size() > 0);
    assert!(coeffs[0] >= 0.0);
    assert!(num_iters >= 440);
    assert!(num_iters <= 1100);
}

#[test]
fn test_mle_corr_fn() {
    let site = create_2d_sites(10, 8);
    let y = create_2d_isotropic(&site, 3000);
    let (coeffs, num_iters) = mle_corr_poly(&y, &site, 4);
    assert!(coeffs.is_some());
    let c = coeffs.unwrap();
    assert!(c.size() > 0);
    assert!(c[0] >= 0.0);
    assert!(num_iters >= 50);
    assert!(num_iters <= 500);
}
