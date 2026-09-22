#![allow(non_snake_case)]
use corr_solver_rs::convert::arr_to_ndarray;
use corr_solver_rs::corr_helper::{create_2d_isotropic, create_2d_sites};
use corr_solver_rs::fitting::{lsq_corr_poly, mle_corr_poly};

#[test]
fn test_create_2d_isotropic() {
    let site = create_2d_sites(5, 4);
    let y = create_2d_isotropic(&site, 3000);
    assert!(y.rows() > 0);
    assert_eq!(y.rows(), 20);
    assert_eq!(y.cols(), 20);
}

#[test]
fn test_lsq_corr_fn() {
    let site = create_2d_sites(10, 8);
    let y = create_2d_isotropic(&site, 3000);
    let fit = lsq_corr_poly(&arr_to_ndarray(&y), &site, 4);
    assert!(fit.ok);
    assert!(fit.coeffs.size() > 0);
    assert!(fit.coeffs[0] >= 0.0);
    assert!(fit.iters >= 440);
    assert!(fit.iters <= 1100);
}

#[test]
fn test_mle_corr_fn() {
    let site = create_2d_sites(10, 8);
    let y = create_2d_isotropic(&site, 3000);
    let fit = mle_corr_poly(&arr_to_ndarray(&y), &site, 4);
    assert!(fit.ok);
    assert!(fit.coeffs.size() > 0);
    assert!(fit.coeffs[0] >= 0.0);
    assert!(fit.iters >= 50);
    assert!(fit.iters <= 500);
}
