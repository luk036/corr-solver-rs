//! Initial-guess strategies for the cutting-plane drivers.

use ellalgo_rs::arr::Arr;
use ellalgo_rs::ell::Ell;

/// Best-so-far objective seed for a fresh cutting-plane run.
pub const INITIAL_T: f64 = 1e100;

/// Augmented `(coeffs..., t)` ellipsoid for the least-squares optimization.
pub fn lsq_initial_guess(norm_y: f64, m: usize) -> Ell {
    let norm_y2 = 32.0 * norm_y * norm_y;
    let mut val = vec![256.0; m + 1];
    val[m] = norm_y2 * norm_y2;
    let mut x = Arr::new(m + 1);
    x[0] = 4.0;
    x[m] = norm_y2 / 2.0;
    Ell::new(Arr::from(val), x)
}

/// Plain coefficient ellipsoid for the maximum-likelihood fit.
pub fn mle_initial_guess(m: usize) -> Ell {
    let mut x = Arr::new(m);
    x[0] = 4.0;
    Ell::new_with_scalar(500.0, x)
}

/// Plain coefficient ellipsoid for one CCP round, centred on `x`.
pub fn cccp_initial_guess(x: &Arr) -> Ell {
    Ell::new_with_scalar(100.0, x.clone())
}
