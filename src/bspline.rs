//! Quadratic B-spline basis with clamped knots and scipy-compatible
//! extrapolation, plus the monotone-decreasing coefficient oracle.

use crate::corr_helper::construct_distance_matrix;
use ellalgo_rs::arr::{linspace, Arr};
use ellalgo_rs::cutting_plane::{OracleOptim, SingleCut};
use ndarray::Array2;

fn basis_funs(u: f64, i: usize, k: usize, t: &Arr, n: &mut [f64]) {
    let mut left = vec![0.0; k + 1];
    let mut right = vec![0.0; k + 1];
    for v in n.iter_mut() {
        *v = 0.0;
    }
    n[0] = 1.0;
    for j in 1..=k {
        left[j] = u - t[i + 1 - j];
        right[j] = t[i + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
}

fn find_span(t: &Arr, k: usize, n: usize, x: f64) -> usize {
    if x >= t[n] {
        return n - 1;
    }
    if x < t[k] {
        return k;
    }
    let mut i = k;
    while i + 1 < n && x >= t[i + 1] {
        i += 1;
    }
    i
}

/// Clamped knot vector: `k + 1` zeros, `m - k - 1` interior knots, `k + 1`
/// copies of `dmax`.
pub fn clamped_knots(dmax: f64, m: usize, k: usize) -> Arr {
    let full = linspace(0.0, dmax, m - k + 1);
    let mut t = Arr::new(m + k + 1);
    let mut pos = 0;
    for _ in 0..=k {
        t[pos] = 0.0;
        pos += 1;
    }
    for i in 1..(full.size() - 1) {
        t[pos] = full[i];
        pos += 1;
    }
    for _ in 0..=k {
        t[pos] = dmax;
        pos += 1;
    }
    t
}

/// Evaluate all `m = t.len() - k - 1` basis functions at every entry of `d`.
pub fn eval_bspline_basis(t: &Arr, k: usize, d: &Arr) -> Vec<Array2<f64>> {
    let n = t.size() - k - 1;
    let nr = d.rows();
    let nc = d.cols();
    let mut out: Vec<Array2<f64>> = (0..n).map(|_| Array2::zeros((nr, nc))).collect();
    let mut basis = vec![0.0; k + 1];
    for r in 0..nr {
        for c in 0..nc {
            let x = d.get(r, c);
            let i = find_span(t, k, n, x);
            basis_funs(x, i, k, t, &mut basis);
            let start = i - k;
            for (j, &bj) in basis.iter().enumerate() {
                out[start + j][[r, c]] = bj;
            }
        }
    }
    out
}

/// Evaluate `sum_i c(i) * B_i` at every point of a 1-D grid `x`.
pub fn eval_bspline_curve(t: &Arr, k: usize, c: &Arr, x: &Arr) -> Arr {
    let n = t.size() - k - 1;
    let mut out = Arr::new(x.size());
    let mut basis = vec![0.0; k + 1];
    for j in 0..x.size() {
        let i = find_span(t, k, n, x[j]);
        basis_funs(x[j], i, k, t, &mut basis);
        let start = i - k;
        let mut s = 0.0;
        for (b, &bj) in basis.iter().enumerate() {
            s += c[start + b] * bj;
        }
        out[j] = s;
    }
    out
}

/// Build the clamped basis matrices and knot vector for a site set.
///
/// # Panics
///
/// Panics if `m < k + 1` (the quadratic basis needs at least 3 control points).
pub fn generate_bspline_info(site: &Arr, m: usize) -> (Vec<Array2<f64>>, Arr, usize) {
    let k = 2;
    assert!(
        m > k,
        "quadratic B-spline needs m >= {} control points, got {}",
        k + 1,
        m
    );
    let d = construct_distance_matrix(site);
    let mut dmax: f64 = 0.0;
    for i in 0..d.size() {
        dmax = dmax.max(d[i]);
    }
    let t = clamped_knots(dmax, m, k);
    let sigma = eval_bspline_basis(&t, k, &d);
    (sigma, t, k)
}

/// Return the first monotonicity violation of `x`, or `None` if non-increasing.
pub fn mono_oracle(x: &Arr) -> Option<(Arr, f64)> {
    let n = x.len();
    let mut g = Arr::new(n);
    for i in 0..n.saturating_sub(1) {
        let fj = x[i + 1] - x[i];
        if fj > 0.0 {
            g[i] = -1.0;
            g[i + 1] = 1.0;
            return Some((g, fj));
        }
    }
    None
}

/// Enforce monotone non-increasing coefficients on the leading `n_coeff`
/// entries, then delegate to the wrapped basis oracle.
pub struct MonoDecreasingOracle2<O> {
    basis: O,
    n_coeff: Option<usize>,
}

impl<O> MonoDecreasingOracle2<O> {
    pub fn new(basis: O, n_coeff: Option<usize>) -> Self {
        MonoDecreasingOracle2 { basis, n_coeff }
    }
}

impl<O: OracleOptim<Arr, CutChoice = SingleCut>> OracleOptim<Arr> for MonoDecreasingOracle2<O> {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, t: &mut f64) -> ((Arr, SingleCut), bool) {
        let n = x.len();
        let k = self.n_coeff.unwrap_or(n.saturating_sub(1));
        let mut xk = Arr::new(k);
        for i in 0..k {
            xk[i] = x[i];
        }
        if let Some((g1, fj)) = mono_oracle(&xk) {
            let mut g = Arr::new(n);
            for i in 0..k {
                g[i] = g1[i];
            }
            return ((g, SingleCut(fj)), false);
        }
        self.basis.assess_optim(x, t)
    }
}
