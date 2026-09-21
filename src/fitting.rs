//! Public fitting drivers for the polynomial and B-spline correlation models.

use crate::bspline::{generate_bspline_info, MonoDecreasingOracle2};
use crate::corr_helper::construct_poly_matrix;
use crate::linalg;
use crate::lmi0_oracle::LMI0Oracle;
use crate::lsq_oracle::LsqOracle;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
use ellalgo_rs::ell::Ell;
use ndarray::Array2;

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

fn ndarray_to_arr(a: &Array2<f64>) -> Arr {
    let n = a.nrows();
    let m = a.ncols();
    let mut out = Arr::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            out.set(i, j, a[[i, j]]);
        }
    }
    out
}

fn ndarray_norm(a: &Array2<f64>) -> f64 {
    let mut s = 0.0;
    for v in a.iter() {
        s += v * v;
    }
    s.sqrt()
}

fn ndarray_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut out = Array2::zeros((m, n));
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let ad = a.as_slice().expect("standard layout is contiguous");
    let bd = b.as_slice().expect("standard layout is contiguous");
    let od = out.as_slice_mut().expect("out is contiguous");
    for i in 0..m {
        let orow = &mut od[i * n..(i + 1) * n];
        for t in 0..k {
            let ait = ad[i * k + t];
            let brow = &bd[t * n..(t + 1) * n];
            for j in 0..n {
                orow[j] += ait * brow[j];
            }
        }
    }
    out
}

fn trace_ndarray(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0;
    for i in 0..n {
        s += a[[i, i]];
    }
    s
}

fn frob_inner_ndarray(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn inv_upper_tri(r: &Array2<f64>) -> Array2<f64> {
    let n = r.nrows();
    let mut x = Array2::zeros((n, n));
    for j in 0..n {
        for i in (0..=j).rev() {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in (i + 1)..=j {
                s -= r[[i, k]] * x[[k, j]];
            }
            x[[i, j]] = s / r[[i, i]];
        }
    }
    x
}

fn lsq_corr_core2<O: OracleOptim<Arr, CutChoice = SingleCut>>(
    norm_y: f64,
    m: usize,
    omega: &mut O,
) -> (Arr, usize) {
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

/// Least-squares fit over an explicit basis `sigma`, optionally constraining the
/// leading `n_coeff` coefficients to be non-increasing.
pub fn lsq_corr_generic(
    y: &Array2<f64>,
    sigma: &[Array2<f64>],
    n_coeff: Option<usize>,
) -> (Arr, usize) {
    let m = sigma.len();
    let norm_y = ndarray_norm(y);
    let mut omega = LsqOracle::new(y.nrows(), sigma.to_vec(), y.clone());
    if let Some(nc) = n_coeff {
        let mut wrapped = MonoDecreasingOracle2::new(omega, Some(nc));
        lsq_corr_core2(norm_y, m, &mut wrapped)
    } else {
        lsq_corr_core2(norm_y, m, &mut omega)
    }
}

/// Least-squares fit of the polynomial basis `D^0..D^(m-1)`.
pub fn lsq_corr_poly(y: &Array2<f64>, site: &Arr, m: usize) -> (Arr, usize) {
    let sigma = construct_poly_matrix(site, m);
    lsq_corr_generic(y, &sigma, None)
}

/// Least-squares fit of the quadratic B-spline basis, coefficients non-increasing.
pub fn lsq_corr_bspline(y: &Array2<f64>, site: &Arr, m: usize) -> (Arr, usize) {
    let (sigma, _t, _k) = generate_bspline_info(site, m);
    lsq_corr_generic(y, &sigma, Some(m))
}

/// Assemble `Omega(x) = sum_i x_i Sigma_i`.
pub fn corr_omega(x: &Arr, sigma: &[Array2<f64>]) -> Array2<f64> {
    let n = sigma[0].nrows();
    let mut om = Array2::zeros((n, n));
    for (i, f) in sigma.iter().enumerate() {
        let xi = x[i];
        for r in 0..n {
            for c in 0..n {
                om[[r, c]] += xi * f[[r, c]];
            }
        }
    }
    om
}

/// MLE objective `log det Omega(x) + Tr(Omega(x)^-1 Y)`.
pub fn corr_mle_obj(x: &Arr, sigma: &[Array2<f64>], y: &Array2<f64>) -> f64 {
    let om = ndarray_to_arr(&corr_omega(x, sigma));
    let l = linalg::cholesky(&om);
    let n = l.rows();
    let mut logdet = 0.0;
    for i in 0..n {
        logdet += 2.0 * l.get(i, i).ln();
    }
    let inv_om = arr_to_ndarray(&linalg::inv(&om));
    logdet + trace_ndarray(&ndarray_matmul(&inv_om, y))
}

/// One CCP round: linearize the concave `-log det` part at `Omega(x)` and
/// minimize the convex surrogate with the cutting-plane method.
pub struct CccpMleOracle {
    y: Array2<f64>,
    sigma: Vec<Array2<f64>>,
    lmi0: LMI0Oracle,
    mk: Arr,
}

impl CccpMleOracle {
    pub fn new(sigma: Vec<Array2<f64>>, y: Array2<f64>, m_mat: &Array2<f64>) -> Self {
        let lmi0 = LMI0Oracle::new(sigma.clone());
        let mut mk = Arr::new(sigma.len());
        for (i, f) in sigma.iter().enumerate() {
            mk[i] = trace_ndarray(&ndarray_matmul(m_mat, f));
        }
        CccpMleOracle { y, sigma, lmi0, mk }
    }
}

impl OracleOptim<Arr> for CccpMleOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, t: &mut f64) -> ((Arr, SingleCut), bool) {
        if let Some((g, fj)) = self.lmi0.assess_feas(x) {
            return ((g, SingleCut(fj)), false);
        }
        let r = self.lmi0.ldlt_mgr.sqrt();
        let inv_r = inv_upper_tri(&r);
        let s = ndarray_matmul(&inv_r, &inv_r.t().to_owned());
        let sy = ndarray_matmul(&s, &self.y);
        let sys = ndarray_matmul(&sy, &s);

        let mut h = trace_ndarray(&sy);
        for i in 0..x.len() {
            h += x[i] * self.mk[i];
        }
        let mut f = h - *t;
        let shrunk = f < 0.0;
        if shrunk {
            *t = h;
            f = 0.0;
        }
        let n = x.len();
        let mut g = Arr::new(n);
        for i in 0..n {
            g[i] = -frob_inner_ndarray(&self.sigma[i], &sys) + self.mk[i];
        }
        ((g, SingleCut(f)), shrunk)
    }
}

/// Run a single CCP round from `x` over the explicit basis `sigma`.
pub fn cccp_corr_step(
    sigma: &[Array2<f64>],
    y: &Array2<f64>,
    x: Arr,
    n_coeff: Option<usize>,
) -> (Arr, usize) {
    let m_inv = arr_to_ndarray(&linalg::inv(&ndarray_to_arr(&corr_omega(&x, sigma))));
    let omega = CccpMleOracle::new(sigma.to_vec(), y.clone(), &m_inv);
    let mut ellip = Ell::new_with_scalar(100.0, x.clone());
    let mut t = 1e100;
    let (x_best, iters) = if let Some(nc) = n_coeff {
        let mut wrapped = MonoDecreasingOracle2::new(omega, Some(nc));
        cutting_plane_optim(&mut wrapped, &mut ellip, &mut t, &Options::default())
    } else {
        let mut omega = omega;
        cutting_plane_optim(&mut omega, &mut ellip, &mut t, &Options::default())
    };
    (x_best.unwrap_or(x), iters)
}

/// Run the CCP loop (up to 50 rounds) until the MLE objective stalls.
pub fn cccp_corr_generic(
    sigma: &[Array2<f64>],
    y: &Array2<f64>,
    x: Arr,
    n_coeff: Option<usize>,
) -> (Arr, usize) {
    let mut x = x;
    let mut f_old = 1e100;
    let mut total_iters = 0;
    for _ in 0..50 {
        let (x_new, iters) = cccp_corr_step(sigma, y, x.clone(), n_coeff);
        total_iters += iters;
        if x_new.size() != x.size() {
            break;
        }
        let f_new = corr_mle_obj(&x_new, sigma, y);
        if (f_old - f_new).abs() < 1e-8 {
            x = x_new;
            break;
        }
        f_old = f_new;
        x = x_new;
    }
    (x, total_iters)
}

/// CCP MLE fit of the polynomial basis, warm-started from `lsq_corr_poly`.
pub fn cccp_corr_poly(y: &Array2<f64>, site: &Arr, m: usize) -> (Arr, usize) {
    let sigma = construct_poly_matrix(site, m);
    let (x_lsq, _) = lsq_corr_poly(y, site, m);
    cccp_corr_generic(&sigma, y, x_lsq, None)
}

/// CCP MLE fit of the B-spline basis, warm-started from `lsq_corr_bspline`.
pub fn cccp_corr_bspline(y: &Array2<f64>, site: &Arr, m: usize) -> (Arr, usize) {
    let (x0, _) = lsq_corr_bspline(y, site, m);
    let (sigma, _t, _k) = generate_bspline_info(site, m);
    cccp_corr_generic(&sigma, y, x0, Some(m))
}
