//! Public fitting drivers for the polynomial and B-spline correlation models.

use crate::bspline::{generate_bspline_info, MonoDecreasingOracle2};
use crate::convert::{arr_to_ndarray, ndarray_to_arr};
use crate::corr_helper::construct_poly_matrix;
use crate::layouts::{cccp_initial_guess, lsq_initial_guess, mle_initial_guess, INITIAL_T};
use crate::linalg;
use crate::lmi0_oracle::LMI0Oracle;
use crate::lsq_oracle::LsqOracle;
use crate::mle_common::{optim_cut, MleScratch};
use crate::mle_oracle::MleOracle;
use crate::ndops;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options, OracleOptim, SingleCut};
use ndarray::Array2;

/// Outcome of a correlation fit. `ok` is false when the cutting-plane search
/// failed, in which case `coeffs` is empty.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub coeffs: Arr,
    pub iters: usize,
    pub ok: bool,
}

/// Evaluate the polynomial with ascending coefficients `c` at every point of `x`.
pub fn eval_poly_curve(c: &Arr, x: &Arr) -> Arr {
    let mut out = Arr::new(x.size());
    for (j, &xj) in x.iter().enumerate() {
        let mut v = 0.0;
        for &ci in c.iter().rev() {
            v = v * xj + ci;
        }
        out[j] = v;
    }
    out
}

/// Run `body` against the raw oracle, or against the monotone-decorated one.
///
/// The `body` must refer to the oracle through the `omega` binding.
macro_rules! with_mono {
    ($n_coeff:expr, $omega:expr, |$o:ident| $body:expr) => {
        match $n_coeff {
            Some(nc) => {
                let mut $o = MonoDecreasingOracle2::new($omega, Some(nc));
                $body
            }
            None => {
                let mut $o = $omega;
                $body
            }
        }
    };
}

fn lsq_corr_core2<O: OracleOptim<Arr, CutChoice = SingleCut>>(
    norm_y: f64,
    m: usize,
    omega: &mut O,
) -> FitResult {
    let mut ellip = lsq_initial_guess(norm_y, m);
    let mut t = INITIAL_T;
    let (x_best, iters) = cutting_plane_optim(omega, &mut ellip, &mut t, &Options::default());
    match x_best {
        Some(xb) => {
            let mut coeffs = Arr::new(m);
            for i in 0..m {
                coeffs[i] = xb[i];
            }
            FitResult {
                coeffs,
                iters,
                ok: true,
            }
        }
        None => FitResult {
            coeffs: Arr::new(0),
            iters,
            ok: false,
        },
    }
}

/// Least-squares fit over an explicit basis `sigma`, optionally constraining the
/// leading `n_coeff` coefficients to be non-increasing.
pub fn lsq_corr_generic(
    y: &Array2<f64>,
    sigma: &[Array2<f64>],
    n_coeff: Option<usize>,
) -> FitResult {
    let m = sigma.len();
    let norm_y = ndops::norm(y);
    let omega = LsqOracle::new(y.nrows(), sigma.to_vec(), y.clone());
    with_mono!(n_coeff, omega, |o| lsq_corr_core2(norm_y, m, &mut o))
}

/// Least-squares fit of the polynomial basis `D^0..D^(m-1)`.
pub fn lsq_corr_poly(y: &Array2<f64>, site: &Arr, m: usize) -> FitResult {
    let sigma = construct_poly_matrix(site, m);
    lsq_corr_generic(y, &sigma, None)
}

/// Least-squares fit of the quadratic B-spline basis, coefficients non-increasing.
pub fn lsq_corr_bspline(y: &Array2<f64>, site: &Arr, m: usize) -> FitResult {
    let (sigma, _t, _k) = generate_bspline_info(site, m);
    lsq_corr_generic(y, &sigma, Some(m))
}

fn mle_corr_core(m: usize, omega: &mut MleOracle) -> FitResult {
    let mut ellip = mle_initial_guess(m);
    let mut t = INITIAL_T;
    let (x_best, iters) = cutting_plane_optim(omega, &mut ellip, &mut t, &Options::default());
    match x_best {
        Some(xb) => FitResult {
            coeffs: xb,
            iters,
            ok: true,
        },
        None => FitResult {
            coeffs: Arr::new(0),
            iters,
            ok: false,
        },
    }
}

/// Maximum-likelihood fit over an explicit basis.
pub fn mle_corr_generic(y: &Array2<f64>, sigma: &[Array2<f64>]) -> FitResult {
    let m = sigma.len();
    let mut omega = MleOracle::new(sigma.to_vec(), y.clone());
    mle_corr_core(m, &mut omega)
}

/// Maximum-likelihood fit of the polynomial basis subject to `2Y >= Omega >= 0`.
pub fn mle_corr_poly(y: &Array2<f64>, site: &Arr, m: usize) -> FitResult {
    let sigma = construct_poly_matrix(site, m);
    mle_corr_generic(y, &sigma)
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
    logdet + ndops::trace(&ndops::matmul(&inv_om, y))
}

/// One CCP round: linearize the concave `-log det` part at `Omega(x)` and
/// minimize the convex surrogate with the cutting-plane method.
pub struct CccpMleOracle {
    y: Array2<f64>,
    sigma: Vec<Array2<f64>>,
    lmi0: LMI0Oracle,
    mk: Arr,
    scratch: MleScratch,
}

impl CccpMleOracle {
    pub fn new(sigma: Vec<Array2<f64>>, y: Array2<f64>, m_mat: &Array2<f64>) -> Self {
        let lmi0 = LMI0Oracle::new(sigma.clone());
        let mut mk = Arr::new(sigma.len());
        for (i, f) in sigma.iter().enumerate() {
            mk[i] = ndops::trace(&ndops::matmul(m_mat, f));
        }
        CccpMleOracle {
            y,
            sigma,
            lmi0,
            mk,
            scratch: MleScratch::new(),
        }
    }
}

impl OracleOptim<Arr> for CccpMleOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, t: &mut f64) -> ((Arr, SingleCut), bool) {
        if let Some((g, fj)) = self.lmi0.assess_feas(x) {
            return ((g, SingleCut(fj)), false);
        }
        self.scratch.update(&mut self.lmi0, &self.y);
        let sys = ndops::matmul(&self.scratch.sy, &self.scratch.s);

        let mut h = ndops::trace(&self.scratch.sy);
        for i in 0..x.len() {
            h += x[i] * self.mk[i];
        }

        let n = x.len();
        let mut g = Arr::new(n);
        for i in 0..n {
            g[i] = -ndops::frob_inner(&self.sigma[i], &sys) + self.mk[i];
        }
        optim_cut(g, h, t)
    }
}

/// Run a single CCP round from `x` over the explicit basis `sigma`.
pub fn cccp_corr_step(
    sigma: &[Array2<f64>],
    y: &Array2<f64>,
    x: Arr,
    n_coeff: Option<usize>,
) -> FitResult {
    let m_inv = arr_to_ndarray(&linalg::inv(&ndarray_to_arr(&corr_omega(&x, sigma))));
    let omega = CccpMleOracle::new(sigma.to_vec(), y.clone(), &m_inv);
    let size = x.size();
    with_mono!(n_coeff, omega, |o| {
        let mut ellip = cccp_initial_guess(&x);
        let mut t = INITIAL_T;
        let (x_best, iters) = cutting_plane_optim(&mut o, &mut ellip, &mut t, &Options::default());
        match x_best {
            Some(xb) if xb.size() == size => FitResult {
                coeffs: xb,
                iters,
                ok: true,
            },
            _ => FitResult {
                coeffs: Arr::new(0),
                iters,
                ok: false,
            },
        }
    })
}

/// Run the CCP loop (up to 50 rounds) until the MLE objective stalls.
pub fn cccp_corr_generic(
    sigma: &[Array2<f64>],
    y: &Array2<f64>,
    x: Arr,
    n_coeff: Option<usize>,
) -> FitResult {
    let mut x = x;
    let mut f_old = 1e100;
    let mut total_iters = 0;
    for _ in 0..50 {
        let step = cccp_corr_step(sigma, y, x.clone(), n_coeff);
        total_iters += step.iters;
        if !step.ok {
            break;
        }
        let f_new = corr_mle_obj(&step.coeffs, sigma, y);
        if (f_old - f_new).abs() < 1e-8 {
            x = step.coeffs;
            break;
        }
        f_old = f_new;
        x = step.coeffs;
    }
    FitResult {
        coeffs: x,
        iters: total_iters,
        ok: true,
    }
}

/// CCP MLE fit of the polynomial basis, warm-started from [`lsq_corr_poly`].
pub fn cccp_corr_poly(y: &Array2<f64>, site: &Arr, m: usize) -> FitResult {
    let sigma = construct_poly_matrix(site, m);
    let lsq = lsq_corr_poly(y, site, m);
    if !lsq.ok {
        return FitResult {
            coeffs: Arr::new(0),
            iters: 0,
            ok: false,
        };
    }
    cccp_corr_generic(&sigma, y, lsq.coeffs, None)
}

/// CCP MLE fit of the B-spline basis, warm-started from [`lsq_corr_bspline`].
pub fn cccp_corr_bspline(y: &Array2<f64>, site: &Arr, m: usize) -> FitResult {
    let lsq = lsq_corr_bspline(y, site, m);
    if !lsq.ok {
        return FitResult {
            coeffs: Arr::new(0),
            iters: 0,
            ok: false,
        };
    }
    let (sigma, _t, _k) = generate_bspline_info(site, m);
    cccp_corr_generic(&sigma, y, lsq.coeffs, Some(m))
}
