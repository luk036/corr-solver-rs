//! Scratch and cut helpers shared by the MLE and CCP oracles.

use crate::lmi0_oracle::LMI0Oracle;
use crate::ndops;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::SingleCut;
use ndarray::Array2;

/// Cholesky-based scratch shared by the MLE and CCP oracles: the factor `R` with
/// `Omega(x) = R^T R`, `S = R^-T R^-1`, and `SY = S Y`.
pub struct MleScratch {
    /// Factor `R` with `Omega(x) = R^T R`.
    pub r: Array2<f64>,
    /// `S = R^-T R^-1`.
    pub s: Array2<f64>,
    /// `S Y`.
    pub sy: Array2<f64>,
}

impl MleScratch {
    pub fn new() -> Self {
        MleScratch {
            r: Array2::zeros((0, 0)),
            s: Array2::zeros((0, 0)),
            sy: Array2::zeros((0, 0)),
        }
    }

    /// Refresh `R`, `S` and `SY` from the current LDLT factor of `lmi0`.
    pub fn update(&mut self, lmi0: &mut LMI0Oracle, y: &Array2<f64>) {
        let r = lmi0.ldlt_mgr.sqrt();
        let inv_r = ndops::inv_upper_tri(&r);
        self.s = ndops::matmul(&inv_r, &inv_r.t().to_owned());
        self.sy = ndops::matmul(&self.s, y);
        self.r = r;
    }
}

impl Default for MleScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Fold the best-so-far value into a cut, reporting whether `t` improved.
#[inline]
pub fn optim_cut(g: Arr, value: f64, t: &mut f64) -> ((Arr, SingleCut), bool) {
    let mut f = value - *t;
    let shrunk = f < 0.0;
    if shrunk {
        *t = value;
        f = 0.0;
    }
    ((g, SingleCut(f)), shrunk)
}
