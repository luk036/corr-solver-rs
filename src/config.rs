//! Numeric configuration for the cutting-plane solver cores.

use ellalgo_rs::cutting_plane::Options;

/// Numeric constants for the cutting-plane / bisection cores.
///
/// The initial radii are kept close to the coefficient scale: an ellipsoid much
/// larger than the solution wastes iterations (the method needs
/// `O(n^2 log(R/r))`), while one that is too small makes the subproblem
/// infeasible. The MLE radius is shared by the CCP outer loop, which starts from
/// the least-squares solution rather than the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Initial MLE / CCP ellipsoid scale.
    pub mle_r0: f64,
    /// Initial ellipsoid scale for the augmented LSQ core.
    pub lsq_aug_r0: f64,
    /// Multiplier on `||Y||_F^2` for the augmented bound.
    pub lsq_frob_scale: f64,
    /// Convergence tolerance forwarded to ellalgo.
    pub tolerance: f64,
    /// Iteration cap forwarded to ellalgo.
    pub max_iters: usize,
    /// Maximum CCP outer linearization rounds.
    pub cccp_max_rounds: usize,
    /// CCP objective-change tolerance for early stopping.
    pub cccp_tol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            mle_r0: 4.0,
            lsq_aug_r0: 16.0,
            lsq_frob_scale: 1.0,
            tolerance: 1e-12,
            max_iters: 2000,
            cccp_max_rounds: 50,
            cccp_tol: 1e-8,
        }
    }
}

impl SolverConfig {
    /// Build the ellalgo `Options` for the cutting-plane drivers.
    pub fn options(&self) -> Options {
        Options::new(self.max_iters, self.tolerance)
    }
}
