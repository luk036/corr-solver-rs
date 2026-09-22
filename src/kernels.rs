//! Radial covariance kernels shared by the generators and experiments.

/// Gaussian kernel `exp(-rate * r^2)`, with `r` the distance.
#[inline]
pub fn gaussian_kernel(r: f64, rate: f64) -> f64 {
    (-rate * r * r).exp()
}

/// Matern 1/2 (exponential) kernel `exp(-rate * r)`, with `r` the distance.
#[inline]
pub fn exponential_kernel(r: f64, rate: f64) -> f64 {
    (-rate * r).exp()
}
