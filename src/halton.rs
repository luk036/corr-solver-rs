//! Self-contained Halton low-discrepancy site generator.
//!
//! Mirrors `create_2d_sites` from the Python package, which uses the `lds-gen`
//! Halton generator with bases 2 and 3 scaled by the `(10.0, 8.0)` extent.

use ellalgo_rs::arr::Arr;

/// Radical inverse of `i` in `base`: the base-`base` digits of `i`, reflected
/// about the radix point.
#[inline]
pub fn radical_inverse(i: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    let mut n = i;
    while n > 0 {
        result += f * (n % base) as f64;
        n /= base;
        f /= base as f64;
    }
    result
}

/// `nx * ny` Halton sites scaled to `[0, 10] x [0, 8]`, returned as a
/// `(nx * ny) x 2` row-major matrix.
pub fn create_2d_sites_halton(nx: usize, ny: usize) -> Arr {
    let num = nx * ny;
    let mut site = Arr::zeros(num, 2);
    for k in 1..=num {
        site.set(k - 1, 0, radical_inverse(k as u64, 2) * 10.0);
        site.set(k - 1, 1, radical_inverse(k as u64, 3) * 8.0);
    }
    site
}
