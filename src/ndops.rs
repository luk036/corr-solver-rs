//! Matrix helpers on [`ndarray::Array2`], used by the oracle implementations.

use ndarray::Array2;

/// `out = a * b` for row-major `Array2` operands.
pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut out = Array2::zeros((m, n));
    // `as_standard_layout` is a no-op for C-order input and only copies when the
    // strides require it — `invR.t().to_owned()` yields an F-order array.
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let ad = a.as_slice().expect("standard layout is contiguous");
    let bd = b.as_slice().expect("standard layout is contiguous");
    let od = out.as_slice_mut().expect("out is contiguous");
    // i-k-j order: `b`'s row `t` and `out`'s row `i` are both contiguous, so the
    // inner loop auto-vectorizes and `b` streams through cache. The i-j-t order
    // this replaces walked `b` one column at a time (stride n).
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

/// Trace of a square `Array2`.
pub fn trace(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0;
    for i in 0..n {
        s += a[[i, i]];
    }
    s
}

/// Frobenius inner product.
pub fn frob_inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Frobenius norm.
pub fn norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Inverse of an upper-triangular matrix by back substitution.
pub fn inv_upper_tri(r: &Array2<f64>) -> Array2<f64> {
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
