//! Conversions between [`ellalgo_rs::arr::Arr`] and [`ndarray::Array2`].

use ellalgo_rs::arr::Arr;
use ndarray::Array2;

/// Copy an `Arr` matrix into a row-major `Array2`.
pub fn arr_to_ndarray(a: &Arr) -> Array2<f64> {
    let (n, m) = (a.rows(), a.cols());
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a.get(i, j);
        }
    }
    out
}

/// Copy an `Array2` into a row-major `Arr` matrix.
pub fn ndarray_to_arr(a: &Array2<f64>) -> Arr {
    let (n, m) = (a.nrows(), a.ncols());
    let mut out = Arr::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            out.set(i, j, a[[i, j]]);
        }
    }
    out
}
