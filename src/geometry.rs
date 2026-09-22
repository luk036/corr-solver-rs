//! Pairwise distance geometry for site layouts.

use ellalgo_rs::arr::Arr;

/// Euclidean distance matrix: `D[i, j] = ||site(i) - site(j)||`.
pub fn construct_distance_matrix(site: &Arr) -> Arr {
    let n = site.rows();
    let mut d1 = Arr::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0;
            for k in 0..site.cols() {
                let diff = site.get(j, k) - site.get(i, k);
                d += diff * diff;
            }
            let val = d.sqrt();
            d1.set(i, j, val);
            d1.set(j, i, val);
        }
    }
    d1
}
