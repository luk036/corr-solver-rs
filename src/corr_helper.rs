use crate::linalg;
use ellalgo_rs::arr::{linspace, Arr};
use ndarray::Array2;

pub fn create_2d_sites(nx: usize, ny: usize) -> Arr {
    let sx = linspace(0.0, 10.0, nx);
    let sy = linspace(0.0, 8.0, ny);
    let (xx, yy) = linalg::meshgrid(&sx, &sy);
    let fx = linalg::flatten(&xx);
    let fy = linalg::flatten(&yy);
    let st = linalg::stack(&fx, &fy);
    linalg::transpose(&st)
}

pub fn create_2d_isotropic(site: &Arr, n: usize) -> Arr {
    let n_sites = site.rows();
    let sdkern = 0.3;
    let var = 2.0;
    let tau = 0.00001;
    linalg::random_seed(5);

    let mut sig = Arr::zeros(n_sites, n_sites);
    for i in 0..n_sites {
        for j in i..n_sites {
            let mut d = 0.0;
            for k in 0..site.cols() {
                let diff = site.get(j, k) - site.get(i, k);
                d += diff * diff;
            }
            let g = -sdkern * d.sqrt();
            let val = g.exp();
            sig.set(i, j, val);
            sig.set(j, i, val);
        }
    }

    let a = linalg::cholesky(&sig);
    let mut y = Arr::zeros(n_sites, n_sites);
    for _ in 0..n {
        let x = var * linalg::randn(n_sites);
        let mut y_tmp = Arr::new(n_sites);
        for i in 0..n_sites {
            let mut s = 0.0;
            for j in 0..n_sites {
                s += a.get(i, j) * x[j];
            }
            y_tmp[i] = s;
        }
        let noise = linalg::randn(n_sites);
        for i in 0..n_sites {
            y_tmp[i] += tau * noise[i];
        }
        for i in 0..n_sites {
            for j in 0..n_sites {
                let v = y.get(i, j) + y_tmp[i] * y_tmp[j];
                y.set(i, j, v);
            }
        }
    }
    let nf = n as f64;
    for i in 0..n_sites {
        for j in 0..n_sites {
            y.set(i, j, y.get(i, j) / nf);
        }
    }
    y
}

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

pub fn construct_poly_matrix(site: &Arr, m: usize) -> Vec<Array2<f64>> {
    let n = site.rows();
    let d1 = construct_distance_matrix(site);
    let mut d = Array2::ones((n, n));
    let mut sig: Vec<Array2<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        if i > 0 {
            for r in 0..n {
                for c in 0..n {
                    d[[r, c]] *= d1.get(r, c);
                }
            }
        }
        let mut arr = Array2::zeros((n, n));
        for r in 0..n {
            for c in 0..n {
                arr[[r, c]] = d[[r, c]];
            }
        }
        sig.push(arr);
    }
    sig
}
