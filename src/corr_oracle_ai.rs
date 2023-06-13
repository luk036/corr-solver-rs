use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::Cholesky;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::Distribution;
use std::f64::consts::PI;

type Arr = Array2<f64>;
type Cut = (Arr, f64);

fn arr2(a: &[f64]) -> Arr {
    Array::from_shape_vec((a.len(), 1), a.to_vec()).unwrap()
}

fn create_2d_sites(nx: usize, ny: usize) -> Arr {
    let n = nx * ny;
    let s_end = arr2(&[10.0, 8.0]);
    let mut hgen = Halton::new(&[2, 3]);
    let s = s_end * arr2(&hgen.take(n).collect::<Vec<f64>>());
    s
}

fn create_2d_isotropic(s: &Arr, N: usize) -> Arr {
    let n = s.shape()[0];
    let sdkern = 0.12;
    let var = 2.0;
    let tau = 0.00001;
    let mut rng = StdRng::seed_from_u64(5);
    let mut Sig = Array::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let d = s.slice(s![j, ..]) - s.slice(s![i, ..]);
            Sig[[i, j]] = (-sdkern * d.dot(&d)).exp();
            Sig[[j, i]] = Sig[[i, j]];
        }
    }
    let A = Sig.cholesky().unwrap();
    let mut Y = Array::zeros((n, n));
    for _ in 0..N {
        let x = Array::random_using(n, StandardNormal, &mut rng) * var;
        let y = A.solve_triangular(&x, false).unwrap() + Array::random_using(n, StandardNormal, &mut rng) * tau;
        Y += y.outer(&y);
    }
    Y /= N as f64;
    Y
}

fn construct_poly_matrix(s: &Arr, m: usize) -> Vec<Arr> {
    let n = s.shape()[0];
    let D1 = construct_distance_matrix(s);
    let mut D = Array::ones((n, n));
    let mut Sig = vec![D.clone()];
    for _ in 1..m {
        D *= &D1;
        Sig.push(D.clone());
    }
    Sig
}

fn corr_poly(Y: &Arr, s: &Arr, m: usize, oracle: fn(&[Arr], &Arr) -> Arr, corr_core: fn(&Arr, usize, &Arr) -> Cut) -> (Arr, usize, bool) {
    let Sig = construct_poly_matrix(s, m);
    let omega = oracle(&Sig, Y);
    let (a, num_iters, feasible) = corr_core(Y, m, &omega);
    let pa = a.iter().rev().copied().collect::<Vec<f64>>();
    let b = a.mapv(|x| x.powi(2));
    (poly1d(&pa), num_iters, feasible)
}

fn construct_distance_matrix(s: &Arr) -> Arr {
    let n = s.shape()[0];
    let mut D = Array::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let d = s.slice(s![j, ..]) - s.slice(s![i, ..]);
            D[[i, j]] = d.dot(&d).sqrt();
            D[[j, i]] = D[[i, j]];
        }
    }
    let max_d = D.max().unwrap();
    D.mapv(|x| x / max_d)
}

fn poly1d(a: &[f64]) -> String {
    let mut res = String::new();
    for (i, &x) in a.iter().enumerate() {
        if i == 0 {
            res.push_str(&format!("{:.6}", x));
        } else {
            res.push_str(&format!(" + {:.6}x^{}", x, i));
        }
    }
    res
}

