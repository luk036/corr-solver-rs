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

/// Creates a 2D array of site locations using the Halton sequence.
///
/// This function generates a 2D array of site locations using the Halton sequence, a quasi-random number generation technique.
/// The number of sites is determined by the `nx` and `ny` parameters, which specify the number of sites in the x and y dimensions, respectively.
/// The sites are scaled to the range `[0, 10] x [0, 8]`.
///
/// # Arguments
/// * `nx` - The number of sites in the x dimension.
/// * `ny` - The number of sites in the y dimension.
///
/// # Returns
/// A 2D array of site locations.
fn create_2d_sites(nx: usize, ny: usize) -> Arr {
    let n = nx * ny;
    let s_end = arr2(&[10.0, 8.0]);
    let mut hgen = Halton::new(&[2, 3]);
    let s = s_end * arr2(&hgen.take(n).collect::<Vec<f64>>());
    s
}

/// Creates a 2D array of isotropic covariance matrices based on the given site locations.
///
/// This function generates a 2D array of isotropic covariance matrices using the given site locations `s`. The covariance matrices are generated using an exponential kernel with a specified standard deviation `sdkern`, variance `var`, and noise parameter `tau`. The covariance matrices are then averaged over `N` samples.
///
/// # Arguments
/// * `s` - A 2D array of site locations.
/// * `N` - The number of samples to average the covariance matrices over.
///
/// # Returns
/// A 2D array of isotropic covariance matrices.
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

/// Constructs a matrix of polynomial terms based on the given site locations and the specified degree.
///
/// This function takes the site locations `s` and the desired degree `m`, and constructs a matrix of polynomial terms. The matrix is constructed by first computing the distance matrix `D1` using the `construct_distance_matrix` function. Then, the function iteratively computes powers of `D1` up to the specified degree `m`, and collects these matrices into a vector `Sig`.
///
/// # Arguments
/// * `s` - A 2D array of site locations.
/// * `m` - The desired degree of the polynomial terms.
///
/// # Returns
/// A vector of 2D arrays, where each array represents the polynomial terms of a specific degree.
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

/// Computes the correlation polynomial for a given input array `Y` and site locations `s`, using the specified degree `m`, an oracle function `oracle`, and a correlation core function `corr_core`.
///
/// # Arguments
/// * `Y` - The input array for which to compute the correlation polynomial.
/// * `s` - The site locations.
/// * `m` - The desired degree of the polynomial terms.
/// * `oracle` - A function that takes a vector of polynomial matrices and the input array `Y`, and returns an array `omega`.
/// * `corr_core` - A function that takes the input array `Y`, the degree `m`, and the array `omega`, and returns the coefficients `a`, the number of iterations `num_iters`, and a boolean `feasible` indicating whether the computation was feasible.
///
/// # Returns
/// A tuple containing:
/// * The correlation polynomial as a string.
/// * The number of iterations required.
/// * A boolean indicating whether the computation was feasible.
fn corr_poly(Y: &Arr, s: &Arr, m: usize, oracle: fn(&[Arr], &Arr) -> Arr, corr_core: fn(&Arr, usize, &Arr) -> Cut) -> (Arr, usize, bool) {
    let Sig = construct_poly_matrix(s, m);
    let omega = oracle(&Sig, Y);
    let (a, num_iters, feasible) = corr_core(Y, m, &omega);
    let pa = a.iter().rev().copied().collect::<Vec<f64>>();
    let b = a.mapv(|x| x.powi(2));
    (poly1d(&pa), num_iters, feasible)
}

/// Constructs a distance matrix from the given site locations.
///
/// This function takes the site locations `s` and constructs a distance matrix `D` where `D[i, j]` represents the Euclidean distance between the `i`-th and `j`-th site locations. The distance matrix is normalized by the maximum distance.
///
/// # Arguments
/// * `s` - A 2D array of site locations.
///
/// # Returns
/// A 2D array representing the normalized distance matrix.
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

/// Constructs a string representation of a 1D polynomial from the given array of coefficients.
///
/// # Arguments
/// * `a` - An array of coefficients for the polynomial.
///
/// # Returns
/// A string representation of the 1D polynomial.
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

