use ellalgo_rs::arr::Arr;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

pub fn random_seed(seed: u64) {
    RNG.with(|rng| {
        *rng.borrow_mut() = StdRng::seed_from_u64(seed);
    });
}

pub fn randn(n: usize) -> Arr {
    let normal = Normal::new(0.0, 1.0).unwrap();
    RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let mut out = Arr::new(n);
        for i in 0..n {
            out[i] = normal.sample(&mut *rng);
        }
        out
    })
}

pub fn transpose(a: &Arr) -> Arr {
    assert!(a.is_2d());
    let m = a.rows();
    let n = a.cols();
    let mut out = Arr::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            out.set(j, i, a.get(i, j));
        }
    }
    out
}

pub fn matmul(a: &Arr, b: &Arr) -> Arr {
    assert!(a.is_2d() && b.is_2d() && a.cols() == b.rows());
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    let mut out = Arr::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a.get(i, t) * b.get(t, j);
            }
            out.set(i, j, s);
        }
    }
    out
}

pub fn cholesky(a: &Arr) -> Arr {
    let n = a.rows();
    assert!(a.is_2d() && a.cols() == n);
    let mut l = Arr::zeros(n, n);
    for j in 0..n {
        let mut s = 0.0;
        for k in 0..j {
            s += l.get(j, k) * l.get(j, k);
        }
        let val = a.get(j, j) - s;
        l.set(j, j, val.sqrt());
        for i in (j + 1)..n {
            s = 0.0;
            for k in 0..j {
                s += l.get(i, k) * l.get(j, k);
            }
            l.set(i, j, (a.get(i, j) - s) / l.get(j, j));
        }
    }
    l
}

pub fn inv(a: &Arr) -> Arr {
    let n = a.rows();
    let l = cholesky(a);
    let mut y = Arr::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l.get(i, k) * y.get(k, j);
            }
            y.set(i, j, s / l.get(i, i));
        }
    }
    let mut x = Arr::zeros(n, n);
    for j in 0..n {
        for i in (0..n).rev() {
            let mut s = y.get(i, j);
            for k in (i + 1)..n {
                s -= l.get(k, i) * x.get(k, j);
            }
            x.set(i, j, s / l.get(i, i));
        }
    }
    x
}

pub fn trace(a: &Arr) -> f64 {
    let d = diagonal(a);
    d.sum()
}

pub fn norm(a: &Arr) -> f64 {
    let mut s = 0.0;
    for i in 0..a.size() {
        let v = a[i];
        s += v * v;
    }
    s.sqrt()
}

pub fn diagonal(a: &Arr) -> Arr {
    assert!(a.is_2d() && a.rows() == a.cols());
    let n = a.rows();
    let mut out = Arr::new(n);
    for i in 0..n {
        out[i] = a.get(i, i);
    }
    out
}

pub fn flatten(a: &Arr) -> Arr {
    let mut out = Arr::new(a.size());
    for i in 0..a.size() {
        out[i] = a[i];
    }
    out
}

pub fn stack(a: &Arr, b: &Arr) -> Arr {
    assert!(!a.is_2d() && !b.is_2d() && a.size() == b.size());
    let n = a.size();
    let mut out = Arr::zeros(2, n);
    for j in 0..n {
        out.set(0, j, a[j]);
        out.set(1, j, b[j]);
    }
    out
}

pub fn meshgrid(x: &Arr, y: &Arr) -> (Arr, Arr) {
    let nx = x.size();
    let ny = y.size();
    let mut xx = Arr::zeros(ny, nx);
    let mut yy = Arr::zeros(ny, nx);
    for i in 0..ny {
        for j in 0..nx {
            xx.set(i, j, x[j]);
            yy.set(i, j, y[i]);
        }
    }
    (xx, yy)
}
