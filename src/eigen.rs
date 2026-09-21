//! Symmetric Jacobi eigenvalue solver and design-matrix conditioning helper.

use ellalgo_rs::arr::Arr;
use ndarray::Array2;

/// Eigenvalues of a symmetric matrix, sorted ascending (cyclic Jacobi sweeps).
pub fn jacobi_eigvals(a: &Arr) -> Vec<f64> {
    let n = a.rows();
    let mut m = a.clone();
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m.get(p, q) * m.get(p, q);
            }
        }
        if off <= 0.0 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq == 0.0 {
                    continue;
                }
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let sgn = if theta >= 0.0 { 1.0 } else { -1.0 };
                let tt = sgn / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (tt * tt + 1.0).sqrt();
                let s = tt * c;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = m.get(r, p);
                        let arq = m.get(r, q);
                        m.set(r, p, c * arp - s * arq);
                        let np = m.get(r, p);
                        m.set(p, r, np);
                        m.set(r, q, s * arp + c * arq);
                        let nq = m.get(r, q);
                        m.set(q, r, nq);
                    }
                }
                m.set(p, p, c * c * app - 2.0 * s * c * apq + s * s * aqq);
                m.set(q, q, s * s * app + 2.0 * s * c * apq + c * c * aqq);
                m.set(p, q, 0.0);
                m.set(q, p, 0.0);
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| m.get(i, i)).collect();
    ev.sort_by(f64::total_cmp);
    ev
}

/// Smallest eigenvalue of a symmetric matrix.
#[inline]
pub fn min_eig(a: &Arr) -> f64 {
    jacobi_eigvals(a)[0]
}

#[derive(Clone, Copy, Default)]
struct Dd {
    hi: f64,
    lo: f64,
}

#[inline]
fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd {
        hi: s,
        lo: b - (s - a),
    }
}

#[inline]
fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd {
        hi: s,
        lo: (a - (s - bb)) + (b - bb),
    }
}

#[inline]
fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd {
        hi: p,
        lo: a.mul_add(b, -p),
    }
}

#[inline]
fn dd_add(a: Dd, b: Dd) -> Dd {
    let mut s = two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    quick_two_sum(s.hi, s.lo)
}

#[inline]
fn dd_neg(a: Dd) -> Dd {
    Dd {
        hi: -a.hi,
        lo: -a.lo,
    }
}

#[inline]
fn dd_sub(a: Dd, b: Dd) -> Dd {
    dd_add(a, dd_neg(b))
}

#[inline]
fn dd_mul(a: Dd, b: Dd) -> Dd {
    let mut p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p.hi, p.lo)
}

#[inline]
fn dd_div(a: Dd, b: Dd) -> Dd {
    let q1 = a.hi / b.hi;
    let r = dd_sub(a, dd_mul(b, Dd { hi: q1, lo: 0.0 }));
    let q2 = r.hi / b.hi;
    let r = dd_sub(r, dd_mul(b, Dd { hi: q2, lo: 0.0 }));
    let q3 = r.hi / b.hi;
    let mut q = two_sum(q1, q2);
    q.lo += q3;
    quick_two_sum(q.hi, q.lo)
}

#[inline]
fn dd_sqrt(a: Dd) -> Dd {
    if a.hi <= 0.0 {
        return Dd { hi: 0.0, lo: 0.0 };
    }
    let x = a.hi.sqrt();
    let diff = dd_sub(a, two_prod(x, x));
    quick_two_sum(x, diff.hi / (2.0 * x))
}

/// 2-norm condition number of the design matrix whose column `k` is
/// `vec(Sigma[k])`. The `m x m` Gram matrix and its Jacobi eigen-decomposition
/// are carried in double-double arithmetic because squaring the condition
/// number pushes the smallest eigenvalue below double precision.
pub fn design_cond(sigma: &[Array2<f64>]) -> f64 {
    let m = sigma.len();
    let mut g = vec![Dd::default(); m * m];
    let mut dmax: f64 = 0.0;
    for p in 0..m {
        for q in p..m {
            let mut s = Dd { hi: 0.0, lo: 0.0 };
            for (&sp, &sq) in sigma[p].iter().zip(sigma[q].iter()) {
                s = dd_add(s, two_prod(sp, sq));
            }
            g[p * m + q] = s;
            g[q * m + p] = s;
            if p == q {
                dmax = dmax.max(s.hi.abs());
            }
        }
    }
    let thresh = 1e-31 * dmax;
    for _ in 0..80 {
        let mut changed = false;
        for p in 0..m {
            for q in (p + 1)..m {
                let apq = g[p * m + q];
                if apq.hi.abs() <= thresh {
                    continue;
                }
                changed = true;
                let app = g[p * m + p];
                let aqq = g[q * m + q];
                let theta = dd_div(
                    dd_sub(aqq, app),
                    Dd {
                        hi: 2.0 * apq.hi,
                        lo: 2.0 * apq.lo,
                    },
                );
                let abtheta = if theta.hi >= 0.0 {
                    theta
                } else {
                    dd_neg(theta)
                };
                let denom = dd_add(
                    abtheta,
                    dd_sqrt(dd_add(Dd { hi: 1.0, lo: 0.0 }, dd_mul(theta, theta))),
                );
                let tt = dd_div(
                    Dd {
                        hi: if theta.hi >= 0.0 { 1.0 } else { -1.0 },
                        lo: 0.0,
                    },
                    denom,
                );
                let cc = dd_div(
                    Dd { hi: 1.0, lo: 0.0 },
                    dd_sqrt(dd_add(Dd { hi: 1.0, lo: 0.0 }, dd_mul(tt, tt))),
                );
                let ss = dd_mul(tt, cc);
                for r in 0..m {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = g[r * m + p];
                    let arq = g[r * m + q];
                    let np = dd_sub(dd_mul(cc, arp), dd_mul(ss, arq));
                    let nq = dd_add(dd_mul(ss, arp), dd_mul(cc, arq));
                    g[r * m + p] = np;
                    g[p * m + r] = np;
                    g[r * m + q] = nq;
                    g[q * m + r] = nq;
                }
                let cc2 = dd_mul(cc, cc);
                let ss2 = dd_mul(ss, ss);
                let cs2 = dd_mul(Dd { hi: 2.0, lo: 0.0 }, dd_mul(ss, cc));
                g[p * m + p] = dd_add(dd_sub(dd_mul(cc2, app), dd_mul(cs2, apq)), dd_mul(ss2, aqq));
                g[q * m + q] = dd_add(dd_add(dd_mul(ss2, app), dd_mul(cs2, apq)), dd_mul(cc2, aqq));
                g[p * m + q] = Dd { hi: 0.0, lo: 0.0 };
                g[q * m + p] = Dd { hi: 0.0, lo: 0.0 };
            }
        }
        if !changed {
            break;
        }
    }
    let mut lmin = g[0];
    let mut lmax = g[0];
    for i in 1..m {
        if g[i * m + i].hi < lmin.hi {
            lmin = g[i * m + i];
        }
        if g[i * m + i].hi > lmax.hi {
            lmax = g[i * m + i];
        }
    }
    let cond = dd_sqrt(dd_div(lmax, lmin));
    cond.hi + cond.lo
}
