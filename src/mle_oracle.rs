use crate::lmi0_oracle::LMI0Oracle;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleFeas, OracleOptim, SingleCut};
use lmi_solver_rs::lmi_oracle::LMIOracle;
use ndarray::Array2;

pub struct MleOracle {
    Y: Array2<f64>,
    Sigma: Vec<Array2<f64>>,
    lmi0: LMI0Oracle,
    lmi: LMIOracle,
}

impl MleOracle {
    pub fn new(Sigma: Vec<Array2<f64>>, Y: Array2<f64>) -> Self {
        let two_y = &Y * 2.0;
        let lmi0 = LMI0Oracle::new(Sigma.clone());
        let lmi = LMIOracle::new(Sigma.clone(), two_y);
        MleOracle {
            Y,
            Sigma,
            lmi0,
            lmi,
        }
    }
}

impl OracleOptim<Arr> for MleOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, t: &mut f64) -> ((Arr, SingleCut), bool) {
        if let Some((g, SingleCut(ep))) = self.lmi.assess_feas(x) {
            return ((g, SingleCut(ep)), false);
        }
        if let Some((g, fj)) = self.lmi0.assess_feas(x) {
            return ((g, SingleCut(fj)), false);
        }

        let m = self.Y.nrows();

        let R = self.lmi0.ldlt_mgr.sqrt();
        let invR = inv_upper_tri(&R);
        let Rt = invR.t().to_owned();
        let S = ndarray_matmul(&invR, &Rt);
        let SY = ndarray_matmul(&S, &self.Y);

        let mut f1 = 0.0;
        let dim = R.nrows();
        for i in 0..dim {
            f1 += (R[[i, i]]).ln();
        }
        f1 *= 2.0;
        f1 += trace_ndarray(&SY);

        let n = x.len();
        let mut g = Arr::new(n);
        let V = &S - &ndarray_matmul(&SY, &S);
        for i in 0..n {
            let mut gi = 0.0;
            for r in 0..m {
                for c in 0..m {
                    gi += V[[c, r]] * self.Sigma[i][[r, c]];
                }
            }
            g[i] = gi;
        }

        let f = f1 - *t;
        if f >= 0.0 {
            return ((g, SingleCut(f)), false);
        }
        *t = f1;
        ((g, SingleCut(0.0)), true)
    }
}

fn inv_upper_tri(r: &Array2<f64>) -> Array2<f64> {
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

fn ndarray_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    let mut out = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[[i, t]] * b[[t, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

fn trace_ndarray(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut s = 0.0;
    for i in 0..n {
        s += a[[i, i]];
    }
    s
}
