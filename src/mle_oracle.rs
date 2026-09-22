use crate::lmi0_oracle::LMI0Oracle;
use crate::mle_common::{optim_cut, MleScratch};
use crate::ndops;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleFeas, OracleOptim, SingleCut};
use lmi_solver_rs::lmi_oracle::LMIOracle;
use ndarray::Array2;

pub struct MleOracle {
    Y: Array2<f64>,
    Sigma: Vec<Array2<f64>>,
    lmi0: LMI0Oracle,
    lmi: LMIOracle,
    scratch: MleScratch,
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
            scratch: MleScratch::new(),
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

        self.scratch.update(&mut self.lmi0, &self.Y);
        let s = &self.scratch.s;
        let sy = &self.scratch.sy;

        let mut f1 = 0.0;
        let dim = self.scratch.r.nrows();
        for i in 0..dim {
            f1 += self.scratch.r[[i, i]].ln();
        }
        f1 *= 2.0;
        f1 += ndops::trace(sy);

        let n = x.len();
        let mut g = Arr::new(n);
        let v = s - &ndops::matmul(sy, s);
        for i in 0..n {
            let mut gi = 0.0;
            for r in 0..m {
                for c in 0..m {
                    gi += v[[c, r]] * self.Sigma[i][[r, c]];
                }
            }
            g[i] = gi;
        }

        optim_cut(g, f1, t)
    }
}
