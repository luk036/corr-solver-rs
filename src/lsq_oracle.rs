use crate::lmi0_oracle::LMI0Oracle;
use crate::qmi_oracle::QMIOracle;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{OracleOptim, SingleCut};
use ndarray::Array2;

pub struct LsqOracle {
    qmi: QMIOracle,
    lmi0: LMI0Oracle,
}

impl LsqOracle {
    pub fn new(_m: usize, F: Vec<Array2<f64>>, F0: Array2<f64>) -> Self {
        let qmi = QMIOracle::new(F.clone(), F0);
        let lmi0 = LMI0Oracle::new(F);
        LsqOracle { qmi, lmi0 }
    }
}

impl OracleOptim<Arr> for LsqOracle {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, t: &mut f64) -> ((Arr, SingleCut), bool) {
        let n = x.len();
        let mut g = Arr::new(n);

        let mut v = Arr::new(n - 1);
        for i in 0..n - 1 {
            v[i] = x[i];
        }

        if let Some((g0, fj)) = self.lmi0.assess_feas(&v) {
            for i in 0..n - 1 {
                g[i] = g0[i];
            }
            g[n - 1] = 0.0;
            return ((g, SingleCut(fj)), false);
        }

        self.qmi.update(x[n - 1]);
        if let Some((g1, fj)) = self.qmi.assess_feas(&v) {
            for i in 0..n - 1 {
                g[i] = g1[i];
            }
            let (start, stop) = self.qmi.gmi.ldlt_mgr.pos;
            let mut v2norm2 = 0.0;
            for i in start..stop {
                v2norm2 += self.qmi.gmi.ldlt_mgr.wit[i] * self.qmi.gmi.ldlt_mgr.wit[i];
            }
            g[n - 1] = -(v2norm2);
            return ((g, SingleCut(fj)), false);
        }

        g[n - 1] = 1.0;
        let fj = x[n - 1] - *t;
        if fj > 0.0 {
            return ((g, SingleCut(fj)), false);
        }
        *t = x[n - 1];
        ((g, SingleCut(0.0)), true)
    }
}
