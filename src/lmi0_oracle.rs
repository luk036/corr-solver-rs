use ellalgo_rs::arr::Arr;
use lmi_solver_rs::ldlt_mgr::LDLTMgr;
use ndarray::Array2;

pub struct LMI0Oracle {
    pub ldlt_mgr: LDLTMgr,
    mat_f: Vec<Array2<f64>>,
}

impl LMI0Oracle {
    pub fn new(mat_f: Vec<Array2<f64>>) -> Self {
        let ndim = mat_f[0].nrows();
        LMI0Oracle {
            ldlt_mgr: LDLTMgr::new(ndim),
            mat_f,
        }
    }

    pub fn assess_feas(&mut self, x: &Arr) -> Option<(Arr, f64)> {
        let n = x.len();
        let get_elem = |i: usize, j: usize| {
            let mut a = 0.0;
            for k in 0..n {
                a += self.mat_f[k][(i, j)] * x[k];
            }
            a
        };
        if self.ldlt_mgr.factor(get_elem) {
            return None;
        }
        let ep = self.ldlt_mgr.witness();
        let mut g_vec = vec![0.0; n];
        for i in 0..n {
            g_vec[i] = -self.ldlt_mgr.sym_quad(&self.mat_f[i]);
        }
        Some((Arr::from(g_vec), ep))
    }
}
