use ellalgo_rs::arr::Arr;
use lmi_solver_rs::ldlt_mgr::LDLTMgr;
use ndarray::Array2;
use std::cell::{Cell, RefCell};
use crate::gmi_oracle::{GMIOracle, HOmni};

pub struct Qmi {
    pub F: Vec<Array2<f64>>,
    pub F0: Array2<f64>,
    pub Fx: RefCell<Array2<f64>>,
    pub count: Cell<usize>,
    pub t: Cell<f64>,
    pub m: usize,
    pub nx: Cell<usize>,
}

impl Qmi {
    pub fn new(F: Vec<Array2<f64>>, F0: Array2<f64>) -> Self {
        let m = F0.ncols();
        let n = F0.nrows();
        Qmi {
            F,
            F0,
            Fx: RefCell::new(Array2::zeros((m, n))),
            count: Cell::new(0),
            t: Cell::new(0.0),
            m,
            nx: Cell::new(0),
        }
    }

    pub fn update(&self, t: f64) {
        self.t.set(t);
    }
}

impl HOmni for Qmi {
    fn reset_count(&self) {
        self.count.set(0);
    }

    fn eval(&self, row: usize, col: usize, x: &Arr) -> f64 {
        if row < col {
            panic!("Qmi::eval: row < col not supported");
        }
        if self.count.get() < row + 1 {
            let nx = x.len();
            self.nx.set(nx);
            self.count.set(row + 1);
            let mut fx = self.Fx.borrow_mut();
            for c in 0..self.m {
                let mut val = self.F0[[c, row]];
                for k in 0..nx {
                    val -= self.F[k][[c, row]] * x[k];
                }
                fx[[row, c]] = val;
            }
        }
        let fx = self.Fx.borrow();
        let mut a = 0.0;
        for c in 0..self.m {
            a -= fx[[row, c]] * fx[[col, c]];
        }
        if row == col {
            a += self.t.get();
        }
        a
    }

    fn neg_grad_sym_quad(&self, ldlt_mgr: &LDLTMgr, _x: &Arr) -> Arr {
        let (start, stop) = ldlt_mgr.pos;
        let nx = self.nx.get();
        let fx = self.Fx.borrow();
        let mut wit_vec = vec![0.0; self.m];
        for (i, w) in wit_vec[start..stop].iter_mut().enumerate() {
            *w = ldlt_mgr.wit[start + i];
        }
        let mut Av = vec![0.0; fx.ncols()];
        for c in 0..fx.ncols() {
            let mut s = 0.0;
            for r in start..stop {
                s += wit_vec[r] * fx[[r, c]];
            }
            Av[c] = s;
        }
        let mut g = Arr::new(nx);
        for k in 0..nx {
            let Fk = &self.F[k];
            let mut vFkp = 0.0;
            for r in start..stop {
                let mut row_dot = 0.0;
                for c in 0..fx.ncols() {
                    row_dot += Fk[[r, c]] * Av[c];
                }
                vFkp += wit_vec[r] * row_dot;
            }
            g[k] = -2.0 * vFkp;
        }
        g
    }
}

pub struct QMIOracle {
    pub qmi: Qmi,
    pub gmi: GMIOracle,
}

impl QMIOracle {
    pub fn new(F: Vec<Array2<f64>>, F0: Array2<f64>) -> Self {
        let m = F0.ncols();
        let qmi = Qmi::new(F, F0);
        let gmi = GMIOracle::new(m);
        QMIOracle { qmi, gmi }
    }

    pub fn update(&self, t: f64) {
        self.qmi.update(t);
    }

    pub fn assess_feas(&mut self, x: &Arr) -> Option<(Arr, f64)> {
        self.gmi.assess_feas(x, &self.qmi)
    }
}
