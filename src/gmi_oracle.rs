use ellalgo_rs::arr::Arr;
use lmi_solver_rs::ldlt_mgr::LDLTMgr;

pub trait HOmni {
    fn reset_count(&self);
    fn eval(&self, row: usize, col: usize, x: &Arr) -> f64;
    fn neg_grad_sym_quad(&self, ldlt_mgr: &LDLTMgr, x: &Arr) -> Arr;
}

pub struct GMIOracle {
    pub ldlt_mgr: LDLTMgr,
}

impl GMIOracle {
    pub fn new(m: usize) -> Self {
        GMIOracle {
            ldlt_mgr: LDLTMgr::new(m),
        }
    }

    pub fn assess_feas<H: HOmni>(&mut self, x: &Arr, h: &H) -> Option<(Arr, f64)> {
        h.reset_count();
        let get_elem = |i: usize, j: usize| h.eval(i, j, x);
        if self.ldlt_mgr.factor(get_elem) {
            return None;
        }
        let ep = self.ldlt_mgr.witness();
        let g = h.neg_grad_sym_quad(&self.ldlt_mgr, x);
        Some((g, ep))
    }
}
