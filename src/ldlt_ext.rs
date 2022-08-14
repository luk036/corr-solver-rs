use ndarray::prelude::*;

/**
 * LDLT factorization for LMI
 *
 *  - LDL^temp_storage square-root-free version
 *  - Option allow semidefinite
 *  - A matrix A in R^{m x m} is positive definite iff v' A v > 0
 *      for all v in R^n.
 *  - O(p^2) per iteration, independent of N
 */
#[derive(Debug)]
pub struct LDLTMgr {
    /// the rows where the process self.p.0s and self.p.1s
    pub p: (usize, usize),
    /// witness vector
    pub witness_vec: Array1<f64>,
    /// dimension
    n: usize,
    /// temporary storage
    temp_storage: Array2<f64>,
}

impl LDLTMgr {
    // static Array1<f64> zeros_vec(usize n);
    // static Array2<f64> zeros_mat(usize n);

    /**
     * Construct a new ldlt ext object
     *
     * @param[in] N dimension
     */
    pub fn new(n: usize) -> Self {
        LDLTMgr {
            n,
            p: (0, 0),
            witness_vec: Array1::zeros(n),
            temp_storage: Array2::zeros((n, n)),
        }
    }

    /**
     * Perform LDLT Factorization
     *
     * @param[in] A Symmetric Array2<f64>rix
     *
     * If $A$ is positive definite, then $p$ is zero.
     * If it is not, then $p$ is a positive integer,
     * such that $v = R^-1 e_p$ is a certificate vector
     * to make $v'*A[:p,:p]*v < 0$
     */
    pub fn factorize(&mut self, mat: &Array2<f64>) -> bool {
        self.factor(&mut |i: usize, j: usize| mat[(i, j)])
    }

    /**
     * Perform LDLT Factorization (Lazy evaluation)
     *
     * @tparam Fn
     * @param[in] get_elem function to access the elements of A
     *
     * See also: factorize()
     */
    // template <typename Callable, bool Allow_semidefinite = false>
    pub fn factor<F>(&mut self, get_elem: &mut F) -> bool
    where
        F: FnMut(usize, usize) -> f64,
    {
        self.p = (0, 0);
        // let mut (self.p.0, self.p.1) = &mut self.p;

        for i in 0..self.n {
            // let mut j = self.p.0;
            let mut d = get_elem(i, self.p.0);
            for j in self.p.0..i {
                self.temp_storage[(j, i)] = d;
                self.temp_storage[(i, j)] = d / self.temp_storage[(j, j)]; // note: temp_storage(j, i) here!
                let s = j + 1;
                d = get_elem(i, s);
                for k in self.p.0..s {
                    d -= self.temp_storage[(i, k)] * self.temp_storage[(k, s)];
                }
            }
            self.temp_storage[(i, i)] = d;

            if d <= 0.0 {
                self.p.1 = i + 1;
                break;
            }
        }

        self.is_spd()
    }

    /**
     * Is $A$ symmetric positive definite (spd)
     *
     * @return true
     * @return false
     */
    #[inline]
    pub fn is_spd(&self) -> bool {
        self.p.1 == 0
    }

    /*
     * witness that certifies $A$ is not
     * symmetric positive definite (spd)
     *
     */
    // pub fn witness(&self) -> f64;

    /*
     * Calculate v'*{A}(p,p)*v
     *
     * @param[in] A
     * @return f64
     */
    // pub fn sym_quad(&self, A: &Array1<f64>) -> f64;

    // pub fn sqrt(&self) -> Array2<f64>;
}
