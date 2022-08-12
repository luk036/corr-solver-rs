// todo
pub mod ldlt_ext;
pub use ldlt_ext::LDLTMgr;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_ldlt1() {
        let m1 = array![[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]];
        let mut mgr1 = LDLTMgr::new(3);
        assert!(mgr1.factorize(&m1));
    }
}
