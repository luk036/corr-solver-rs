use corr_solver_rs::bspline::{
    clamped_knots, eval_bspline_basis, generate_bspline_info, mono_oracle, MonoDecreasingOracle2,
};
use corr_solver_rs::corr_helper::{construct_distance_matrix, construct_poly_matrix};
use corr_solver_rs::eigen::design_cond;
use corr_solver_rs::halton::{create_2d_sites_halton, radical_inverse};
use ellalgo_rs::arr::{linspace, Arr};
use ellalgo_rs::cutting_plane::{OracleOptim, SingleCut};
use std::cell::Cell;
use std::rc::Rc;

const NX: usize = 5;
const NY: usize = 4;

fn assert_rel(actual: f64, expected: f64, tol: f64) {
    let rel = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel <= tol,
        "expected {expected}, got {actual} (rel err {rel:e})"
    );
}

fn halton_setup() -> (Arr, Arr, f64, f64) {
    let site = create_2d_sites_halton(NX, NY);
    let d = construct_distance_matrix(&site);
    let dmax = d.iter().cloned().fold(0.0_f64, f64::max);
    let d_span = d.get(0, 19);
    (site, d, dmax, d_span)
}

fn legacy_knots(d_span: f64, m: usize, k: usize) -> Arr {
    linspace(0.0, 1.2 * d_span, m + k + 1)
}

#[test]
fn test_halton_sites() {
    let site = create_2d_sites_halton(NX, NY);
    assert_eq!(site.rows(), 20);
    assert_eq!(site.cols(), 2);
    assert_eq!(radical_inverse(1, 2), 0.5);
    assert_eq!(radical_inverse(1, 3), 1.0 / 3.0);
    assert_eq!(site.get(0, 0), 5.0);
    assert_rel(site.get(0, 1), 2.6666666666666665, 1e-12);
    assert_eq!(site.get(6, 0), 8.75);
    assert_rel(site.get(6, 1), 4.444444444444445, 1e-12);
    assert_eq!(site.get(19, 0), 1.5625);
    assert_rel(site.get(19, 1), 5.925925925925926, 1e-12);
}

#[test]
fn test_distance_dmax_dspan() {
    let (_, _, dmax, d_span) = halton_setup();
    assert_rel(dmax, 10.096248912961826, 1e-12);
    assert_rel(d_span, 4.737000862261608, 1e-12);
}

#[test]
fn test_clamped_knots_m4() {
    let (_, _, dmax, _) = halton_setup();
    let t = clamped_knots(dmax, 4, 2);
    assert_eq!(t.len(), 7);
    let expected = [
        0.0,
        0.0,
        0.0,
        5.048124456480913,
        10.096248912961826,
        10.096248912961826,
        10.096248912961826,
    ];
    for (i, e) in expected.iter().enumerate() {
        assert_rel(t[i], *e, 1e-12);
    }
}

#[test]
fn test_clamped_basis_m4() {
    let (site, _, _, _) = halton_setup();
    let (sigma, t, k) = generate_bspline_info(&site, 4);
    assert_eq!(k, 2);
    assert_eq!(sigma.len(), 4);
    assert_rel(t[3], 5.048124456480913, 1e-12);
    assert_rel(sigma[0][[0, 1]], 0.07612753830941012, 1e-12);
    assert_rel(sigma[0][[0, 19]], 0.0037984445216406814, 1e-12);
    assert_eq!(sigma[3][[0, 1]], 0.0);
    assert_eq!(sigma[3][[0, 19]], 0.0);
    let sum: f64 = (0..4).map(|b| sigma[b][[0, 19]]).sum();
    assert_rel(sum, 1.0, 1e-12);
}

#[test]
fn test_legacy_knots_and_extrapolation() {
    let (_, d, _, d_span) = halton_setup();
    let t = legacy_knots(d_span, 4, 2);
    assert_eq!(t.len(), 7);
    let expected = [
        0.0,
        0.9474001724523217,
        1.8948003449046433,
        2.842200517356965,
        3.7896006898092867,
        4.737000862261608,
        5.68440103471393,
    ];
    for (i, e) in expected.iter().enumerate() {
        assert_rel(t[i], *e, 1e-12);
    }
    assert_rel(1.2 * d_span, 5.68440103471393, 1e-12);
    let sigma = eval_bspline_basis(&t, 2, &d);
    assert_rel(sigma[0][[0, 19]], 0.0, 1e-12);
    assert_rel(sigma[3][[0, 19]], 2.0, 1e-12);
}

#[test]
fn test_design_cond() {
    let (site, d, _, d_span) = halton_setup();

    println!("cond_poly:");
    for (m, expected) in [
        (2usize, 11.59155345705951),
        (4, 1210.3830070574695),
        (6, 187488.94022165914),
        (8, 67217446.67806517),
        (10, 34546585438.8765),
    ] {
        let cond = design_cond(&construct_poly_matrix(&site, m));
        println!("  {m} -> {cond:.17e}");
        assert_rel(cond, expected, 1e-9);
    }

    println!("cond_bs_clamped:");
    for (m, expected) in [
        (4usize, 5.052018587714944),
        (6, 6.478335493279365),
        (8, 6.507672636371483),
        (10, 6.751181601819613),
    ] {
        let (sigma, _, _) = generate_bspline_info(&site, m);
        let cond = design_cond(&sigma);
        println!("  {m} -> {cond:.17e}");
        assert_rel(cond, expected, 1e-9);
    }

    println!("cond_bs_legacy:");
    for (m, expected) in [
        (4usize, 40.26126074981864),
        (6, 99.40044633714751),
        (8, 203.56551712290315),
        (10, 409.4721392889807),
    ] {
        let sigma = eval_bspline_basis(&legacy_knots(d_span, m, 2), 2, &d);
        let cond = design_cond(&sigma);
        println!("  {m} -> {cond:.17e}");
        assert_rel(cond, expected, 1e-9);
    }
}

#[test]
fn test_mono_oracle() {
    let dec = Arr::from(vec![3.0, 2.0, 1.0, 0.5]);
    assert!(mono_oracle(&dec).is_none());

    let inc = Arr::from(vec![1.0, 2.0, 1.0]);
    let (g, fj) = mono_oracle(&inc).expect("increasing sequence must violate");
    assert_rel(fj, 1.0, 1e-12);
    assert_eq!(g[0], -1.0);
    assert_eq!(g[1], 1.0);
    assert_eq!(g[2], 0.0);
}

struct StubBasis {
    calls: Rc<Cell<usize>>,
}

impl OracleOptim<Arr> for StubBasis {
    type CutChoice = SingleCut;

    fn assess_optim(&mut self, x: &Arr, _t: &mut f64) -> ((Arr, SingleCut), bool) {
        self.calls.set(self.calls.get() + 1);
        ((Arr::new(x.len()), SingleCut(0.0)), true)
    }
}

#[test]
fn test_mono_decreasing_oracle2() {
    let calls = Rc::new(Cell::new(0));
    let mut t = 1e100;

    // Leading 2 entries (2, 1) are decreasing -> delegate to the basis oracle.
    let mut oracle = MonoDecreasingOracle2::new(
        StubBasis {
            calls: calls.clone(),
        },
        Some(2),
    );
    let x = Arr::from(vec![2.0, 1.0, 5.0]);
    let ((g, SingleCut(f)), shrunk) = oracle.assess_optim(&x, &mut t);
    assert!(shrunk);
    assert_eq!(calls.get(), 1);
    assert_rel(f, 0.0, 1e-12);
    assert_eq!(g.len(), 3);
    assert_eq!(g[0], 0.0);
    assert_eq!(g[1], 0.0);
    assert_eq!(g[2], 0.0);

    // n_coeff = 3 constrains all entries; (2, 1, 5) increases at index 1.
    let mut oracle2 = MonoDecreasingOracle2::new(
        StubBasis {
            calls: calls.clone(),
        },
        Some(3),
    );
    let ((g2, SingleCut(f2)), shrunk2) = oracle2.assess_optim(&x, &mut t);
    assert!(!shrunk2);
    assert_eq!(calls.get(), 1);
    assert_rel(f2, 4.0, 1e-12);
    assert_eq!(g2[0], 0.0);
    assert_eq!(g2[1], -1.0);
    assert_eq!(g2[2], 1.0);

    // n_coeff = 2 must leave the trailing entry unconstrained (zero-padded cut).
    let mut oracle3 = MonoDecreasingOracle2::new(
        StubBasis {
            calls: calls.clone(),
        },
        Some(2),
    );
    let x3 = Arr::from(vec![1.0, 2.0, 5.0]);
    let ((g3, SingleCut(f3)), _) = oracle3.assess_optim(&x3, &mut t);
    assert_eq!(calls.get(), 1);
    assert_rel(f3, 1.0, 1e-12);
    assert_eq!(g3[0], -1.0);
    assert_eq!(g3[1], 1.0);
    assert_eq!(g3[2], 0.0);
}

#[test]
fn test_generate_bspline_info_rejects_small_m() {
    let (site, _, _, _) = halton_setup();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| generate_bspline_info(&site, 2));
    std::panic::set_hook(prev);
    assert!(result.is_err());
}
