use corr_solver_rs::bspline::{
    clamped_knots, eval_bspline_basis, eval_bspline_curve, generate_bspline_info,
};
use corr_solver_rs::corr_helper::{construct_distance_matrix, construct_poly_matrix};
use corr_solver_rs::eigen::{design_cond, min_eig};
use corr_solver_rs::fitting::{cccp_corr_step, corr_mle_obj, corr_omega, lsq_corr_generic};
use corr_solver_rs::halton::create_2d_sites_halton;
use corr_solver_rs::linalg;
use ellalgo_rs::arr::{linspace, Arr};
use ndarray::Array2;
use std::fs::{create_dir_all, File};
use std::io::Write;

const N_SITE: usize = 20;
const N_GRID: usize = 1200;

fn arr_to_nd(a: &Arr) -> Array2<f64> {
    let (n, m) = (a.rows(), a.cols());
    let mut out = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a.get(i, j);
        }
    }
    out
}

fn nd_to_arr(a: &Array2<f64>) -> Arr {
    let (n, m) = (a.nrows(), a.ncols());
    let mut out = Arr::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            out.set(i, j, a[[i, j]]);
        }
    }
    out
}

fn frob_nd(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn matvec(a: &Arr, x: &Arr) -> Arr {
    let n = a.rows();
    let k = a.cols();
    let mut y = Arr::new(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..k {
            s += a.get(i, j) * x[j];
        }
        y[i] = s;
    }
    y
}

struct Data {
    site: Arr,
    d: Arr,
    dmax: f64,
    d_span: f64,
    true_cov: Array2<f64>,
    xg: Arr,
}

fn make_data() -> Data {
    let site = create_2d_sites_halton(5, 4);
    let d = construct_distance_matrix(&site);
    let dmax = d.iter().cloned().fold(0.0_f64, f64::max);
    let last = site.rows() - 1;
    let mut s = 0.0;
    for j in 0..site.cols() {
        let diff = site.get(last, j) - site.get(0, j);
        s += diff * diff;
    }
    let d_span = s.sqrt();
    let mut true_cov = Array2::zeros((N_SITE, N_SITE));
    for i in 0..N_SITE {
        for j in 0..N_SITE {
            let dd = d.get(i, j);
            true_cov[[i, j]] = 4.0 * (-0.12 * dd * dd).exp();
        }
    }
    let xg = linspace(0.0, dmax, N_GRID);
    Data {
        site,
        d,
        dmax,
        d_span,
        true_cov,
        xg,
    }
}

fn make_y(d: &Arr, n: usize) -> Array2<f64> {
    let ns = d.rows();
    let mut s = Arr::zeros(ns, ns);
    for i in 0..ns {
        for j in 0..ns {
            let dd = d.get(i, j);
            s.set(i, j, (-0.12 * dd * dd).exp());
        }
    }
    let a = linalg::cholesky(&s);
    linalg::random_seed(5);
    let mut y = Arr::zeros(ns, ns);
    for _ in 0..n {
        let mut x = linalg::randn(ns);
        for v in x.iter_mut() {
            *v *= 2.0;
        }
        let ax = matvec(&a, &x);
        let noise = linalg::randn(ns);
        let mut yv = Arr::new(ns);
        for i in 0..ns {
            yv[i] = ax[i] + 1e-5 * noise[i];
        }
        for i in 0..ns {
            for j in 0..ns {
                let cur = y.get(i, j) + yv[i] * yv[j];
                y.set(i, j, cur);
            }
        }
    }
    let nf = n as f64;
    for i in 0..ns {
        for j in 0..ns {
            let cur = y.get(i, j) / nf;
            y.set(i, j, cur);
        }
    }
    arr_to_nd(&y)
}

fn poly_curve(c: &Arr, xg: &Arr) -> Arr {
    let mut out = Arr::new(xg.size());
    for (j, &x) in xg.iter().enumerate() {
        let mut v = 0.0;
        for &ci in c.iter().rev() {
            v = v * x + ci;
        }
        out[j] = v;
    }
    out
}

fn count_increasing(curve: &Arr) -> usize {
    let n = curve.size();
    let mut cnt = 0;
    for j in 0..n.saturating_sub(1) {
        if curve[j + 1] - curve[j] > 1e-9 {
            cnt += 1;
        }
    }
    cnt
}

struct Variant {
    name: String,
    sigma: Vec<Array2<f64>>,
    t: Arr,
    is_bs: bool,
    n_coeff: Option<usize>,
}

fn make_poly(dat: &Data, m: usize) -> Variant {
    Variant {
        name: format!("poly{m}"),
        sigma: construct_poly_matrix(&dat.site, m),
        t: Arr::new(0),
        is_bs: false,
        n_coeff: None,
    }
}

fn make_bs(dat: &Data, m: usize) -> Variant {
    let (sigma, t, _k) = generate_bspline_info(&dat.site, m);
    Variant {
        name: format!("bs{m}"),
        sigma,
        t,
        is_bs: true,
        n_coeff: Some(m),
    }
}

fn make_bs_old(dat: &Data, m: usize) -> Variant {
    let t = linspace(0.0, 1.2 * dat.d_span, m + 3);
    let sigma = eval_bspline_basis(&t, 2, &dat.d);
    Variant {
        name: format!("bs{m}old"),
        sigma,
        t,
        is_bs: true,
        n_coeff: Some(m),
    }
}

struct Fit {
    ok: bool,
    iters: usize,
    rel_err: f64,
    min_eig: f64,
    n_inc: usize,
    coeffs: Arr,
}

fn run_lsq(dat: &Data, v: &Variant, y: &Array2<f64>) -> Fit {
    let (c, iters) = lsq_corr_generic(y, &v.sigma, v.n_coeff);
    let mut f = Fit {
        ok: false,
        iters,
        rel_err: 0.0,
        min_eig: 0.0,
        n_inc: 0,
        coeffs: Arr::new(0),
    };
    if c.size() != v.sigma.len() || c.iter().any(|z| !z.is_finite()) {
        return f;
    }
    let om = corr_omega(&c, &v.sigma);
    let rel = frob_nd(&(&om - &dat.true_cov)) / frob_nd(&dat.true_cov);
    let me = min_eig(&nd_to_arr(&om));
    if !rel.is_finite() || !me.is_finite() {
        return f;
    }
    let curve = if v.is_bs {
        eval_bspline_curve(&v.t, 2, &c, &dat.xg)
    } else {
        poly_curve(&c, &dat.xg)
    };
    f.ok = true;
    f.rel_err = rel;
    f.min_eig = me;
    f.n_inc = count_increasing(&curve);
    f.coeffs = c;
    f
}

fn experiment1(dat: &Data, csv: &mut File) {
    println!("=== Experiment 1: design condition numbers ===");
    println!("{:<8} {:>4} {:>18}", "family", "m", "cond");
    for m in [2usize, 4, 6, 8, 10] {
        let c = design_cond(&construct_poly_matrix(&dat.site, m));
        println!("{:<8} {:>4} {:>18.10e}", "poly", m, c);
        writeln!(csv, "poly,{m},{c}").unwrap();
    }
    for m in [4usize, 6, 8, 10] {
        let (sigma, _, _) = generate_bspline_info(&dat.site, m);
        let c = design_cond(&sigma);
        println!("{:<8} {:>4} {:>18.10e}", "clamped", m, c);
        writeln!(csv, "clamped,{m},{c}").unwrap();
    }
    for m in [4usize, 6, 8, 10] {
        let sigma = eval_bspline_basis(&linspace(0.0, 1.2 * dat.d_span, m + 3), 2, &dat.d);
        let c = design_cond(&sigma);
        println!("{:<8} {:>4} {:>18.10e}", "legacy", m, c);
        writeln!(csv, "legacy,{m},{c}").unwrap();
    }
}

fn experiment2(dat: &Data) {
    println!("=== Experiment 2: knot diagnostic ===");
    println!(
        "dmax={} d_span={} 1.2*d_span={}",
        dat.dmax,
        dat.d_span,
        1.2 * dat.d_span
    );
    let t = linspace(0.0, 1.2 * dat.d_span, 7);
    print!("legacy knots m=4:");
    for i in 0..t.size() {
        print!(" {}", t[i]);
    }
    println!();
    println!("legacy valid domain m=4: [{}, {}]", t[2], t[4]);
    let n = dat.d.rows();
    let mut above_all = 0;
    for i in 0..n {
        for j in 0..n {
            if dat.d.get(i, j) > t[4] {
                above_all += 1;
            }
        }
    }
    let mut above_upper = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if dat.d.get(i, j) > t[4] {
                above_upper += 1;
            }
        }
    }
    let upper_total = n * (n - 1) / 2;
    println!(
        "entries > t(4): all={}/{} ({:.4}) upper={}/{} ({:.4})",
        above_all,
        dat.d.size(),
        above_all as f64 / dat.d.size() as f64,
        above_upper,
        upper_total,
        above_upper as f64 / upper_total as f64
    );
    let tc = clamped_knots(dat.dmax, 4, 2);
    print!("clamped knots m=4:");
    for i in 0..tc.size() {
        print!(" {}", tc[i]);
    }
    println!();
}

fn experiment3(dat: &Data, vs: &[Variant], y: &Array2<f64>, csv: &mut File) {
    println!("=== Experiment 3: fits at N=3000 ===");
    println!(
        "{:<8} {:>6} {:>14} {:>14} {:>6}",
        "variant", "iters", "rel_err", "min_eig", "nInc"
    );
    for v in vs {
        let f = run_lsq(dat, v, y);
        if !f.ok {
            println!("{:<8} FAIL", v.name);
            writeln!(csv, "{},FAIL,,,", v.name).unwrap();
            continue;
        }
        println!(
            "{:<8} {:>6} {:>14.6e} {:>14.6e} {:>6}",
            v.name, f.iters, f.rel_err, f.min_eig, f.n_inc
        );
        writeln!(
            csv,
            "{},{},{},{},{}",
            v.name, f.iters, f.rel_err, f.min_eig, f.n_inc
        )
        .unwrap();
        if v.is_bs {
            print!("  {} coeffs:", v.name);
            for i in 0..f.coeffs.size() {
                print!(" {}", f.coeffs[i]);
            }
            println!();
        }
    }
}

fn run_ccp(v: &Variant, y: &Array2<f64>, n: usize, csv: &mut File, method: &str) {
    let (x0, _lsq_iters) = lsq_corr_generic(y, &v.sigma, v.n_coeff);
    if x0.size() != v.sigma.len() || x0.iter().any(|z| !z.is_finite()) {
        println!("{method:<10} FAIL");
        writeln!(csv, "{n},{method},FAIL,,,").unwrap();
        return;
    }
    let f0 = corr_mle_obj(&x0, &v.sigma, y);
    let mut x = x0;
    let mut f_old = 1e100;
    let mut rounds = 0usize;
    let mut f1 = f0;
    for _ in 0..50 {
        let (xn, _iters) = cccp_corr_step(&v.sigma, y, x.clone(), v.n_coeff);
        if xn.size() != x.size() {
            break;
        }
        let f = corr_mle_obj(&xn, &v.sigma, y);
        rounds += 1;
        x = xn;
        f1 = f;
        if (f_old - f).abs() < 1e-8 {
            break;
        }
        f_old = f;
    }
    let me = min_eig(&nd_to_arr(&corr_omega(&x, &v.sigma)));
    println!("{method:<10} rounds={rounds:>2} f0={f0:.6} -> f1={f1:.6} min_eig={me:.6e}");
    writeln!(csv, "{n},{method},{rounds},{f0},{f1},{me}").unwrap();
}

fn experiment4(vpoly4: &Variant, vbs4: &Variant, y: &Array2<f64>, csv: &mut File) {
    println!("=== Experiment 4: CCP at N=3000 ===");
    run_ccp(vpoly4, y, 3000, csv, "ccp_poly4");
    run_ccp(vbs4, y, 3000, csv, "ccp_bs4");
}

fn experiment5(dat: &Data, vs: &[Variant], vpoly4: &Variant, vbs4: &Variant, csv: &mut File) {
    println!("=== Experiment 5: N-sweep ===");
    for n in 1..=50 {
        let y = make_y(&dat.d, n);
        println!("--- N={n} ---");
        for v in vs {
            let f = run_lsq(dat, v, &y);
            if !f.ok {
                println!("{:<8} FAIL", v.name);
                writeln!(csv, "{n},{},FAIL,,,", v.name).unwrap();
                continue;
            }
            println!(
                "{:<8} {:>6} {:>14.6e} {:>14.6e} {:>6}",
                v.name, f.iters, f.rel_err, f.min_eig, f.n_inc
            );
            writeln!(
                csv,
                "{n},{},{},{},{},{}",
                v.name, f.iters, f.rel_err, f.min_eig, f.n_inc
            )
            .unwrap();
        }
        if [1, 5, 10, 20, 50].contains(&n) {
            run_ccp(vpoly4, &y, n, csv, "ccp_poly4");
            run_ccp(vbs4, &y, n, csv, "ccp_bs4");
        }
    }
}

fn main() {
    let dat = make_data();
    let vpoly4 = make_poly(&dat, 4);
    let vpoly6 = make_poly(&dat, 6);
    let vbs4 = make_bs(&dat, 4);
    let vbs6 = make_bs(&dat, 6);
    let vbs4old = make_bs_old(&dat, 4);
    let vs = vec![vpoly4, vpoly6, vbs4, vbs6, vbs4old];

    create_dir_all("experiments/results").unwrap();

    {
        let mut csv = File::create("experiments/results/cond.csv").unwrap();
        writeln!(csv, "family,m,cond").unwrap();
        experiment1(&dat, &mut csv);
    }
    experiment2(&dat);
    let y = make_y(&dat.d, 3000);
    {
        let mut csv = File::create("experiments/results/fits.csv").unwrap();
        writeln!(csv, "variant,iters,rel_err,min_eig,nInc").unwrap();
        experiment3(&dat, &vs, &y, &mut csv);
    }
    {
        let mut csv = File::create("experiments/results/ccp.csv").unwrap();
        writeln!(csv, "N,method,rounds,f0,f1,min_eig").unwrap();
        experiment4(&vs[0], &vs[2], &y, &mut csv);
    }
    {
        let mut csv = File::create("experiments/results/nsweep.csv").unwrap();
        writeln!(csv, "N,variant,iters,rel_err,min_eig,nInc").unwrap();
        experiment5(&dat, &vs, &vs[0], &vs[2], &mut csv);
    }
}
