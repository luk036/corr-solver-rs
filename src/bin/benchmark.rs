use std::fs::File;
use std::io::Read;
use std::time::Instant;

use corr_solver_rs::corr_helper::construct_poly_matrix;
use corr_solver_rs::linalg;
use corr_solver_rs::lsq_oracle::LsqOracle;
use corr_solver_rs::mle_oracle::MleOracle;
use ellalgo_rs::arr::Arr;
use ellalgo_rs::cutting_plane::{cutting_plane_optim, Options};
use ellalgo_rs::ell::Ell;

fn read_arr(file: &mut File) -> Arr {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    let n = u64::from_le_bytes(buf) as usize;
    let mut data = vec![0.0; n * n];
    let total = n * n * 8;
    let mut bytes = vec![0u8; total];
    file.read_exact(&mut bytes).unwrap();
    for i in 0..n * n {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        data[i] = f64::from_le_bytes(word);
    }
    Arr::from_shape_vec(n, n, data)
}

fn read_site(file: &mut File) -> Arr {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    let ns = u64::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf).unwrap();
    let nd = u64::from_le_bytes(buf) as usize;
    let mut data = vec![0.0; ns * nd];
    let total = ns * nd * 8;
    let mut bytes = vec![0u8; total];
    file.read_exact(&mut bytes).unwrap();
    for i in 0..ns * nd {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        data[i] = f64::from_le_bytes(word);
    }
    Arr::from_shape_vec(ns, nd, data)
}

fn arr_to_ndarray(a: &Arr) -> ndarray::Array2<f64> {
    let n = a.rows();
    let m = a.cols();
    let mut out = ndarray::Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            out[[i, j]] = a.get(i, j);
        }
    }
    out
}

fn main() {
    let num_runs = 5;
    let m = 4usize;

    println!("Reading data from benchmark_data.bin...");
    let mut file = File::open("benchmark_data.bin")
        .expect("benchmark_data.bin not found (run C++ benchmark first)");
    let y_arr = read_arr(&mut file);
    let n_sites = y_arr.rows();
    println!("Matrix dimension: {}", n_sites);

    let site = read_site(&mut file);
    println!("Site: {} x {}", site.rows(), site.cols());

    let sig_vec = construct_poly_matrix(&site, m);
    println!("Poly matrices: {}", sig_vec.len());

    let y = arr_to_ndarray(&y_arr);

    println!("\n=== LSQ Correlation ===");
    let mut total_lsq = 0.0;
    let mut lsq_iters = 0;
    for run in 0..num_runs {
        let mut omega = LsqOracle::new(n_sites, sig_vec.clone(), y.clone());
        let norm_y = linalg::norm(&y_arr);
        let norm_y2 = 32.0 * norm_y * norm_y;
        let mut val = vec![256.0; m + 1];
        val[m] = norm_y2 * norm_y2;
        let mut x = Arr::new(m + 1);
        x[0] = 4.0;
        x[m] = norm_y2 / 2.0;
        let mut ellip = Ell::new(Arr::from(val), x);
        let mut t = 1e100;
        let start = Instant::now();
        let (x_best, num_iters) =
            cutting_plane_optim(&mut omega, &mut ellip, &mut t, &Options::default());
        let elapsed = start.elapsed().as_secs_f64();
        total_lsq += elapsed;
        lsq_iters = num_iters;
        println!("  Run {}: {:.5} s, iters={}", run + 1, elapsed, num_iters);
        if run == 0 {
            if let Some(xb) = &x_best {
                print!("  coeffs = [");
                for i in 0..m {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{:.5}", xb[i]);
                }
                println!("]");
            }
        }
    }
    println!("  Avg time: {:.5} s", total_lsq / num_runs as f64);
    println!("  iters = {}", lsq_iters);

    println!("\n=== MLE Correlation ===");
    let mut total_mle = 0.0;
    let mut mle_iters = 0;
    for run in 0..num_runs {
        let mut omega = MleOracle::new(sig_vec.clone(), y.clone());
        let mut x = Arr::new(m);
        x[0] = 4.0;
        let mut ellip = Ell::new_with_scalar(500.0, x);
        let mut t = 1e100;
        let start = Instant::now();
        let (_x_best, num_iters) =
            cutting_plane_optim(&mut omega, &mut ellip, &mut t, &Options::default());
        let elapsed = start.elapsed().as_secs_f64();
        total_mle += elapsed;
        mle_iters = num_iters;
        println!("  Run {}: {:.5} s, iters={}", run + 1, elapsed, num_iters);
    }
    println!("  Avg time: {:.5} s", total_mle / num_runs as f64);
    println!("  iters = {}", mle_iters);
}
