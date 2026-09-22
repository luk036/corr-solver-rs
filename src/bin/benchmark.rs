use std::fs::File;
use std::io::Read;
use std::time::Instant;

use corr_solver_rs::convert::arr_to_ndarray;
use corr_solver_rs::corr_helper::construct_poly_matrix;
use corr_solver_rs::fitting::{lsq_corr_poly, mle_corr_poly};
use ellalgo_rs::arr::Arr;

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
        let start = Instant::now();
        let result = lsq_corr_poly(&y, &site, m);
        let elapsed = start.elapsed().as_secs_f64();
        total_lsq += elapsed;
        lsq_iters = result.iters;
        println!(
            "  Run {}: {:.5} s, iters={}",
            run + 1,
            elapsed,
            result.iters
        );
        if run == 0 && result.ok {
            print!("  coeffs = [");
            for i in 0..m {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.5}", result.coeffs[i]);
            }
            println!("]");
        }
    }
    println!("  Avg time: {:.5} s", total_lsq / num_runs as f64);
    println!("  iters = {}", lsq_iters);

    println!("\n=== MLE Correlation ===");
    let mut total_mle = 0.0;
    let mut mle_iters = 0;
    for run in 0..num_runs {
        let start = Instant::now();
        let result = mle_corr_poly(&y, &site, m);
        let elapsed = start.elapsed().as_secs_f64();
        total_mle += elapsed;
        mle_iters = result.iters;
        println!(
            "  Run {}: {:.5} s, iters={}",
            run + 1,
            elapsed,
            result.iters
        );
    }
    println!("  Avg time: {:.5} s", total_mle / num_runs as f64);
    println!("  iters = {}", mle_iters);
}
