use rayon::prelude::*;
use std::time::Instant;

/// Computes the Periodic Autocorrelation for a symmetric sequence at shift s
fn paf(seq: &[i8], s: usize) -> i32 {
    let n = seq.len();
    let mut sum = 0;
    for i in 0..n {
        sum += (seq[i] * seq[(i + s) % n]) as i32;
    }
    sum
}

/// Generates all possible symmetric ±1 sequences of length m
/// For m=9, symmetry means index 1=8, 2=7, 3=6, 4=5. Index 0 is always 1 (normalized).
fn generate_symmetric_sequences(m: usize) -> Vec<Vec<i8>> {
    let half = m / 2; // for m=9, half=4
    let num_variants = 1 << half;
    let mut results = Vec::new();

    for i in 0..num_variants {
        let mut seq = vec![1i8; m];
        for bit in 0..half {
            if (i >> bit) & 1 == 1 {
                seq[bit + 1] = -1;
                seq[m - 1 - bit] = -1;
            }
        }
        results.push(seq);
    }
    results
}

fn solve_williamson_36() {
    let m = 9;
    println!(">>> TARGET: Williamson Hadamard n=36 (m=9)");
    let candidates = generate_symmetric_sequences(m);
    println!(">>> Symmetric Candidates per block: {}", candidates.len());

    // We pre-calculate the PAFs for all candidates to avoid redundant math
    let pafs: Vec<Vec<i32>> = candidates.iter()
        .map(|seq| (1..=(m/2)).map(|s| paf(seq, s)).collect())
        .collect();

    let start = Instant::now();

    // Use Rayon to parallelize the 4-way search
    // We search over all combinations of 4 blocks (A, B, C, D)
    let found = (0..candidates.len()).into_par_iter().find_map_any(|ia| {
        for ib in 0..candidates.len() {
            for ic in 0..candidates.len() {
                for id in 0..candidates.len() {
                    let mut valid = true;
                    for s_idx in 0..(m/2) {
                        if pafs[ia][s_idx] + pafs[ib][s_idx] + pafs[ic][s_idx] + pafs[id][s_idx] != 0 {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        return Some((ia, ib, ic, id));
                    }
                }
            }
        }
        None
    });

    match found {
        Some((ia, ib, ic, id)) => {
            println!("\n>>> SUCCESS: Williamson Quadruplet Found!");
            println!("A: {:?}", candidates[ia]);
            println!("B: {:?}", candidates[ib]);
            println!("C: {:?}", candidates[ic]);
            println!("D: {:?}", candidates[id]);
        }
        None => println!("\n>>> FAILURE: No Williamson structure exists for m=9."),
    }
    println!("Search completed in: {:.2?}", start.elapsed());
}

fn main() {
    solve_williamson_36();
}
