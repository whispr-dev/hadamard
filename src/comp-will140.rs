use rayon::prelude::*;
use std::time::Instant;

fn paf(seq: &[i8], s: usize) -> i32 {
    let n = seq.len();
    let mut sum = 0;
    for i in 0..n {
        sum += (seq[i] * seq[(i + s) % n]) as i32;
    }
    sum
}

fn generate_filtered_candidates(m: usize) -> Vec<(Vec<i8>, Vec<i32>, i32)> {
    let half = m / 2;
    let num_variants = 1 << half;
    
    (0..num_variants).into_par_iter().filter_map(|i| {
        let mut seq = vec![1i8; m];
        for bit in 0..half {
            if (i >> bit) & 1 == 1 {
                seq[bit + 1] = -1;
                seq[m - 1 - bit] = -1;
            }
        }
        let row_sum: i32 = seq.iter().map(|&x| x as i32).sum();
        let s2 = row_sum * row_sum;
        
        // Only keep candidates that can fit in the sum-of-four-squares = 140
        // Possible odd squares < 140: {1, 9, 25, 49, 81, 121}
        if matches!(s2, 1 | 9 | 25 | 49 | 81 | 121) {
            let paf_vec = (1..=(m/2)).map(|s| paf(&seq, s)).collect();
            Some((seq, paf_vec, row_sum))
        } else {
            None
        }
    }).collect()
}

fn solve_williamson_140_low_mem() {
    let m = 35;
    println!(">>> TARGET: n=140 (Memory-Optimized)");
    
    let candidates = generate_filtered_candidates(m);
    println!(">>> Filtered Candidates: {}", candidates.len());

    let start = Instant::now();
    
    // Split candidates by their squared row sums to prune the search
    // We only need to check combinations (a,b,c,d) where s_a^2 + s_b^2 + s_c^2 + s_d^2 = 140
    let target_sum = 140;
    
    // This is a triple-nested search with early exit
    // We parallelize the outer loop
    let found = candidates.par_iter().enumerate().find_map_any(|(ia, a_info)| {
        let (seq_a, paf_a, s_a) = a_info;
        
        for ib in ia..candidates.len() {
            let (seq_b, paf_b, s_b) = &candidates[ib];
            let current_s2 = s_a*s_a + s_b*s_b;
            if current_s2 >= target_sum { continue; }

            for ic in ib..candidates.len() {
                let (seq_c, paf_c, s_c) = &candidates[ic];
                let current_s2_2 = current_s2 + s_c*s_c;
                if current_s2_2 >= target_sum { continue; }

                let s_d_required_sq = target_sum - current_s2_2;
                let s_d_required = (s_d_required_sq as f64).sqrt() as i32;
                
                // Only proceed if the remaining square is a perfect square
                if s_d_required * s_d_required != s_d_required_sq { continue; }

                for id in ic..candidates.len() {
                    let (seq_d, paf_d, s_d) = &candidates[id];
                    if s_d.abs() != s_d_required { continue; }

                    // Now check the PAF sum = 0 condition
                    let mut valid = true;
                    for s_idx in 0..(m/2) {
                        if paf_a[s_idx] + paf_b[s_idx] + paf_c[s_idx] + paf_d[s_idx] != 0 {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        return Some((seq_a.clone(), seq_b.clone(), seq_c.clone(), seq_d.clone()));
                    }
                }
            }
        }
        None
    });

    match found {
        Some((a, b, c, d)) => {
            println!("\n>>> SUCCESS: n=140 Found!");
            println!("A: {:?}\nB: {:?}\nC: {:?}\nD: {:?}", a, b, c, d);
        }
        None => println!("\n>>> FAILURE: No Williamson structure found."),
    }
    println!("Total Search time: {:.2?}", start.elapsed());
}

fn main() {
    solve_williamson_140_low_mem();
}