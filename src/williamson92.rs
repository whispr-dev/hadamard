use std::collections::HashMap;
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

fn generate_symmetric_sequences(m: usize) -> Vec<Vec<i8>> {
    let half = m / 2; 
    let num_variants = 1 << half;
    (0..num_variants).map(|i| {
        let mut seq = vec![1i8; m];
        for bit in 0..half {
            if (i >> bit) & 1 == 1 {
                seq[bit + 1] = -1;
                seq[m - 1 - bit] = -1;
            }
        }
        seq
    }).collect()
}

fn solve_williamson_92() {
    let m = 23;
    println!(">>> TARGET: THE HISTORIC n=92 (m=23)");
    let candidates = generate_symmetric_sequences(m);
    let num_c = candidates.len();
    println!(">>> Candidates per block: {}", num_c);

    let pafs: Vec<Vec<i32>> = candidates.iter()
        .map(|seq| (1..=(m/2)).map(|s| paf(seq, s)).collect())
        .collect();

    let start = Instant::now();

    // Meet-in-the-middle: Store A + B results
    println!(">>> Building Hash Map for (A + B) pairings...");
    let mut sum_map: HashMap<Vec<i32>, (usize, usize)> = HashMap::with_capacity(num_c * num_c);
    
    for ia in 0..num_c {
        for ib in 0..num_c {
            let combined: Vec<i32> = (0..(m/2))
                .map(|s_idx| pafs[ia][s_idx] + pafs[ib][s_idx])
                .collect();
            sum_map.insert(combined, (ia, ib));
        }
    }

    println!(">>> Map complete. Searching for matching (C + D)...");

    // Search for C + D such that (C + D) == -(A + B)
    let found = (0..num_c).into_par_iter().find_map_any(|ic| {
        for id in 0..num_c {
            let target: Vec<i32> = (0..(m/2))
                .map(|s_idx| -(pafs[ic][s_idx] + pafs[id][s_idx]))
                .collect();
            
            if let Some(&(ia, ib)) = sum_map.get(&target) {
                return Some((ia, ib, ic, id));
            }
        }
        None
    });

    match found {
        Some((ia, ib, ic, id)) => {
            println!("\n>>> SUCCESS: Historic n=92 Quadruplet Found!");
            println!("A: {:?}", candidates[ia]);
            println!("B: {:?}", candidates[ib]);
            println!("C: {:?}", candidates[ic]);
            println!("D: {:?}", candidates[id]);
        }
        None => println!("\n>>> FAILURE: No Williamson structure found."),
    }
    println!("Total execution time: {:.2?}", start.elapsed());
}

fn main() {
    solve_williamson_92();
}