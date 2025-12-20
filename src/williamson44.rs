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

fn generate_symmetric_candidates(m: usize) -> Vec<Vec<i8>> {
    let half = m / 2; // for m=11, half=5
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
        
        // --- THE POWER SUM FILTER ---
        // For m=11, row sums squared must add to 44.
        // Valid squares: 1, 9, 25. (Sum of squares logic)
        let sum: i32 = seq.iter().map(|&x| x as i32).sum();
        let s2 = sum * sum;
        if s2 == 1 || s2 == 9 || s2 == 25 {
            results.push(seq);
        }
    }
    results
}

fn solve_williamson_44() {
    let m = 11;
    println!(">>> TARGET: Williamson Hadamard n=44 (m=11)");
    
    let candidates = generate_symmetric_candidates(m);
    let num_c = candidates.len();
    println!(">>> Filtered Candidates per block: {} (Intelligence active)", num_c);

    let pafs: Vec<Vec<i32>> = candidates.iter()
        .map(|seq| (1..=(m/2)).map(|s| paf(seq, s)).collect())
        .collect();

    let start = Instant::now();

    // Meet-in-the-middle
    let mut sum_map: HashMap<Vec<i32>, (usize, usize)> = HashMap::with_capacity(num_c * num_c);
    
    for ia in 0..num_c {
        for ib in 0..num_c {
            let combined: Vec<i32> = (0..(m/2))
                .map(|s_idx| pafs[ia][s_idx] + pafs[ib][s_idx])
                .collect();
            sum_map.insert(combined, (ia, ib));
        }
    }

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
            println!("\n>>> SUCCESS: Williamson n=44 Found!");
            println!("A: {:?}", candidates[ia]);
            println!("B: {:?}", candidates[ib]);
            println!("C: {:?}", candidates[ic]);
            println!("D: {:?}", candidates[id]);
        }
        None => println!("\n>>> FAILURE: No Williamson structure for m=11."),
    }
    println!("Execution time: {:.2?}", start.elapsed());
}

fn main() {
    solve_williamson_44();
}