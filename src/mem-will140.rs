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

fn generate_filtered_candidates(m: usize) -> Vec<Vec<i8>> {
    let half = m / 2; // m=35, half=17
    let num_variants = 1 << half;
    
    // Valid absolute row sums for m=35: {1, 3, 5, 7, 9, 11}
    // These are the only odd numbers whose squares could possibly sum to 140.
    (0..num_variants).into_par_iter().filter_map(|i| {
        let mut seq = vec![1i8; m];
        for bit in 0..half {
            if (i >> bit) & 1 == 1 {
                seq[bit + 1] = -1;
                seq[m - 1 - bit] = -1;
            }
        }
        let sum: i32 = seq.iter().map(|&x| x as i32).sum::<i32>().abs();
        if matches!(sum, 1 | 3 | 5 | 7 | 9 | 11) {
            Some(seq)
        } else {
            None
        }
    }).collect()
}

fn solve_williamson_140() {
    let m = 35;
    println!(">>> TARGET: THE DEEP DIVE n=140 (m=35)");
    
    let start_gen = Instant::now();
    let candidates = generate_filtered_candidates(m);
    let num_c = candidates.len();
    println!(">>> Filtered Candidates: {} | Generation time: {:.2?}", num_c, start_gen.elapsed());

    let pafs: Vec<Vec<i32>> = candidates.par_iter()
        .map(|seq| (1..=(m/2)).map(|s| paf(seq, s)).collect())
        .collect();

    let start_search = Instant::now();
    println!(">>> Building Hash Map (Meet-in-the-middle)...");

    // We store the sums in a HashMap. 
    // Key: The vector of combined PAFs. Value: indices of (ia, ib)
    let mut sum_map: HashMap<Vec<i32>, (usize, usize)> = HashMap::with_capacity(num_c * 1000); 

    // Note: Due to the size, we'll iterate and fill. 
    // For n=140, we might need to be careful with RAM.
    for ia in 0..num_c {
        for ib in ia..num_c { // ib starts at ia to avoid redundant pairs
            let combined: Vec<i32> = (0..(m/2))
                .map(|s_idx| pafs[ia][s_idx] + pafs[ib][s_idx])
                .collect();
            sum_map.insert(combined, (ia, ib));
        }
        if ia % 1000 == 0 { println!(">>> Map Building: {}/{}", ia, num_c); }
    }

    println!(">>> Map complete. Scanning for match...");

    let found = (0..num_c).into_par_iter().find_map_any(|ic| {
        for id in ic..num_c {
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
            println!("\n>>> SUCCESS: Williamson n=140 Found!");
            println!("A: {:?}", candidates[ia]);
            println!("B: {:?}", candidates[ib]);
            println!("C: {:?}", candidates[ic]);
            println!("D: {:?}", candidates[id]);
        }
        None => println!("\n>>> FAILURE: No Williamson structure for m=35."),
    }
    println!("Total Search time: {:.2?}", start_search.elapsed());
}

fn main() {
    solve_williamson_140();
}