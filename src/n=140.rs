use std::collections::HashMap;
use rayon::prelude::*;
use std::time::Instant;

type PAF = Vec<i32>;

#[derive(Clone)]
struct Candidate {
    seq: Vec<i8>,
    paf: PAF,
    s2: i32,
}

fn paf_calc(seq: &[i8], s: usize) -> i32 {
    let n = seq.len();
    let mut sum = 0;
    for i in 0..n {
        sum += (seq[i] * seq[(i + s) % n]) as i32;
    }
    sum
}

fn generate_partitioned_candidates(m: usize) -> HashMap<i32, Vec<Candidate>> {
    let half = m / 2;
    let num_variants = 1 << half;
    
    let raw: Vec<Candidate> = (0..num_variants).into_par_iter().filter_map(|i| {
        let mut seq = vec![1i8; m];
        for bit in 0..half {
            if (i >> bit) & 1 == 1 {
                seq[bit + 1] = -1;
                seq[m - 1 - bit] = -1;
            }
        }
        let row_sum: i32 = seq.iter().map(|&x| x as i32).sum();
        let s2 = row_sum * row_sum;
        
        if matches!(s2, 1 | 9 | 25 | 49 | 81 | 121) {
            let p = (1..=(m/2)).map(|s| paf_calc(&seq, s)).collect();
            Some(Candidate { seq, paf: p, s2 })
        } else {
            None
        }
    }).collect();

    let mut map = HashMap::new();
    for c in raw {
        map.entry(c.s2).or_insert_with(Vec::new).push(c);
    }
    map
}

fn main() {
    let m = 35;
    println!(">>> TARGET: n=140 | Partitioned Meet-in-the-Middle");
    let partitions = generate_partitioned_candidates(m);
    
    // Valid sum-of-four-square combinations for 140
    let quads = vec![
        (1, 9, 9, 121),
        (1, 9, 49, 81),
        (9, 25, 25, 81),
        (9, 9, 41, 81), // wait, 41 isn't an odd square.
        (25, 25, 9, 81),
        (49, 49, 1, 40), // non-square
        (25, 25, 41, 49), // non-square
        (1, 49, 41, 49), // non-square
    ];
    // Let's use a cleaner approach: find all a2+b2+c2+d2 = 140
    let keys: Vec<i32> = vec![1, 9, 25, 49, 81, 121];
    let mut combinations = Vec::new();
    for &a in &keys {
        for &b in &keys {
            for &c in &keys {
                for &d in &keys {
                    if a + b + c + d == 140 {
                        let mut v = vec![a, b, c, d];
                        v.sort();
                        if !combinations.contains(&v) { combinations.push(v); }
                    }
                }
            }
        }
    }

    println!(">>> Valid Square-Sum Sectors found: {}", combinations.len());
    let start = Instant::now();

    for comb in combinations {
        let (s2a, s2b, s2c, s2d) = (comb[0], comb[1], comb[2], comb[3]);
        println!(">>> Searching Sector: {} + {} + {} + {} = 140", s2a, s2b, s2c, s2d);
        
        let empty = Vec::new();
        let list_a = partitions.get(&s2a).unwrap_or(&empty);
        let list_b = partitions.get(&s2b).unwrap_or(&empty);
        let list_c = partitions.get(&s2c).unwrap_or(&empty);
        let list_d = partitions.get(&s2d).unwrap_or(&empty);

        // Build Map for A + B
        let mut ab_map: HashMap<PAF, (usize, usize)> = HashMap::with_capacity(list_a.len() * list_b.len());
        for (ia, ca) in list_a.iter().enumerate() {
            for (ib, cb) in list_b.iter().enumerate() {
                let combined: PAF = ca.paf.iter().zip(cb.paf.iter()).map(|(x, y)| x + y).collect();
                ab_map.insert(combined, (ia, ib));
            }
        }

        // Search in C + D
        for (ic, cc) in list_c.iter().enumerate() {
            for (id, cd) in list_d.iter().enumerate() {
                let target: PAF = cc.paf.iter().zip(cd.paf.iter()).map(|(x, y)| -(x + y)).collect();
                if let Some(&(ia, ib)) = ab_map.get(&target) {
                    println!("\n>>> SUCCESS: Sector matched!");
                    println!("A (s2={}): {:?}", s2a, list_a[ia].seq);
                    println!("B (s2={}): {:?}", s2b, list_b[ib].seq);
                    println!("C (s2={}): {:?}", s2c, list_c[ic].seq);
                    println!("D (s2={}): {:?}", s2d, list_d[id].seq);
                    println!("Total time: {:.2?}", start.elapsed());
                    return;
                }
            }
        }
    }
    println!("No solution found in these sectors.");
}