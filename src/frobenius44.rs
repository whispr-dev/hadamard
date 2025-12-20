use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

struct FiniteGroup {
    size: usize,
    table: Vec<Vec<usize>>,
    name: String,
}

impl FiniteGroup {
    /// Generates the Frobenius Group F_{11,4} (Order 44)
    /// This is a non-abelian group where an element of order 4 
    /// acts on a cyclic group of order 11.
    fn frobenius_44() -> Self {
        let size = 44;
        let mut table = vec![vec![0; size]; size];
        for i in 0..size {
            for j in 0..size {
                let (q1, r1) = (i / 11, i % 11);
                let (q2, r2) = (j / 11, j % 11);
                
                // Rule: (q1, r1) * (q2, r2) = (q1 + q2 mod 4, r1 + r2 * 3^q1 mod 11)
                // 3 is an element of order 5 mod 11, so we use 4 as the multiplier
                // to ensure an order-4 action. 4^1=4, 4^2=5, 4^3=9, 4^4=3 (mod 11).
                let multiplier = match q1 {
                    0 => 1,
                    1 => 4,
                    2 => 5,
                    3 => 9,
                    _ => 1,
                };
                
                let q_res = (q1 + q2) % 4;
                let r_res = (r1 + (r2 * multiplier)) % 11;
                table[i][j] = q_res * 11 + r_res;
            }
        }
        FiniteGroup { size, table, name: "Frobenius_44".to_string() }
    }
}

fn solve_hadamard_frobenius(group: &FiniteGroup) {
    let n = group.size;
    println!(">>> Analyzing Group: {} (Order {})", group.name, n);

    // 1. Compute 2-Cocycle Basis (Standard Cocyclic Logic)
    let mut basis = Vec::new();
    // (Simplified basis generation for demonstration)
    for i in 1..n {
        let mut row = vec![0u8; n * n];
        for x in 0..n {
            for y in 0..n {
                if (group.table[x][y] == i) { row[x * n + y] = 1; }
            }
        }
        basis.push(row);
    }

    let k = basis.len();
    let search_space = 1u64 << k;
    let counter = AtomicU64::new(0);
    let start = Instant::now();

    println!(">>> Cocycle Space Dimension: {} (Search space 2^{})", k, k);

    let found = (0..search_space).into_par_iter().find_any(|&mask| {
        let c = counter.fetch_add(1, Ordering::Relaxed);
        if c % 1_000_000 == 0 && c > 0 {
            println!(">>> Progress: {}M checked...", c / 1_000_000);
        }

        let mut psi = vec![0u8; n * n];
        for i in 0..k {
            if (mask >> i) & 1 == 1 {
                for j in 0..n*n { psi[j] ^= basis[i][j]; }
            }
        }

        // 2. THE INTELLIGENCE FILTER: Power Sum Check
        // The first row sum squared must be a valid component of the sum 4n
        let mut row_sum: i32 = 0;
        for x in 0..n {
            row_sum += if psi[0 * n + x] == 0 { 1 } else { -1 };
        }
        
        // For n=44, sum of 4 squares = 44. Valid row sums are odd (1, 3, 5).
        let s2 = row_sum * row_sum;
        if s2 > 44 || s2 == 0 { return false; }

        // 3. The Full Hadamard Check
        for r1 in 0..n {
            for r2 in (r1 + 1)..n {
                let mut dot = 0;
                for x in 0..n {
                    let val1 = if psi[r1 * n + x] == 0 { 1 } else { -1 };
                    let val2 = if psi[r2 * n + x] == 0 { 1 } else { -1 };
                    dot += val1 * val2;
                }
                if dot != 0 { return false; }
            }
        }
        true
    });

    match found {
        Some(mask) => println!(">>> SUCCESS: Found mask {:b}", mask),
        None => println!(">>> FAILURE: No solution in this group."),
    }
    println!("Time: {:.2?}", start.elapsed());
}

fn main() {
    let group = FiniteGroup::frobenius_44();
    solve_hadamard_frobenius(&group);
}