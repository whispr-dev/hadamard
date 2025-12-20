use rayon::prelude::*;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// I. THE ALGEBRAIC OBJECTS (Cayley Tables)
// ============================================================================

#[derive(Clone, Debug)]
struct FiniteGroup {
    size: usize,
    table: Vec<Vec<usize>>, // table[i][j] = i * j
    name: String,
}

impl FiniteGroup {
    /// Generates the Cyclic Group Z_n
    fn cyclic(n: usize) -> Self {
        let mut table = vec![vec![0; n]; n];
        for i in 0..n {
            for j in 0..n {
                table[i][j] = (i + j) % n;
            }
        }
        FiniteGroup { size: n, table, name: format!("Z_{}", n) }
    }

    /// Generates the Dihedral Group D_n of order 2n
    /// Elements 0..n are rotations (r^i), n..2n are reflections (s r^i)
    /// Relations: r^n = 1, s^2 = 1, s r s = r^{-1}
    fn dihedral(n: usize) -> Self {
        let size = 2 * n;
        let mut table = vec![vec![0; size]; size];

        for i in 0..size {
            for j in 0..size {
                let (is_rot_i, val_i) = if i < n { (true, i) } else { (false, i - n) };
                let (is_rot_j, val_j) = if j < n { (true, j) } else { (false, j - n) };

                if is_rot_i && is_rot_j {
                    // r^i * r^j = r^{i+j}
                    table[i][j] = (val_i + val_j) % n;
                } else if is_rot_i && !is_rot_j {
                    // r^i * s r^j = s r^{j - i} (wait, relations are tricky)
                    // Standard: s r^i = r^{-i} s. 
                    // Let's stick to standard definition:
                    // Elements are (k, x) where k in {0,1}, x in Z_n. k=0 is rot.
                    // (0, x) * (0, y) = (0, x+y)
                    // (0, x) * (1, y) = (1, x+y)
                    // (1, x) * (0, y) = (1, x-y)  <-- The twist
                    // (1, x) * (1, y) = (0, x-y)
                    table[i][j] = (val_i + val_j) % n + n; // THIS IS WRONG for D_n standard.
                }
                // Let's use the explicit logic below for safety
            }
        }
        
        // Correct implementation of D_n
        for i in 0..size {
            for j in 0..size {
                let a = if i < n { 0 } else { 1 }; // 0 = rot, 1 = ref
                let x = if i < n { i } else { i - n };
                
                let b = if j < n { 0 } else { 1 };
                let y = if j < n { j } else { j - n };

                // Group law for D_n: (a, x) * (b, y) = (a+b, x + (-1)^a * y)
                let res_a = (a + b) % 2;
                let sign = if a == 0 { 1 } else { -1 };
                // Rust modulo is tricky with negatives, so add n
                let res_x = (x as isize + sign * y as isize).rem_euclid(n as isize) as usize;

                table[i][j] = if res_a == 0 { res_x } else { res_x + n };
            }
        }

        FiniteGroup { size, table, name: format!("D_{}", n) }

    }

    /// Generates the Binary Tetrahedral Group SL(2, 3) of order 24
    fn sl2_3() -> Self {
        let mut elements = Vec::new();
        // Generate all 2x2 matrices over F3 (0, 1, 2)
        for a in 0..3 {
            for b in 0..3 {
                for c in 0..3 {
                    for d in 0..3 {
                        // Determinant: (ad - bc) mod 3
                        let det = (a * d + 30 - b * c) % 3;
                        if det == 1 {
                            elements.push([a, b, c, d]);
                        }
                    }
                }
            }
        }

        let size = elements.len(); // Should be 24
        let mut table = vec![vec![0; size]; size];

        for i in 0..size {
            for j in 0..size {
                let m1 = elements[i];
                let m2 = elements[j];
                
                // Matrix multiplication mod 3
                let res = [
                    (m1[0] * m2[0] + m1[1] * m2[2]) % 3, // top-left
                    (m1[0] * m2[1] + m1[1] * m2[3]) % 3, // top-right
                    (m1[2] * m2[0] + m1[3] * m2[2]) % 3, // bot-left
                    (m1[2] * m2[1] + m1[3] * m2[3]) % 3, // bot-right
                ];

                // Find the index of the resulting matrix
                table[i][j] = elements.iter().position(|&m| m == res).unwrap();
            }
        }

        FiniteGroup { size, table, name: "SL(2, 3)".to_string() }
    }

    /// Generates the Symmetric Group S_4 of order 24
    fn s4() -> Self {
        use itertools::Itertools;
        let elements: Vec<Vec<usize>> = (0..4).permutations(4).collect();
        let size = elements.len(); // 24
        let mut table = vec![vec![0; size]; size];

        for i in 0..size {
            for j in 0..size {
                let p1 = &elements[i];
                let p2 = &elements[j];
                
                // Permutation composition: (p1 ∘ p2)(x) = p1(p2(x))
                let mut res = vec![0; 4];
                for k in 0..4 {
                    res[k] = p1[p2[k]];
                }

                table[i][j] = elements.iter().position(|p| p == &res).unwrap();
            }
        }

        FiniteGroup { size, table, name: "S_4".to_string() }
    }

    /// Generates the Dicyclic Group Dic_n of order 4n
    /// Elements: a^k (0 to 2n-1) and x*a^k (2n to 4n-1)
    fn dicyclic(n: usize) -> Self {
        let size = 4 * n;
        let mut table = vec![vec![0; size]; size];

        for i in 0..size {
            for j in 0..size {
                // Decompose i and j into (is_x, power_of_a)
                let (xi, pi) = if i < 2 * n { (0, i) } else { (1, i - 2 * n) };
                let (xj, pj) = if j < 2 * n { (0, j) } else { (1, j - 2 * n) };

                let (res_x, res_p) = match (xi, xj) {
                    (0, 0) => (0, (pi + pj) % (2 * n)),
                    (0, 1) => (1, (pj + 2 * n - pi) % (2 * n)),
                    (1, 0) => (1, (pi + pj) % (2 * n)),
                    (1, 1) => (0, (pi + n + 2 * n - pj) % (2 * n)),
                    _ => unreachable!(),
                };

                table[i][j] = if res_x == 0 { res_p } else { res_p + 2 * n };
            }
        }

        FiniteGroup { size, table, name: format!("Dic_{}", n) }
    }
}

// ============================================================================
// II. THE COHOMOLOGICAL SOLVER (Linear Algebra over F2)
// ============================================================================

type CocycleVec = Vec<u8>; // Flattened vector of size n*n

/// Solves the linear system d(psi) = 0 to find a basis for Z^2(G, F2).
fn compute_cocycle_basis(group: &FiniteGroup) -> Vec<CocycleVec> {
    let n = group.size;
    let num_vars = n * n;
    
    // We are looking for psi(g, h) satisfying:
    // psi(g, h) + psi(gh, k) + psi(g, hk) + psi(h, k) = 0
    // for all g, h, k in G.
    
    // However, we can fix normalized cocycles: psi(1, x) = psi(x, 1) = 0.
    // This reduces variables, but for simplicity we keep all n^2 and add constraints.

    let mut equations: Vec<Vec<u8>> = Vec::new();

    // 1. Cocycle Identity Constraints
    // Note: We don't need ALL triples. O(n^3) is fine for n=12 (1728 eqs).
    for g in 0..n {
        for h in 0..n {
            for k in 0..n {
                let mut eq = vec![0u8; num_vars];
                
                // The terms in the coboundary equation:
                // psi(g, h) -> index g*n + h
                // psi(gh, k)
                // psi(h, k)
                // psi(g, hk)
                
                let gh = group.table[g][h];
                let hk = group.table[h][k];
                
                eq[g * n + h] ^= 1;
                eq[gh * n + k] ^= 1;
                eq[h * n + k] ^= 1;
                eq[g * n + hk] ^= 1;
                
                equations.push(eq);
            }
        }
    }

    // 2. Normalization Constraints (Optional but clean): psi(0, x) = 0, psi(x, 0) = 0
    // Assuming 0 is identity (which it is for our generators).
    for x in 0..n {
        let mut eq1 = vec![0u8; num_vars];
        eq1[0 * n + x] = 1; // psi(0, x)
        equations.push(eq1);

        let mut eq2 = vec![0u8; num_vars];
        eq2[x * n + 0] = 1; // psi(x, 0)
        equations.push(eq2);
    }

    // 3. Solve using Gaussian Elimination over F2
    gaussian_elimination_f2(equations, num_vars)
}

/// Simple Gaussian Elimination to find Nullspace Basis
fn gaussian_elimination_f2(mut matrix: Vec<Vec<u8>>, num_vars: usize) -> Vec<Vec<u8>> {
    // This finds the kernel (nullspace) basis.
    // Actually, the matrix above represents Ax = 0. We need the basis of solutions.
    
    // Standard RREF
    let mut pivot_row = 0;
    let num_rows = matrix.len();
    let mut pivot_cols = vec![-1isize; num_rows]; // Stores which col is pivot for a row

    for col in 0..num_vars {
        if pivot_row >= num_rows { break; }

        // Find pivot
        let mut pivot = pivot_row;
        while pivot < num_rows && matrix[pivot][col] == 0 {
            pivot += 1;
        }

        if pivot < num_rows {
            // Swap
            matrix.swap(pivot_row, pivot);
            pivot_cols[pivot_row] = col as isize;

            // Eliminate
            for r in 0..num_rows {
                if r != pivot_row && matrix[r][col] == 1 {
                    // row_r = row_r + pivot_row
                    let (p_row, other_row) = if r < pivot_row {
                        let (a, b) = matrix.split_at_mut(pivot_row);
                        (&mut a[r], &mut b[0])
                    } else {
                        let (a, b) = matrix.split_at_mut(r);
                        (&mut b[0], &mut a[pivot_row])
                    };
                    
                    for i in 0..num_vars {
                        p_row[i] ^= other_row[i];
                    }
                }
            }
            pivot_row += 1;
        }
    }

    // Extract Basis for Nullspace
    // Free variables correspond to columns without pivots.
    // For each free variable x_f, set x_f = 1, other free = 0.
    // Solve for pivot variables.
    
    let mut basis = Vec::new();
    let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter()
        .filter(|&&c| c >= 0)
        .map(|&c| c as usize)
        .collect();

    for free_col in 0..num_vars {
        if !pivot_set.contains(&free_col) {
            let mut solution = vec![0u8; num_vars];
            solution[free_col] = 1;

            // Back substitution (trivial in RREF: pivot_var = sum(free_vars))
            for r in 0..pivot_row {
                let p_col = pivot_cols[r] as usize;
                // If matrix[r][free_col] is 1, then x_p + x_f = 0 => x_p = 1
                if matrix[r][free_col] == 1 {
                    solution[p_col] = 1;
                }
            }
            basis.push(solution);
        }
    }

    basis
}

// ============================================================================
// III. THE ANALYTIC FILTER
// ============================================================================

fn solve_hadamard(group: &FiniteGroup) {
    println!(">>> Analyzing Group: {} (Order {})", group.name, group.size);
    
    let basis = compute_cocycle_basis(group);
    let k = basis.len();
    println!(">>> 2-Cocycle Space Dimension: {} (Search space 2^{})", k, k);

    let search_space = 1u64 << k;
    let n = group.size;

    let found = (0..search_space).into_par_iter().find_any(|&coeffs_mask| {
        // Construct the specific cocycle for this mask
        // psi_flat[i] is the value at index i (row * n + col)
        let mut psi_flat = vec![0u8; n * n];
        
        for (i, basis_vec) in basis.iter().enumerate() {
            if (coeffs_mask >> i) & 1 == 1 {
                for j in 0..n*n {
                    psi_flat[j] ^= basis_vec[j];
                }
            }
        }

        // Check Orthogonality
        // Row 0 (Identity) vs Row r
        // Sum_{x} psi(0, x) * psi(r, x)
        // Since we normalized psi(0, x) = 0 (mapped to +1), we just sum psi(r, x).
        // Wait! In multiplicative notation:
        // H_uv = (-1)^psi(u, v)
        // <Row u, Row v> = Sum_x (-1)^psi(u, x) * (-1)^psi(v, x)
        //                = Sum_x (-1)^(psi(u, x) + psi(v, x))
        // So we need: Sum_{x} (psi(u, x) + psi(v, x) == 1) to be n/2
        // OR equivalently: sum is 0 if n terms.
        
        // Let's verify Row 0 (id) vs Row r.
        // Row 0 is all 1s (since psi(0, x) = 0).
        // So we just need Row r to have equal number of +1 and -1.
        // i.e. Sum_{x} psi(r, x) should be n/2 (counting 1s).

        for r in 1..n {
            let mut sum_1s = 0;
            for x in 0..n {
                // To get the matrix entry H_{r, x}, we need the cocycle value psi(r, x)
                // But strictly, Cocyclic matrix is M_g = psi(g, x) * P_g?
                // Standard definition: H = [ psi(g_i, g_j) ] is NOT usually Hadamard directly
                // unless it's a specific type.
                // The standard Cocyclic Matrix is H_{gh, k} ?? 
                
                // Correction: A Cocyclic Hadamard Matrix over G is a matrix H = [ h_{ij} ]
                // where h_{ij} = psi(g_i, g_j).
                // Let's stick to this definition.
                
                // For H to be Hadamard, rows must be orthogonal.
                // <Row a, Row b> = 0
                // Sum_k psi(a, k) + psi(b, k) = n/2 (mod 2 sum) ???
                // No, Sum (-1)^(psi(a, k) + psi(b, k)) = 0.
                
                let val_a = psi_flat[0 * n + x]; // Row 0
                let val_b = psi_flat[r * n + x]; // Row r
                
                if val_a ^ val_b == 1 {
                    sum_1s += 1;
                }
            }
            
            if sum_1s != n / 2 {
                return false;
            }
        }

        // If we only check Row 0 vs others, that is sufficient IF the group action is transitive
        // and the cocycle is "central" enough. For generalized search, let's just do full check
        // to be absolutely sure.
        
        // Full Check (Safety first)
        for r1 in 0..n {
            for r2 in (r1+1)..n {
                let mut hamming_dist = 0;
                for x in 0..n {
                    let v1 = psi_flat[r1 * n + x];
                    let v2 = psi_flat[r2 * n + x];
                    if v1 != v2 { hamming_dist += 1; }
                }
                if hamming_dist != n/2 { return false; }
            }
        }
        
        true
    });

// ... inside solve_hadamard ...

    let search_space = 1u64 << k;
    let counter = AtomicU64::new(0);
    let start_search = Instant::now();

    let found = (0..search_space).into_par_iter().find_any(|&coeffs_mask| {
        // Progress tracking
        let c = counter.fetch_add(1, Ordering::Relaxed);
        if c % 10_000_000 == 0 && c > 0 {
            let elapsed = start_search.elapsed().as_secs_f64();
            let rate = c as f64 / elapsed;
            println!(
                ">>> Progress: {:.2}% | Checked: {}M | Speed: {:.2}M/s",
                (c as f64 / search_space as f64) * 100.0,
                c / 1_000_000,
                rate / 1_000_000.0
            );
        }

        // --- Same inner logic as before ---
        let mut psi_flat = vec![0u8; n * n];
        for (i, basis_vec) in basis.iter().enumerate() {
            if (coeffs_mask >> i) & 1 == 1 {
                for j in 0..n*n {
                    psi_flat[j] ^= basis_vec[j];
                }
            }
        }

        // The Hadamard check
        for r1 in 0..n {
            for r2 in (r1+1)..n {
                let mut hamming_dist = 0;
                for x in 0..n {
                    if psi_flat[r1 * n + x] != psi_flat[r2 * n + x] { 
                        hamming_dist += 1; 
                    }
                }
                if hamming_dist != n/2 { return false; }
            }
        }
        true
    });

    match found {
        Some(mask) => println!(">>> SUCCESS: Solution found with mask {:b}", mask),
        None => println!(">>> FAILURE: No solution in this group."),
    }
}

fn main() {
    let start = Instant::now();
    println!(">>> PHASE: THE STRESS TEST (n=32)");
    println!(">>> TARGET: Dihedral Group D_16");
    solve_hadamard(&FiniteGroup::dihedral(16));
    println!("\nTotal execution time: {:.2?}", start.elapsed());
}