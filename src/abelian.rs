// main.rs
use rayon::prelude::*;
use itertools::Itertools;
use std::time::Instant;

// ============================================================================
// I. THE ALGEBRAIC OBJECTS (The Source Category)
// ============================================================================

/// Represents a finite abelian group G = Z_d1 x Z_d2 x ... x Z_dk
#[derive(Clone, Debug)]
struct AbelianGroup {
    dims: Vec<usize>,
    size: usize,
    elements: Vec<Vec<usize>>, // Pre-computed elements for speed
}

impl AbelianGroup {
    fn new(dims: Vec<usize>) -> Self {
        let size = dims.iter().product();
        
        // Generate all elements via cartesian product
        let mut elements = vec![vec![]];
        for &d in &dims {
            let mut new_elements = Vec::with_capacity(elements.len() * d);
            for e in elements {
                for i in 0..d {
                    let mut new_e = e.clone();
                    new_e.push(i);
                    new_elements.push(new_e);
                }
            }
            elements = new_elements;
        }

        AbelianGroup { dims, size, elements }
    }

    /// Group operation: a + b (component-wise modulo dims)
    #[inline(always)]
    fn add(&self, a: &[usize], b: &[usize]) -> Vec<usize> {
        a.iter()
            .zip(b.iter())
            .zip(self.dims.iter())
            .map(|((x, y), d)| (x + y) % d)
            .collect()
    }

    /// The index of an element in the canonical ordering (lexicographic)
    fn index_of(&self, element: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        // Calculate index in reverse (standard strides)
        for i in (0..self.dims.len()).rev() {
            idx += element[i] * multiplier;
            multiplier *= self.dims[i];
        }
        idx
    }
}

// ============================================================================
// II. THE COHOMOLOGICAL FUNCTOR (The Transport)
// ============================================================================

// We define a Cocycle as a map G x G -> {0, 1} (additive notation for F_2)
// This maps to {-1, 1} via x -> (-1)^x
type CocycleFn = Box<dyn Fn(&[usize], &[usize]) -> u8 + Sync + Send>;

/// Generates a basis for 2-cocycles on the group.
/// For Abelian groups, we focus on Bilinear Forms as the primary source of non-triviality.
struct CohomologyBasis {
    basis_functions: Vec<CocycleFn>,
}

impl CohomologyBasis {
    fn generate(group: &AbelianGroup) -> Self {
        let mut basis: Vec<CocycleFn> = Vec::new();
        let dim_len = group.dims.len();

        // 1. Bilinear Forms: x^T A y
        // For each pair of dimensions (i, j), we create a map.
        // This covers the "interactions" between cyclic components.
        for i in 0..dim_len {
            for j in 0..dim_len {
                let d_i = group.dims[i];
                let d_j = group.dims[j];
                
                // Only relevant if gcd(d_i, d_j) > 1, but we iterate all for completeness 
                // in the "search space" (redundancy is filtered by the solver usually, 
                // but here we just generate a rich set).
                
                let bi_map = move |u: &[usize], v: &[usize]| -> u8 {
                    // Standard bilinear form component: (u_i * v_j) mod 2
                    // We map to {0, 1} strictly.
                    ((u[i] * v[j]) % 2) as u8
                };
                basis.push(Box::new(bi_map));
            }
        }

        // NOTE: A complete search would also include coboundaries (partial derivatives of 1-chains)
        // to ensure we hit every equivalent matrix, but for finding *existence*, 
        // the bilinear forms are the "high value" targets in the cohomology class.
        
        Self { basis_functions: basis }
    }
}

// ============================================================================
// III. THE ANALYTIC FILTER (The Solver)
// ============================================================================

fn solve_hadamard(group: &AbelianGroup) {
    println!(">>> Categorical Object: Abelian Group {:?}", group.dims);
    println!(">>> Order n = {}", group.size);

    if group.size % 4 != 0 && group.size > 2 {
        println!("!!! OBSTRUCTION DETECTED: n not divisible by 4. Aborting (Theorems hold).");
        return;
    }

    let cohomology = CohomologyBasis::generate(group);
    let basis_size = cohomology.basis_functions.len();
    println!(">>> Cohomology Basis Size: {} (Search space 2^{})", basis_size, basis_size);

    // Iterating through all linear combinations of the basis
    // This represents iterating through the cohomology group H^2(G, Z_2) restricted to bilinear forms.
    let search_space = 1u64 << basis_size;
    
    // We use Rayon to parallelize the search over the coefficients
    let found = (0..search_space).into_par_iter().find_any(|&coeffs_mask| {
        // 1. Construct the specific cocycle psi for this combination
        let psi = |u: &[usize], v: &[usize]| -> i8 {
            let mut sum_mod_2 = 0;
            for (i, func) in cohomology.basis_functions.iter().enumerate() {
                if (coeffs_mask >> i) & 1 == 1 {
                    sum_mod_2 ^= func(u, v);
                }
            }
            if sum_mod_2 == 0 { 1 } else { -1 }
        };

        // 2. The Hadamard Check (Orthogonality)
        // Check inner product of Row 0 (Unit) with every other Row k.
        // Due to group homogeneity, if Row 0 is orthogonal to all others, 
        // and the matrix is a group matrix, the whole matrix is Hadamard.
        
        // Row 0 corresponds to element 0 (identity).
        // Row k corresponds to element k.
        // Inner prod: Sum_{x in G} psi(0, x) * psi(k, x)
        // Note: psi(0, x) is usually 1 for normalized cocycles. Let's assume generic.
        
        let identity = vec![0; group.dims.len()];
        
        // We only need to check if row 'identity' is orthogonal to every other row 'a'
        for i in 1..group.size {
            let a = &group.elements[i];
            
            let mut dot_product = 0;
            
            for x in &group.elements {
                // To check orthogonality of Row(0) and Row(a):
                // We sum: M_{0,x} * M_{a,x}
                // M_{u,v} = psi(u, v)
                
                let val_0 = psi(&identity, x);
                let val_a = psi(a, x);
                
                dot_product += val_0 * val_a;
            }
            
            if dot_product != 0 {
                return false; // Not orthogonal, fail early
            }
        }
        
        true // Passed all orthogonality checks
    });

    match found {
        Some(mask) => {
            println!("\n>>> SUCCESS: Hadamard Matrix Found!");
            println!(">>> Basis Mask: {:b}", mask);
            println!(">>> The obstruction vanishes in this cohomology class.");
        },
        None => {
            println!("\n>>> FAILURE: No Cocyclic Hadamard Matrix found in this basis.");
            println!(">>> Try a different group extension or wider basis.");
        }
    }
}

fn main() {
    let start = Instant::now();
    
    // TEST CASE 1: n = 4 (Trivial, Z2 x Z2)
    // solve_hadamard(&AbelianGroup::new(vec![2, 2]));

    // TEST CASE 2: n = 8 (Z2 x Z2 x Z2)
    // solve_hadamard(&AbelianGroup::new(vec![2, 2, 2]));

    // TEST CASE 3: n = 16 (Z2 x Z2 x Z2 x Z2) - Standard
    // solve_hadamard(&AbelianGroup::new(vec![2, 2, 2, 2]));
    
    // TEST CASE 4: n = 12 (Z2 x Z6) -> Expect Failure (No cocyclic hadamard for Z_2 x Z_6 known?)
    // Actually n=12 is the smallest index where cyclic construction fails, 
    // but Dicyclic group (non-abelian) works.
    // Let's try Z_2 x Z_2 x Z_3
    solve_hadamard(&AbelianGroup::new(vec![2, 2, 3]));

    println!("\nComputation finished in {:.2?}", start.elapsed());
}