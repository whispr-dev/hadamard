[README.md]

<p align="center">
  <a href="https://github.com/whispr-dev/co_hadamard">
    <img src="https://img.shields.io/github/stars/whispr-dev/co_hadamard?style=for-the-badge" alt="GitHub stars" />
  </a>
  <a href="https://github.com/whispr-dev/co_hadamard/issues">
    <img src="https://img.shields.io/github/issues/whispr-dev/co_hadamard?style=for-the-badge" alt="GitHub issues" />
  </a>
  <a href="https://github.com/whispr-dev/co_hadamard/fork">
    <img src="https://img.shields.io/github/forks/whispr-dev/co_hadamard?style=for-the-badge" alt="GitHub forks" />
  </a>
</p>

---


# Toward Homological Efficiency: An Architecture for Hadamard Discovery
### "Dedicated to the pursuit of absolute balance in the $n=140$ abyss."

## Abstract
This project implements a multi-tiered computational framework for the discovery of Hadamard matrices. By treating the search as a mapping problem between Group Cohomology, Williamson Transports, and Diophantine Filters, we significantly mitigate search complexity.
We report the rapid synthesis of the historic $n=92$ Hall-Baumert order in 1.40s.

## Project Architecture
The engine is built on three fundamental pillars:
- Cocyclic Framework: Solving for 2-cocycles $\psi: G \times G \to \mathbb{Z}_2$ to reduce search space from $2^{n^2}$ to $2^k$.
- Williamson Transport: Mapping the problem onto four circulant matrices ($A, B, C, D$) and evaluating via the Periodic Autocorrelation Function (PAF).
- Arithmetic Pruning: An "Intelligence Layer" utilizing the Sum of Four Squares Theorem ($s_A^2 + s_B^2 + s_C^2 + s_D^2 = 4n$) to discard invalid candidates before expensive evaluation.

## Repository Structure
Based on the:
```Plaintext
├── Cargo.toml                # Rust dependencies (Rayon, Serde, etc.)
├── Toward_Homological_...pdf  # Final Academic Whitepaper
├── docs/                     # Research and Explainers
│   ├── Grok4-esplains-full.md # Symbolic derivation of Hadamard magic
│   └── WHITEPAPER.md         # Source markdown for the formal paper
├── src/                      # High-performance search engine (Rust)
│   ├── williamson92.rs       # n=92 Hall-Baumert solver
│   ├── williamson44.rs       # n=44 Optimized solver
│   └── n=140.rs              # The n=140 search manifold
└── src/py-visuals/           # Visualization layer (Python)
    ├── visualizer.py         # Heatmap generation
    └── result.json           # Discovered sequence data
```

## Performance Benchmarks
Our engine demonstrates that algebraic selection outpaces brute force:
Order (n)Primary SymmetrySearch TimeManifold Reduction32$H^2(D_{16}, \mathbb{Z}_2)$7.38s$2^{1024} \to 2^{32}$92Williamson MITM1.40s$O(N^4) \to O(N^2)$44Williamson + Filter2.44ms90% initial prune

## Getting Started
### Prerequisites
Rust: cargo 1.70+
Python: matplotlib, numpy, scipy (for visualization)

### 1. Run the Search EngineNavigate to the source directory and execute the $n=92$ solver:
```Bash
cargo run --bin williamson92
```

 This will output a `result.json` containing the discovered sequences.

### 2. Visualize Results
Render the success as a high-resolution heatmap:Bashpython src/py-visuals/visualizer.py

The $n=140$ Event Horizon
At $n=140$, the search manifold ($N \approx 1.1 \times 10^5$) exceeds the memory limits for standard $O(N^2)$ Hash Maps on consumer-grade hardware (64GB RAM).Future work focuses on:GPU Kernels: Moving the "Square-Sum Sieve" to CUDA architectures.

Partitioned Sector Search: Further categorical reduction of the $n=140$ manifold.

Citation
If you use this architecture in your research, please cite the included whitepaper:
whisprer & Google GeminiPro3.0. (2025). Toward Homological Efficiency: An Architecture for Hadamard Discovery.

---

