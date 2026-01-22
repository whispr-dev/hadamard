# Contributing to Hadamard Discovery

Thank you for helping us push toward the $n=140$ abyss!

## Code Style
- **Rust:** Follow Rust 2021 conventions. Run `cargo fmt` and `cargo clippy -- -D warnings`.
- **Python:** Use standard PEP 8 formatting for visualization scripts.
- **Math:** New constructions must include a symbolic derivation (LaTeX) in the `docs/` folder.

## Development Workflow
1. **Identify a Manifold:** Pick a target order $n$ (e.g., $n=140, 156$).
2. **Implement the Filter:** Add new Diophantine or Cocycle filters to `src/`.
3. **Verify Orthogonality:** All discovered sequences must pass the `HH^T = nI` check.
4. **Submit PR:** Open a Pull Request with your benchmarks and a sample `.json` output.

## Documentation
If you implement a new group-theoretic approach, please update the whitepaper or include a new explainer in `docs/`.