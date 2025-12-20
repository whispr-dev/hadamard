# Security Policy

## Supported Versions
| Version | Supported |
|---------|-----------|
| 1.1.x   | ✅ Active  |

## Reporting a Vulnerability
Hadamard matrices are used in cryptography and signal processing. If you find a flaw in the **orthogonality verification** logic or a security issue in the Rust/Python pipeline:
1. Do not open a public issue.
2. Email the maintainers at: `abstractor@hadamard.dev`

## Security Scope
This project focuses on the **computational integrity** of the matrices generated. We ensure that all "Success" outputs are verified against the fundamental axiom $H^T H = nI$.