import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant

def create_williamson_hadamard(a, b, c, d):
    """
    Constructs a Williamson-type Hadamard matrix from four sequences.
    H = [[ A,  B,  C,  D],
         [-B,  A, -D,  C],
         [-C,  D,  A, -B],
         [-D, -C,  B,  A]]
    """
    # Create circulant blocks
    A = circulant(a)
    B = circulant(b)
    C = circulant(c)
    D = circulant(d)
    
    # Assemble the block matrix
    top = np.hstack([A, B, C, D])
    mid1 = np.hstack([-B, A, -D, C])
    mid2 = np.hstack([-C, D, A, -B])
    bot = np.hstack([-D, -C, B, A])
    
    return np.vstack([top, mid1, mid2, bot])

def plot_hadamard(matrix, n, filename="hadamard_heatmap.png"):
    plt.figure(figsize=(12, 12))
    
    # Using a high-contrast 'binary' or 'RdBu' map to show +/- 1
    plt.imshow(matrix, cmap='RdGy', interpolation='nearest')
    
    plt.title(f"Williamson Hadamard Matrix (Order n={n})", fontsize=20, pad=20)
    plt.axis('off')
    
    # Adding a subtle grid to show the block structure
    m = n // 4
    for i in range(1, 4):
        plt.axhline(y=i*m - 0.5, color='blue', linewidth=2, alpha=0.5)
        plt.axvline(x=i*m - 0.5, color='blue', linewidth=2, alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f">>> High-resolution heatmap saved to {filename}")
    plt.show()

# --- INPUT DATA FROM RUST SUCCESS ($n=44$) ---
# Paste your results here:
a_seq = [1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1]
b_seq = [1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1]
c_seq = [1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
d_seq = [1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1]

n = len(a_seq) * 4
H = create_williamson_hadamard(a_seq, b_seq, c_seq, d_seq)

# Verify orthogonality: H * H.T should be n * Identity
identity_check = np.dot(H, H.T)
if np.allclose(identity_check, np.eye(n) * n):
    print(f">>> Verification SUCCESS: Matrix is Hadamard (n={n})")
    plot_hadamard(H, n)
else:
    print(">>> Verification FAILED: Matrix is not orthogonal.")