import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant
import json
import os

def create_williamson_hadamard(a, b, c, d):
    A, B, C, D = circulant(a), circulant(b), circulant(c), circulant(d)
    return np.vstack([
        np.hstack([A, B, C, D]),
        np.hstack([-B, A, -D, C]),
        np.hstack([-C, D, A, -B]),
        np.hstack([-D, -C, B, A])
    ])

# 1. Load the data from the file Rust created
def plot_smart():
    if not os.path.exists('result.json'):
        print(">>> No result.json found!")
        return

    with open('result.json', 'r') as f:
        data = json.load(f)
    
    # TRUTH DERIVATION: Ignore data['n'], look at the actual list
    a_seq = data['a']
    m = len(a_seq)  # This will correctly be 23
    n = 4 * m       # This will correctly be 92
    
    print(f">>> Detected Truth: m={m}, n={n}")
    
    # Construct blocks
    H = create_williamson_hadamard(a_seq, data['b'], data['c'], data['d'])
    
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.imshow(H, cmap='RdGy')
    plt.title(f"Hadamard Matrix (Inferred Order n={n})")
    plt.axis('off')
    
    # Dynamic Grid Sizing
    for i in range(1, 4):
        plt.axhline(y=i*m - 0.5, color='blue', lw=1.5, alpha=0.6)
        plt.axvline(x=i*m - 0.5, color='blue', lw=1.5, alpha=0.6)

    plt.savefig(f"hadamard_{n}_fixed.png", dpi=300)
    print("Done! Check hadamard_feature.png")
    plt.show()

    else:
    print("Run Rust first to create result.json!")