import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Colors: +1 = blue, -1 = orange, 0 = gray
cmap = ListedColormap(['#FF6B6B', '#4ECDC4', '#95A5A6'])  # -1, +1, gray

def plot_hadamard_matrix(H, title, ax):
    n = H.shape[0]
    ax.imshow(H, cmap=cmap, interpolation='none')
    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels([]), ax.set_yticklabels([])

    # Add +1 and -1 labels
    for i in range(n):
        for j in range(n):
            val = H[i,j]
            ax.text(j, i, f'{val:+}', ha='center', va='center',
                    color='white' if abs(val) == 1 else 'black', fontsize=14)

# 1. Smallest Hadamard matrices
H2 = np.array([[ 1,  1],
               [ 1, -1]])
H4 = np.array([[ 1,  1,  1,  1],
               [ 1, -1,  1, -1],
               [ 1,  1, -1, -1],
               [ 1, -1, -1,  1]])

# 2. 2x2 and 4x4 as blocks (silhouette of the conjecture)
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
plot_hadamard_matrix(H2, "Order 2: The smallest non-trivial one", axs[0])
plot_hadamard_matrix(H4, "Order 4: Paley matrix", axs[1])
axs[2].axis('off')
axs[2].text(0.5, 0.7, "Hadamard Conjecture:\n\nIf n = 1, 2 or n ≡ 0 mod 4,\nthere exists an n×n Hadamard matrix\nof +1 and -1.\n\nWe know it for n up to 668,\nbut not yet proven for all n.", 
            ha='center', va='center', fontsize=16, fontweight='bold')
plt.suptitle("Known Small Hadamard Matrices", fontsize=20, y=1.05)
plt.tight_layout()
plt.show()

# 3. Why orthogonality matters - row dot products
rows = [H4[0], H4[1], H4[2]]
row_names = ["Row 0", "Row 1", "Row 2"]

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        if i == j:
            dot = np.dot(rows[i], rows[j])  # should be 4
            ax.text(0.5, 0.5, f"Dot product = {dot}", fontsize=18, ha='center', va='center')
            ax.axis('off')
        else:
            # Show the two rows side by side
            combined = np.vstack([rows[i], rows[j]])
            ax.imshow(combined, cmap=cmap, aspect='auto')
            ax.set_xticks(np.arange(-0.5, 4, 1))
            ax.grid(which='major', color='black', linewidth=2)
            ax.set_xticklabels([]), ax.set_yticklabels([])
            ax.set_title(f"{row_names[i]} • {row_names[j]}", fontsize=14)
            ax.text(0.5, -0.2, f"Dot = {np.dot(rows[i], rows[j])}", 
                    ha='center', va='center', fontsize=18, color='red')
plt.suptitle("Why rows must be orthogonal: their dot product is either n (same row) or 0", fontsize=20)
plt.tight_layout()
plt.show()

# 4. Visual analogy: balanced +1/-1 entries
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')
ax.set_aspect('equal')

# Draw a circle and points
theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
x = np.cos(theta)
y = np.sin(theta)
colors = ['#4ECDC4' if v == 1 else '#FF6B6B' for v in [1, -1, 1, -1]]

for i in range(4):
    ax.scatter(x[i], y[i], s=800, c=colors[i], edgecolors='black', linewidth=3)
    ax.text(x[i], y[i], f'{colors[i]}', ha='center', va='center', fontsize=20, color='white')

ax.text(0, 1.8, "Think of each row as a balanced set of +1 and -1\nthat are perfectly perpendicular to all other rows", fontsize=14)
plt.title("Intuition: Like 4 perfectly balanced, perpendicular vectors", fontsize=16)
plt.show()