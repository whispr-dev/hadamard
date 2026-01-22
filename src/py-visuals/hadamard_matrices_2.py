import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to build Sylvester Hadamard matrices (powers of 2)
def sylvester_hadamard(order):
    if order == 1:
        return np.array([[1]])
    if order == 2:
        return np.array([[1, 1], [1, -1]])
    smaller = sylvester_hadamard(order // 2)
    top = np.hstack((smaller, smaller))
    bottom = np.hstack((smaller, -smaller))
    return np.vstack((top, bottom))

# Pretty plotting function
def plot_hadamard(matrix, title, filename):
    plt.figure(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, cmap='coolwarm_r', center=0,
                cbar=False, linewidths=1, linecolor='gray',
                square=True, xticklabels=False, yticklabels=False,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()  # Comment this out if running headless

# Generate and plot some beauties!
orders = [4, 8, 16]
for n in orders:
    H = sylvester_hadamard(n)
    plot_hadamard(H, f'Hadamard Matrix Order {n} (Sylvester Construction)', f'hadamard_{n}.png')

print("All done, fren! Check the PNGs – they're gorgeous fractaly patterns 😍")