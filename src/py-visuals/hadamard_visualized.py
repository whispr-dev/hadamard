import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.special import comb

# m = 35. Number of -1s (k) determines the row sum.
# Row sum = m - 2k.
m = 35
k_values = np.arange(0, m + 1)
row_sums = m - 2 * k_values
# Number of ways to choose k positions out of m (simplified for symmetric sequences)
# For symmetric length 35, we have 17 degrees of freedom.
counts = [comb(17, i) for i in range(18)]
sums_filtered = [35 - 4*i for i in range(18)]

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plotting the distribution of possible row sums
plt.bar(sums_filtered, counts, color='lightgray', label='All Symmetric Sequences')

# Highlighting the "Golden Filter" values for n=140
# s^2 must be in {1, 9, 25, 49, 81, 121}
golden_sums = [1, 3, 5, 7, 9, 11]
golden_counts = [counts[(35-s)//4] if (35-s)%4==0 else 0 for s in golden_sums]
plt.bar(golden_sums, golden_counts, color='crimson', label='Power Sum Filter Pass')

plt.title(f"The Williamson Filter Landscape (n=140, m=35)", fontsize=15)
plt.xlabel("Row Sum (s)", fontsize=12)
plt.ylabel("Frequency (Log Scale)", fontsize=12)
plt.yscale('log')
plt.legend()
plt.show()