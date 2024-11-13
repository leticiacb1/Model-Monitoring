import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# -------------------------------
#          GENERATE DATA
# -------------------------------

# Generate data from two distributions
np.random.seed(1234)
data_train = np.random.normal(loc=0, scale=1, size=1000)
data_prod = np.random.normal(loc=0.02, scale=1.01, size=1000)

# Rejects H0 data : 
# data_train = np.random.normal(loc=0, scale=1, size=1000)
# data_prod = np.random.normal(loc=0.5, scale=1.2, size=1000)

# -------------------------------
#          COMPARE DATA
# -------------------------------

# Plot histograms of the two distributions
plt.hist(data_train, bins=30, alpha=0.5, label="Data train")
plt.hist(data_prod, bins=30, alpha=0.5, label="Data production")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title("Histogram of Data train and Data production")
plt.legend()
plt.show()

# -------------------------------
#          CHI-SQUARE TEST
# -------------------------------

# Perform the Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(data_train, data_prod)

alpha = 0.05

# Print the KS test results
print(f"KS statistic: {ks_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Check if there is a significant difference
if p_value < alpha:
    print(
        " [Reject H0] Data drift detected: There is a significant difference in the continuos variable distribution."
    )
else:
    print(
        " [Fail to reject H0] No data drift detected: There is no significant difference in the continuos variable distribution."
    )