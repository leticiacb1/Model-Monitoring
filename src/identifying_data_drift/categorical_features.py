import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# -------------------------------
#          GENERATE DATA
# -------------------------------

categories = ['Single', 'Married', 'Divorced']

# Generate two example datasets
dataset_train = np.random.choice(categories, size=1000, p=[0.49, 0.3, 0.21])
dataset_prod = np.random.choice(categories, size=1000, p=[0.50, 0.3, 0.2])

# Rejects H0 data : 
# dataset_train = np.random.choice(categories, size=1000, p=[0.4, 0.3, 0.3])
# dataset_prod = np.random.choice(categories, size=1000, p=[0.5, 0.3, 0.2])

# Print example of generated data
print(dataset_train)

# -------------------------------
#          COMPARE DATA
# -------------------------------

# Calc frequencies
counts_train = [np.sum(dataset_train == category) for category in categories]
counts_prod = [np.sum(dataset_prod == category) for category in categories]

# Plot graph
plt.figure(figsize=(8, 6))

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width / 2, counts_train, width, label="Dataset Train")
plt.bar(x + width / 2, counts_prod, width, label="Dataset Production")

plt.xlabel("Categories")
plt.ylabel("Frequency")
plt.title("Comparison of Dataset Train and Dataset Production")


plt.xticks(x, categories)
plt.legend()
plt.show()

# -------------------------------
#          CHI-SQUARE TEST
# -------------------------------

# Create contingency table from the datasets
observed = np.array(
    [
        [
            np.sum(dataset_train == "Single"),
            np.sum(dataset_train == "Married"),
            np.sum(dataset_train == "Divorced"),
        ],
        [
            np.sum(dataset_prod == "Single"),
            np.sum(dataset_prod == "Married"),
            np.sum(dataset_prod == "Divorced"),
        ],
    ]
)

# Perform chi-squared test
chi2, p_value, _, _ = chi2_contingency(observed)

# Set significance level
alpha = 0.05

print(f"P-value: {p_value:.4f}")

# Check if there is a significant difference
if p_value < alpha:
    print(
        " [Reject H0] Data drift detected: There is a significant difference in the categorical variable distribution."
    )
else:
    print(
        " [Fail to reject H0] No data drift detected: There is no significant difference in the categorical variable distribution."
    )