import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -------------------------------
#          GENERATE DATA
# -------------------------------

# [NO data drift] Set distributions equally 
avg_january = 0.0
std_january = 1.0
avg_april = 0.0
std_april = 1.0

# [WITH data drift]
# avg_january = 0.0
# std_january = 1.0
# avg_april = 0.75
# std_april = 1.2

# Generate data from two distributions
np.random.seed(1234)
data_jan = np.random.normal(loc=avg_january, scale=std_january, size=1000)
data_apr = np.random.normal(loc=avg_april, scale=std_april, size=1000)


# Transform in dataframe
df_jan = pd.DataFrame({"feature": data_jan, "set": [0] * len(data_jan)})
df_jan.head()

df_apr = pd.DataFrame({"feature": data_apr, "set": [1] * len(data_jan)})
df_apr.head()

df_sim = pd.concat([df_jan, df_apr], axis=0)
df_sim

# -------------------------------
#          COMPARE DATA
# -------------------------------

# Plot histograms of the two distributions
plt.hist(data_jan, bins=30, alpha=0.5, label="Data January")
plt.hist(data_apr, bins=30, alpha=0.5, label="Data April")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title("Histogram of Data train and Data production")
plt.legend()
plt.savefig("ks_test_ex.png")
plt.show()

# -------------------------------
#            MOODEL
# -------------------------------


# Separate data
X = df_sim.drop("set", axis=1)
y = df_sim["set"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1912
)

# Train Model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)


# Accuracy
y_pred = clf.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_pred):.2f}")


# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred, labels=clf.classes_)
conf_mat_disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat, display_labels=clf.classes_
)
conf_mat_disp.plot()