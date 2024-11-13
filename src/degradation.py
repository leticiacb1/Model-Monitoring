import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -------------------------------
#          GENERATE DATA
# -------------------------------

# Generate a simulated dataset
X, y = make_classification(
    n_samples=300*100,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

start_date = pd.to_datetime("2023-01-01")
num_days = X.shape[0] // 50  # Number of days based on batch size
dates = pd.date_range(start=start_date, periods=num_days, freq="D").repeat(100)

# Create a DataFrame from the simulated data and the dates column
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 11)])
df["Label"] = y
df["Date"] = dates[:X.shape[0]]

df.head()

# -------------------------------
#             MODEL
# -------------------------------

# Split in Train and Test set using date as threshold

X = df.drop(["Label", "Date"], axis=1)
y = df["Label"]

filter_train = df["Date"] < "2023-10-01"

X_train = X.loc[filter_train, :]
X_test = X.loc[~filter_train, :]

y_train = y.loc[filter_train]
y_test = y.loc[~filter_train]
date_test = df.loc[~filter_train, "Date"]

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ----- WITHOUT DEGRADATION -----

# # Accuracy
# y_pred = clf.predict(X_test)
# print(f"Accuracy score: {accuracy_score(y_test, y_pred):.2f}")

# # Performance overtime
# df_pred = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "date": date_test})
# accuracy_by_date = df_pred.groupby("date").apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
# accuracy_by_date = accuracy_by_date.reset_index()
# accuracy_by_date.columns = ["date", "accuracy"]


# fig = px.bar(accuracy_by_date, x="date", y="accuracy", title="Model Performance Over Time")
# fig.show()

# ----- WITH DEGRADATION -----

# Apply noise
noise_magnitude = np.arange(1, len(X_test) + 1)/len(X_test) * 7.5
np.random.seed(1234)
noise = np.random.normal(0, noise_magnitude[:, np.newaxis], size=X_test.shape)
X_test_noise = X_test + noise

# Performance overtime
y_pred_noise = clf.predict(X_test_noise)
print(f"Accuracy score: {accuracy_score(y_test, y_pred_noise):.2f}")

# Performance overtime
df_pred_noise = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_noise, "date": date_test})
accuracy_by_date_noise = df_pred_noise.groupby("date").apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
accuracy_by_date_noise = accuracy_by_date_noise.reset_index()
accuracy_by_date_noise.columns = ["date", "accuracy"]
accuracy_by_date_noise

fig = px.bar(accuracy_by_date_noise, x="date", y="accuracy", color="accuracy", title="Performance Decrease Over Time")
fig.show()