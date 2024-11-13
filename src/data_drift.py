import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -----------------------------------
#      REBALANCE and PREPROCESS
# -----------------------------------

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the minority class to match the majority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])

def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    # shuffle and return
    return data_bal.sample(frac=1, random_state=42)


# -----------------------------------
#               MAIN
# -----------------------------------

if __name__ == '__main__':

    # ------ Open dataset  ------
    df = pd.read_csv("data/Churn_Modelling.csv")
    df_bal = preprocess(df)
    print(df_bal.shape)
    df_bal.head()

    # ------ Randomly distribute the dataset rows between the sets set 0 and set 1 ------
    # Create column Set
    df_bal["Set"] = random.choices([0, 1], k=len(df_bal))
    df_bal.head()

    # ------ Separate data  ------
    X = df_bal.drop("Set", axis=1)
    y = df_bal["Set"]

    # ------ Train and Test data  ------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)


    # ------ Prepare data ------
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]

    col_transf = make_column_transformer(
        (StandardScaler(), num_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # ------ Train model ------
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Accuracy
    y_pred = clf.predict(X_test)
    print(f"Accuracy score: {accuracy_score(y_test, y_pred):.2f}")

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=clf.classes_
    )
    conf_mat_disp.plot()