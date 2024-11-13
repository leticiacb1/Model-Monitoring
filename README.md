##  🧐 Monitoring

* **Resource level monitoring** : guarantee that the model is running correctly in the production environment. Monitor the CPU, RAM, storage and verify that requests are being processed at the expected rate and with no errors.

* **Performance level monitoring** : monitor the  model performance to verify if it keep its relevance over time.

> Model performance is a crucial aspect of ML that determines the effectiveness of the model in making predictions.
> Evaluating model performance involves the use of various metrics and techniques to assess how well the model generalizes to unseen data.
> 
> **Examples of metrics** 
> 
>  * Classification Problems : Accuracy, Precision, Recall, F1 Score,  AUC-ROC ...
>  * Regression Problems : MAE, MSE, RMSE, R-squared ...


### Evaluate Model Degradation

##### Ground Truth Evaluation

Method used to measure the accuracy of machine learning models by comparing their predictions to a set of pre-labeled, "ground truth" data. This ground truth is typically a high-quality, trusted dataset that represents the "true" answer or desired outcome, collected through careful manual labeling, expert input, or other reliable sources. 

The purpose of ground truth evaluation is to provide a benchmark for model performance, often using metrics that quantify how closely the model's predictions match this known data.

> If there's a large disparity between these initial metrics and the current performance—meaning that the model's accuracy or other metrics have dropped below a certain threshold—it can indicate that the model no longer reflects the current data patterns
>

The **maturation time** of the target variable is the time it takes to obtain the "true" outcome for each prediction. When this time is very long, conducting ground truth evaluation becomes impractical because it delays the availability of actual results.

For example:

* In a medical diagnosis model predicting long-term outcomes, it may take months or years to confirm if the prediction was correct.

##### Data Drift (Input drift detection)

Unlike "Ground Truth Evaluation" which relies only on true labels, data drift focuses on identifying changes in the input data itself, without requiring explicit knowledge of the true results.

> **Data drift** occurs when the distribution of the input data changes over time.

### 📌 Dependencies

Create a `venv` and install dependencies:

```bash
    # Create environment
    $ python3 -m venv venv  

    # Activate environment
    $ source venv/bin/activate

    # Install dependencies
    $ pip install -r requirements.txt
``` 

Configure the secrets in your repository : go to the repository site on `github / settings / Secrets and variables / Actions` and add a **new repository secrets**.

Set all the secrests :

* `AWS_ACCESS_KEY_ID`
  
* `AWS_SECRET_ACCESS_KEY`
  
* `AWS_REGION`
  
* `AWS_LAMBDA_ROLE_ARN`

Also create a `.env` file with the following:

```bash
    # .env content'
    AWS_ACCESS_KEY_ID="XXXXXXXXXXXXXX"
    AWS_SECRET_ACCESS_KEY="aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    AWS_REGION="xx-xxxx-2"
    AWS_LAMBDA_ROLE_ARN="arn:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
``` 

Configure your AWS credentials:

```bash
    aws configure
```

### ❓️ How to use the project

#### Statistics : Identifying Data Drifts

##### Categorical Features

Simulate a categorical variable used by a model and check whether the variable has the same distribution in the production data when compared to the training data.

[Chi-square](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223) is a statistical test that is used to determine if there is a significant association between two categorical variables:

* Null Hypothesis (**H0**): The observed frequencies in each category follow the expected theoretical distribution.

* Alternate Hypothesis (**H1**): The observed frequencies in at least one category do not follow the expected theoretical distribution.

```bash
  # Folder src/identifying_data_drift/
  $ python3 categorical_features.py
```

For **Reject H0** output change this lines :

```python
  # --- Ommited code ---

  # Generate two example datasets 
  # dataset_train = np.random.choice(categories, size=1000, p=[0.49, 0.3, 0.21])
  # dataset_prod = np.random.choice(categories, size=1000, p=[0.50, 0.3, 0.2])

  # Rejects H0 data : 
  dataset_train = np.random.choice(categories, size=1000, p=[0.4, 0.3, 0.3])
  dataset_prod = np.random.choice(categories, size=1000, p=[0.5, 0.3, 0.2])

  # --- Ommited code ---
```


##### Continuos Features

[Kolmogorov-Smirnov](https://towardsdatascience.com/evaluating-classification-models-with-kolmogorov-smirnov-ks-test-e211025f5573) test (KS test) is a 
statistical test used to assess the equality between two one-dimensional probability distributions.

* Null Hypothesis (**H0**): The two samples come from the same distribution.

* Alternate Hypothesis (**H1**):  The two samples do not come from the same distribution.


```bash
  # Folder src/identifying_data_drift/
  $ python3 continuos_features.py
```

For **Reject H0** output change this lines :

```python
  # --- Ommited code ---
  
  # Generate data from two distributions
  np.random.seed(1234)
  # data_train = np.random.normal(loc=0, scale=1, size=1000)
  # data_prod = np.random.normal(loc=0.02, scale=1.01, size=1000)

  # Rejects H0 data : 
  data_train = np.random.normal(loc=0, scale=1, size=1000)
  data_prod = np.random.normal(loc=0.5, scale=1.2, size=1000)

  # --- Ommited code ---
```

> Monitoring model performance degradation is an ongoing process.It is advisable to monitor metrics in real-time and to implement automated alerts to notify when issues arise.

#### Data Drift Detection in Machine Learning

* Download the dataset used [here](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data) and paste it in  `data/` folder.

##### Experiment idea

Randomly select the rows from this dataset and separate them into two new sets: **set 0** and **set 1**.
   
Compring the distributions of fures from **set 0** and **set 1** should be no significant differences (except some random effect), as the data were extracted randomly from the original database. (See more about [CLT](https://www.geeksforgeeks.org/central-limit-theorem-in-machine-learning/))

>If a classification model is built using Set variables (column that will indicate if that is from *set 0** or **set 1** ) as targetand other columns as features the expected accruracy of the model is approximately 50%.

To test this experiment run:

```bash
  # Folder src/
  $ python3 data_drift.py
```

##### Real Application

Suppose a data set captured at different points in time (*Set January** and **Set April**).
Creating a model to predict whether a data was predicted in January or April, what is expected from the model accuracy if:

* No data deviation: Approximately 50% (data with the same distribution)
* Have data deviation: More than 50% (the model begins to have evidence of the distribution of characteristics, being able to predict the origin of the lines with more certainty)

> In this case, the accuracy increase indicates the presence of data drift.


To simulate this run:

```bash
  # Folder src/
  $ python3 data_drift_sim.py
```

For **data drift** data  change this lines :

```python
  # --- Ommited code ---
  
  # [NO data drift] Set distributions equally 
  # avg_january = 0.0
  # std_january = 1.0
  # avg_april = 0.0
  # std_april = 1.0
  # 
  # [WITH data drift]
  avg_january = 0.0
  std_january = 1.0
  avg_april = 0.75
  std_april = 1.2

  # --- Ommited code ---
```

#### Performance Degradation

> :warning: During the model construction is possible to have an idea of how much the model performance will degrade over time !

Instead of randomly splittinthe samples to build the training and test sets, is possible to **"cut in time"**, so the data before this cut belongs to traning set and data after to test set. 

##### Check degradation

Calculate the accuracy per day (consider the time granularity used in the project) and project it in a bar graph so that the expected degradation is visible.

To see this run:

```bash
  # Folder src/
  $ python3 degradation.py
```
To see the performance over time with data degradation, change this lines:

```python
    # --- Ommited code ---

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

    # --- Ommited code ---
```

<br>
@2024, Insper. 9° Semester,  Computer Engineering.
<br>

_Machine Learning Ops & Interviews Discipline_