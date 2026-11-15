import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, ParameterGrid

from src.Classes import FullyConnectedNeuralNetwork
from src.functions import progress_bar

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Load the dataset
df = pd.read_csv(os.path.join("provided_files", "dataset_cell_growth_7factor.csv"))

# Define input features and target variables
features = df.columns[3:]
targets = df.columns[:3]

# Convert DataFrame columns to NumPy arrays
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into training and testing sets
#I believe this will need to be changed to cross validation...:
#cv_results = cross_validate(model, x, y, cv=5, scoring=("neg_root_mean_squared_error", "r2"), return_train_score=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define the hyperparameter search space
param_grid = dict(
    hidden_layers=[
        (5,),
        (6, 5, 4),
        (8, 16, 32, 16, 8, 4),
    ],
    activation=["tanh", "relu"],
    learning_rate=[0.001, 0.01],
)

# Track iterations for progress bar
c_iter = 0
n_iters = len(ParameterGrid(param_grid))

# Records to export
columns = list(param_grid.keys()) + ["target", "NRMSE", "RMSE"]
records = []

# Train and evaluate every hyperparameter combination.
# ParameterGrid enables GridSearchCV-type behaviour.
for hyperparameters in ParameterGrid(param_grid):

    # Initialize the model
    model = FullyConnectedNeuralNetwork(
        epochs=400,
        batch_size=1024,
        verbose=False,
        random_state=0,
        **hyperparameters
    )

    # Split the training data into training and validation sets
    x_train_fit, x_valid, y_train_fit, y_valid = model.split_data(x_train, y_train)

    # Fit the model
    model.fit(x_train_fit, x_valid, y_train_fit, y_valid)

    # Predict the test set targets
    y_pred = model.predict(x_test)

    # Calculate the interquartile range (IQR)
    iqr = np.quantile(y_test, q=0.75, axis=0) - np.quantile(y_test, q=0.25, axis=0)

    # Calculate performance metrics for each target variable
    for j, target in enumerate(targets):
        # Compute RMSE
        rmse = root_mean_squared_error(y_test[:, j], y_pred[:, j])

        # Compute NRMSE using the IQR for normalization
        nrmse = rmse / iqr[j]

        record = tuple([hyperparameters[key] for key in columns[:len(hyperparameters)]] + [target, nrmse, rmse])
        records.append(record)

    c_iter += 1
    progress_bar(c_iter, n_iters)

df_export = pd.DataFrame.from_records(records, columns=columns)
df_export.to_csv(os.path.join("tables", "table_grid_search.csv"), index=False)