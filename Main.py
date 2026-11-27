
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, ParameterGrid

from src.Classes import FullyConnectedNeuralNetwork, PolynomialRegression
from src.functions import progress_bar, calculate_metrics

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load the dataset
df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_7factor.csv"))

# Define input features and target variables
features = df.columns[3:]
targets = df.columns[:3]

# Convert DataFrame columns to NumPy arrays
x = df.loc[:, features].to_numpy()
y = df.loc[:, targets].to_numpy()

# Split the data into development and testing sets
# The testing set will only be used for final model comparison
x_development, x_test, y_development, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Split the development data into training and search validation sets
x_train, x_search_valid, y_train, y_search_valid = train_test_split(x_development, y_development, test_size=0.2, random_state=0)

# Define the hyperparameter search space
param_grid = dict(
    hidden_layers=[
        (5,),
        (6, 5, 4),
        (8, 16, 32, 16, 8, 4),],
    activation=["tanh", "relu"],
    learning_rate=[0.001, 0.01, 0.1],
    batch_size=[64, 256],
    weight_decay=[0.0, 0.01])

# Track iterations for progress bar
c_iter = 0
n_iters = len(ParameterGrid(param_grid))

# Records to export
columns = list(param_grid.keys()) + ["target", "NRMSE", "RMSE"]
records = []

# Train and evaluate every hyperparameter combination
for hyperparameters in ParameterGrid(param_grid):

    # Initialize the model
    model = FullyConnectedNeuralNetwork(epochs=400, verbose=False, random_state=0, **hyperparameters)

    # Split the training data into training and validation sets
    x_train_fit, x_valid, y_train_fit, y_valid = model.split_data(x_train, y_train)

    # Fit the model
    model.fit(x_train_fit, x_valid, y_train_fit, y_valid)

    # Predict the search validation set targets
    y_pred = model.predict(x_search_valid)

    # Calculate performance metrics for each target variable
    rmse, nrmse = calculate_metrics(y_search_valid, y_pred)

    for j, target in enumerate(targets):
        record = tuple([hyperparameters[key] for key in param_grid.keys()] + [target, nrmse[j], rmse[j]])
        records.append(record)

    c_iter += 1
    progress_bar(c_iter, n_iters)

# Export hyperparameter search results
df_export = pd.DataFrame.from_records(records, columns=columns)
df_export.to_csv(os.path.join("tables", "table_grid_search.csv"), index=False)

# Calculate mean NRMSE across all three targets
df_summary = (
    df_export.groupby(list(param_grid.keys()), as_index=False)["NRMSE"]
    .mean()
    .sort_values("NRMSE"))

# Extract the optimal hyperparameters
optimal_params = df_summary.iloc[0][list(param_grid.keys())].to_dict()
optimal_params["hidden_layers"] = tuple(optimal_params["hidden_layers"])

print("\nBest hyperparameters:")
print(optimal_params)

# Export ranked hyperparameter combinations
df_summary.to_csv(os.path.join("tables", "table_grid_search_summary.csv"), index=False)

# Perform four-fold cross-validation using the optimal hyperparameters

model_cv = FullyConnectedNeuralNetwork(epochs=400, verbose=False, random_state=0, **optimal_params)

df_cv, df_cv_summary, learning_curves = model_cv.cross_validate_model(x_train, y_train, targets=targets, n_splits=4)

# Export cross-validation results
df_cv.to_csv(os.path.join("tables", "table_cross_validation.csv"), index=False)
df_cv_summary.to_csv(os.path.join("tables", "table_cross_validation_summary.csv"), index=False)

# Plot cross-validation learning curves
model_cv.plot_cv_learning_curves(learning_curves)

print("\nCross-validation results:")
print(df_cv_summary)

# Define sensitivity analysis parameters
hidden_layers = [tuple(max(1, round(layer * scale)) for layer in optimal_params["hidden_layers"]) for scale in [0.5, 0.75, 1, 1.25, 1.5]]

activation = ["sigmoid", "tanh", "relu", "leakyrelu"]

learning_rates = [optimal_params["learning_rate"] * value for value in [0.1, 0.5, 1, 2, 10]]

batch_sizes = sorted(set([max(1, optimal_params["batch_size"] // 4), max(1, optimal_params["batch_size"] // 2), optimal_params["batch_size"], optimal_params["batch_size"] * 2, optimal_params["batch_size"] * 4]))

if optimal_params["weight_decay"] == 0:
    weight_decays = [0, 0.0001, 0.001, 0.01, 0.1]
else:
    weight_decays = [optimal_params["weight_decay"] * value for value in [0.1, 0.5, 1, 2, 10]]

sensitivity_records = []
sensitivity_parameters = {"hidden_layers": hidden_layers, "activation": activation, "learning_rate": learning_rates, "batch_size": batch_sizes, "weight_decay": weight_decays}

for parameter, values in sensitivity_parameters.items():

    for value in values:

        # Start with the optimal hyperparameters
        hyperparameters = optimal_params.copy()
        hyperparameters[parameter] = value

        # Initialize the model
        model = FullyConnectedNeuralNetwork(epochs=400, verbose=False, random_state=0, **hyperparameters)

        df_sensitivity, df_sensitivity_summary, _ = model.cross_validate_model(x_train, y_train, targets=targets, n_splits=3)

        for _, row in df_sensitivity_summary.iterrows():
            sensitivity_records.append({"Parameter": parameter, "Value": value, "Target": target, "Mean RMSE": row[("RMSE", "mean")], "Std RMSE": row[("RMSE", "std")], "Mean NRMSE": row[("NRMSE", "mean")], "Std NRMSE": row[("NRMSE", "std")]})

df_sensitivity_export = pd.DataFrame(sensitivity_records)
df_sensitivity_export.to_csv(os.path.join("tables", "table_sensitivity_analysis.csv"), index=False)

df_sensitivity_summary = (df_sensitivity_export.groupby(["Parameter", "Value"], as_index=False, sort=False)[["Mean RMSE", "Std RMSE", "Mean NRMSE", "Std NRMSE"]].mean())
df_sensitivity_summary.to_csv(os.path.join("tables", "table_sensitivity_analysis_summary.csv"), index=False)

# Train the final neural network using the optimal hyperparameters
model_nn = FullyConnectedNeuralNetwork(epochs=400, verbose=False, random_state=0, **optimal_params)
x_train_fit, x_valid, y_train_fit, y_valid = model_nn.split_data(x_development, y_development)

nn_start = time.perf_counter()
model_nn.fit(x_train_fit, x_valid, y_train_fit, y_valid)
nn_training_time = time.perf_counter() - nn_start

# Predict the testing set targets
y_pred_nn = model_nn.predict(x_test)
rmse_nn, nrmse_nn = calculate_metrics(y_test, y_pred_nn)

# Train and evaluate polynomial regression
model_poly = PolynomialRegression(degrees=(1, 2, 3), cv=5)

poly_start = time.perf_counter()
model_poly.fit(x_development, y_development)
poly_training_time = time.perf_counter() - poly_start

y_pred_poly = model_poly.predict(x_test)
rmse_poly, nrmse_poly = calculate_metrics(y_test, y_pred_poly)

# Compare final model performance
comparison_records = []

for j, target in enumerate(targets):
    comparison_records.append({"Target": target, "NN RMSE": rmse_nn[j], "NN NRMSE": nrmse_nn[j], "Polynomial RMSE": rmse_poly[j], "Polynomial NRMSE": nrmse_poly[j]})

df_comparison = pd.DataFrame(comparison_records)
df_comparison.to_csv(os.path.join("tables", "table_model_comparison.csv"), index=False)

print("\nFinal model comparison:")
print(df_comparison)

print("\nTraining times:")
print(f"Neural network: {nn_training_time:.2f} seconds")
print(f"Polynomial regression: {poly_training_time:.2f} seconds")

print("\nBest polynomial degree:")
print(model_poly.model.best_params_)
