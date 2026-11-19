
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from src.constants import Random_state, CV_folds, Default_hidden_layers, Default_activation, Default_learning_rate, Default_batch_size, Default_epochs, Default_valid_frac, Default_weight_decay
from src.functions import calculate_metrics


class FullyConnectedNeuralNetwork:

    def __init__(self, hidden_layers=Default_hidden_layers, activation=Default_activation, learning_rate=Default_learning_rate, batch_size=Default_batch_size, weight_decay=Default_weight_decay, epochs=Default_epochs, valid_frac=Default_valid_frac, random_state=Random_state, verbose=False):

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.verbose = verbose

        self.rng = np.random.default_rng(random_state)

        self.model = None
        self.loss_train = []
        self.loss_valid = []

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_data(self):
        self.df = pd.read_csv(os.path.join("datasets", "dataset_cell_growth_7factor.csv"))

        self.features = self.df.columns[3:]
        self.targets = self.df.columns[:3]
        self.x = self.df.loc[:, self.features].to_numpy(dtype=np.float32)
        self.y = self.df.loc[:, self.targets].to_numpy(dtype=np.float32)

        return self.x, self.y

    def split_data(self, x, y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.valid_frac, random_state=self.random_state)
        return x_train, x_valid, y_train, y_valid

    def prepare_data(self, x_train, x_valid, y_train, y_valid):
        self.scaler_x.fit(x_train)
        self.scaler_y.fit(y_train)

        x_train = self.scaler_x.transform(x_train)
        x_valid = self.scaler_x.transform(x_valid)
        y_train = self.scaler_y.transform(y_train)
        y_valid = self.scaler_y.transform(y_valid)

        return x_train, x_valid, y_train, y_valid

    def train_data(self, x_train, y_train):

        dataset_train = TensorDataset(x_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        return dataloader_train

    def build_model(self, n_features, n_targets):
        valid_activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}
        nn_Activation = valid_activations[self.activation]

        seq_stack = []
        n_nodes_in = n_features

        for n_nodes_out in self.hidden_layers:
            seq_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            seq_stack.append(nn_Activation())
            n_nodes_in = n_nodes_out

        seq_stack.append(nn.Linear(n_nodes_in, n_targets))

        self.model = nn.Sequential(*seq_stack)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return self.model

    def train_interaction(self, dataloader_train):
        self.model.train()
        loss_train = 0.0

        for x_batch, y_batch in dataloader_train:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item() * x_batch.size(0)

        loss_train = loss_train / len(dataloader_train.dataset)

        return loss_train

    def valid_interaction(self, x_valid, y_valid):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_valid)
            loss_valid = self.loss_fn(y_pred, y_valid).item()

        return loss_valid

    def fit(self, x_train, x_valid, y_train, y_valid):
        torch.manual_seed(self.random_state)

        x_train, x_valid, y_train, y_valid = self.prepare_data(x_train, x_valid, y_train, y_valid)
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        dataloader_train = self.train_data(x_train, y_train)

        self.build_model(x_train.shape[1], y_train.shape[1])

        self.loss_train = []
        self.loss_valid = []

        best_loss_valid = None
        best_model_state = None

        for epoch in range(1, self.epochs + 1):

            loss_train = self.train_interaction(dataloader_train)
            loss_valid = self.valid_interaction(x_valid, y_valid)

            self.loss_train.append(loss_train)
            self.loss_valid.append(loss_valid)

            if (best_loss_valid is None) or (loss_valid < best_loss_valid):
                best_loss_valid = loss_valid
                best_model_state = {key: value.clone() for key, value in self.model.state_dict().items()}

        self.model.load_state_dict(best_model_state)

        return self

    def predict(self, x):

        x = self.scaler_x.transform(x)
        x = torch.tensor(x, dtype=torch.float32)

        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(x).numpy()

        y_pred = self.scaler_y.inverse_transform(y_pred)

        return y_pred

    def cross_validate_model(self, x, y, targets=None, n_splits=CV_folds):
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        records = []
        learning_curves = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(x), start=1):

            x_train_fold = x[train_idx]
            x_test_fold = x[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]

            model = FullyConnectedNeuralNetwork(hidden_layers=self.hidden_layers, activation=self.activation,
                                                learning_rate=self.learning_rate, batch_size=self.batch_size,
                                                weight_decay=self.weight_decay, epochs=self.epochs,
                                                valid_frac=self.valid_frac, random_state=self.random_state,
                                                verbose=self.verbose)

            x_train, x_valid, y_train, y_valid = model.split_data(x_train_fold, y_train_fold)
            model.fit(x_train, x_valid, y_train, y_valid)

            y_pred = model.predict(x_test_fold)

            rmse, nrmse = calculate_metrics(y_test_fold, y_pred)

            for j in range(y.shape[1]):
                target = targets[j] if targets is not None else f"Target {j + 1}"
                records.append((fold, target, rmse[j], nrmse[j]))

            learning_curves.append((model.loss_train.copy(), model.loss_valid.copy()))

        df_cv = pd.DataFrame(records, columns=["Fold", "Target", "RMSE", "NRMSE"])

        df_summary = df_cv.groupby("Target")[["RMSE", "NRMSE"]].agg(["mean", "std"]).reset_index()

        return df_cv, df_summary, learning_curves

    def plot_cv_learning_curves(self, learning_curves):

        os.makedirs("figures", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for fold, (loss_train, loss_valid) in enumerate(learning_curves):
            ax = axes.flatten()[fold]

            ax.plot(range(1, len(loss_train) + 1), loss_train, label="Training Loss")
            ax.plot(range(1, len(loss_valid) + 1), loss_valid, label="Validation Loss")

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Fold {fold + 1}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join("figures", "cross_validation_learning_curves.png"))
        plt.close()

    def plot_learning_curves(self, fold):

        os.makedirs("figures", exist_ok=True)

        plt.figure()
        plt.plot(range(1, len(self.loss_train) + 1), self.loss_train, label="Training Loss")
        plt.plot(range(1, len(self.loss_valid) + 1), self.loss_valid, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Learning Curves - Fold {fold}")
        plt.legend()
        plt.savefig(os.path.join("figures", f"learning_curves_fold_{fold}.png"))
        plt.close()


class PolynomialRegression:

    def __init__(self, degrees=(1, 2, 3), cv=5):

        self.degrees = degrees
        self.cv = cv
        self.model = None

    def fit(self, x_train, y_train):

        pipeline = Pipeline([("polynomial", PolynomialFeatures(include_bias=False)), ("scaler", StandardScaler()), ("regression", LinearRegression())])

        param_grid = {"polynomial__degree": self.degrees}

        self.model = GridSearchCV(pipeline, param_grid, cv=self.cv, scoring="neg_root_mean_squared_error", n_jobs=-1)

        self.model.fit(x_train, y_train)

        return self

    def predict(self, x):

        return self.model.predict(x)
