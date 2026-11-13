import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from constants import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FullyConnectedNeuralNetwork:

    def __init__(self, hidden_layers=Default_hidden_layers, activation=Default_activation, learning_rate=Default_learning_rate, batch_size=Default_batch_size, weight_decay=Default_weight_decay, epochs=Default_epochs, valid_frac=Default_valid_frac, random_state=Random_state):

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.valid_frac = valid_frac
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        self.model = None
        self.loss_train = []
        self.loss_valid = []

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_data(self):
        self.df = pd.read_csv(os.path.join("provided_files", "dataset_cell_growth_7factor.csv"))
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
                best_model_state = deepcopy(self.model.state_dict())

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
