import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

import numpy as np
from matplotlib import pyplot as plt


class NNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model,
        loss=nn.MSELoss(),
        optimizer_class=optim.Adam,
        optimizer_params={},
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
        metrics={"mse": mean_squared_error},
        val_set=None,
        verbose=False,
    ):
        self.model = model
        self.loss = loss
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.val_set = val_set
        self.device = device
        self.verbose = verbose
        self.train_loss = []
        self.val_loss = []
        self.train_metrics = {k: [] for k, _ in metrics.items()}
        self.val_metrics = {k: [] for k, _ in metrics.items()}

    def on_train_start(self):
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
        self.model.to(self.device)
        self.model.train()

    def on_train_end(self):
        pass
    
    def on_epoch_start(self):
        pass
    
    def on_epoch_end(self, epoch, X, y):
        self.evaluate(X, y)
        if self.val_set:
            X_val, y_val = self.tensorify(*self.val_set)
            self.evaluate(X_val, y_val)

        if self.verbose and epoch % 10 == 0:
            msg = f"Epoch {epoch + 1}/{self.epochs}"
            msg += f" - Loss: {self.train_loss[-1]:.4f}"
            msg += f" - Train metrics: {', '.join([f'{name}: {evals[-1]:.4f}' for name, evals in self.train_metrics.items()])}"
            if self.val_set:
                msg += f" - Val Loss: {self.val_loss[-1]:.4f}"
                msg += f" - Val Metrics: {', '.join([f'{name}: {evals[-1]:.4f}' for name, evals in self.val_metrics.items()])}"
            print(msg)
    
    def evaluate(self, X, y, no_grad=False):
        y_train_preds = self.model(X).detach()
        self.train_loss.append(self.loss(y_train_preds, y).item())
        for name, metric in self.metrics.items():
            train_metric_value = metric(y, y_train_preds.numpy())
            self.train_metrics[name].append(train_metric_value)
        
    def evaluate(self, X, y, is_train=True):
        X, y = self.pytorchify_data(X, y)
        self.model.eval()
        sset = "train" if is_train else "val"
        
        with torch.no_grad():
            preds = self.model(X)
            loss = self.loss(preds, y).item()
            getattr(self, sset+"loss").append(loss)

            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            for name, metric in self.metrics.items():
                getattr(self, sset+"metrics")[name].append(metric(y_np, preds_np))

    def train_step(self, X_batch, y_batch):
        self.optimizer.zero_grad()
        preds = self.model(X_batch)
        loss = self.loss(preds, y_batch)
        loss.backward()
        self.optimizer.step()

    def tensorify(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y:
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32).to(self.device)
            return X, y
        else:
            return y

    def fit(self, X, y):
        X, y = self.tensorify(X, y)
        dataloader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True)

        self.on_train_start()
        for epoch in range(self.epochs):
            self.on_epoch_start(epoch)
            for batch_X, batch_y in dataloader:
                self.train_step(batch_X, batch_y)
            self.on_epoch_end(epoch)
        self.on_train_end()

        return self

    def predict(self, X):
        X = self.tensorify(X)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        return predictions

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label="Training Loss")
        if self.val_set:
            plt.plot(self.val_loss, label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_metrics(self):
        for name, _ in self.metrics.items():
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_metrics[name], label=f"Training {name.upper()}")
            if self.val_set:
                plt.plot(self.val_metrics[name], label=f"Validation {name.upper()}")
            plt.title(f"{name.upper()} Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(name.upper())
            plt.legend()
            plt.show()
