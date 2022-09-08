import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 time_series_loader: TimeSeriesDataLoader,
                 optim: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.time_series_loader = time_series_loader
        self.optim = optim

    def train_test(self, training):
        if training:
            self.model.train()
            loader = self.time_series_loader.train_data_loader
        else:
            self.model.eval()
            loader = self.time_series_loader.test_data_loader

        total_loss = 0.

        for X, y in loader:
            if training:
                self.optim.zero_grad()

            output, memory = self.model.forward(X)

            loss = self.criterion(output, y)

            if training:
                loss.backward()
                self.optim.step()

            total_loss += loss.item() * output.size(0)

        return total_loss

    def train(self):
        return self.train_test(True)

    def test(self):
        return self.train_test(False)
