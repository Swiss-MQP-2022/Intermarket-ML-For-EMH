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

        for X, y in loader:
            if training:
                self.optim.zero_grad()

            output, memory = self.model.forward(X)
            print(output.shape,y.shape)

            loss = self.criterion(output, y)

            if training:
                loss.backward()
                self.optim.step()

    def train(self):
        self.train_test(True)

    def test(self):
        self.train_test(False)
