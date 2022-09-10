import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from enum import Enum

class DataSplit(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2



class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 time_series_loader: TimeSeriesDataLoader):
        """
        :param model: model to train/evaluate
        :param criterion: loss module to use during training
        :param optimizer: optimizer to use during training
        :param time_series_loader: data loader for time series data to use
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.time_series_loader = time_series_loader
        self.cuda_available = torch.cuda.is_available()

    def train_validate(self, split: DataSplit):
        if split == DataSplit.TRAINING:
            training = True
            loader = self.time_series_loader.train_data_loader
        else:
            training = False
            if split == DataSplit.VALIDATION:
                loader = self.time_series_loader.validation_data_loader
            elif split == DataSplit.TESTING:
                loader = self.time_series_loader.test_data_loader
            else:
                raise Exception("Error: Invalid DataSplit provided.")

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.
        batches = 0

        for X, y in loader:
            if self.cuda_available:
                X = X.cuda()
                y = y.cuda()

            if training:
                self.optimizer.zero_grad()

            output, memory = self.model.forward(X)

            loss = self.criterion(output, y)

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            batches += y.size(0)

        return total_loss/batches

    def train(self):
        return self.train_validate(DataSplit.TRAINING)

    def validate(self):
        return self.train_validate(DataSplit.VALIDATION)

    def test(self):
        return self.train_validate(DataSplit.TESTING)
