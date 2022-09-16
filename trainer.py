import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from enum import Enum
import numpy as np
from sklearn.metrics import classification_report


class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATE = 'validation'
    TEST = 'test'
    ALL = 'ALL'


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 time_series_loader: TimeSeriesDataLoader,
                 scheduler=None):
        """
        :param model: model to train/evaluate
        :param criterion: loss module to use during training
        :param optimizer: optimizer to use during training
        :param time_series_loader: data loader for time series data to use
        :param scheduler: learning rate scheduler to use
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.time_series_loader = time_series_loader
        self.loaders = {
            DataSplit.TRAIN: self.time_series_loader.train_data_loader,
            DataSplit.VALIDATE: self.time_series_loader.validation_data_loader,
            DataSplit.TEST: self.time_series_loader.test_data_loader,
            DataSplit.ALL: self.time_series_loader.all_data_loader
        }
        self.cuda_available = torch.cuda.is_available()

    def train_validate(self, split: DataSplit):
        if split == DataSplit.ALL:
            raise Exception("Error: ALL data split is invalid for train_validate.")

        training = split == DataSplit.TRAIN
        loader = self.loaders[split]

        self.model.train() if training else self.model.eval()

        total_loss = 0.

        for X, y in loader:
            if self.cuda_available:
                X = X.cuda()
                y = y.cuda()

            output, memory = self.model(X)

            loss = self.criterion(output, y)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * len(y)  # need * len(y) when criterion reduction = 'mean'

        if split == DataSplit.VALIDATE and self.scheduler is not None:
            self.scheduler.step(total_loss)

        return total_loss / len(loader.dataset)

    def train(self):
        return self.train_validate(DataSplit.TRAIN)

    def validate(self):
        return self.train_validate(DataSplit.VALIDATE)

    def test(self):
        return self.train_validate(DataSplit.TEST)

    def train_loop(self, epochs=100, print_freq=5):
        train_loss = []
        validation_loss = []

        for i in range(epochs):
            if i % print_freq == 0:
                print(f'Epoch {i} in progress...')
            train_loss.append(self.train())
            validation_loss.append(self.validate())

        print('Training loop finished!')

        return train_loss, validation_loss

    def get_classification_report(self, split: DataSplit):
        loader = self.loaders[split]

        print(f'Generating {split.value} classification report...')

        forecast = []
        expected = []

        for X_, y_ in loader:
            if self.cuda_available:
                X_ = X_.cuda()
            forecast.append(self.model.forecast(X_)[0].detach().cpu().numpy())
            expected.append(y_.detach().cpu().numpy())

        test_forecast = np.concatenate(forecast)
        test_expected = np.concatenate(expected)

        return classification_report(test_expected.argmax(axis=1), test_forecast.argmax(axis=1))