import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from enum import Enum
import numpy as np
from sklearn.metrics import classification_report


class DataSplit(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2
    ALL = 3


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
        self.cuda_available = torch.cuda.is_available()

    def train_validate(self, split: DataSplit):
        if split == DataSplit.TRAIN:
            training = True
            loader = self.time_series_loader.train_data_loader
        else:
            training = False
            if split == DataSplit.VALIDATE:
                loader = self.time_series_loader.validation_data_loader
            elif split == DataSplit.TEST:
                loader = self.time_series_loader.test_data_loader
            else:
                raise Exception("Error: Invalid DataSplit provided to train_validate.")

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

            output, memory = self.model(X)

            loss = self.criterion(output, y)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            batches += y.size(0)

        if split == DataSplit.VALIDATE and self.scheduler is not None:
            self.scheduler.step(total_loss)

        return total_loss / batches

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
        if split == DataSplit.TRAIN:
            loader = self.time_series_loader.train_data_loader
            report_name = "train"
        elif split == DataSplit.VALIDATE:
            loader = self.time_series_loader.validation_data_loader
            report_name = "validaion"
        elif split == DataSplit.TEST:
            loader = self.time_series_loader.test_data_loader
            report_name = "test"
        elif split == DataSplit.ALL:
            loader = self.time_series_loader.all_data_loader
            report_name = "ALL"
        else:
            raise Exception('Error: Invalid data split provided')

        print(f'Generating {report_name} classification report...')

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