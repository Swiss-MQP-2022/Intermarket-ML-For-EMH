from tqdm import tqdm
from enum import Enum
import numpy as np

import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from timeseries_dataset import TorchTimeSeriesDataLoader


class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATE = 'validation'
    TEST = 'test'
    ALL = 'ALL'


METRIC_NAMES = ['loss', 'accuracy', 'balanced accuracy']


def append_metrics(metric_dict: dict[str, list[np.number]], data: dict[str, np.number]):
    [metric_dict[metric].append(data[metric]) for metric in METRIC_NAMES]


class TorchTrainer:
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 time_series_loader: TorchTimeSeriesDataLoader,
                 scheduler=None,
                 reduction='mean'):
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
        self.reduction = reduction

        # this is an extremely jank way to dynamically get the number of classes because I don't want to pass it in
        self.n_classes = len(self.time_series_loader.dataset.__getitem__(0)[1])

    def train_validate(self, split: DataSplit) -> dict[str, np.number]:
        if split == DataSplit.ALL:
            raise Exception("Error: ALL data split is invalid for train_validate.")

        training = split == DataSplit.TRAIN
        loader = self.loaders[split]

        self.model.train() if training else self.model.eval()

        total_loss = 0.
        confusion = np.zeros((self.n_classes, self.n_classes))

        for X_, y_ in loader:
            if self.cuda_available:
                X_ = X_.cuda()
                y_ = y_.cuda()

            output, memory = self.model(X_)

            loss = self.criterion(output, y_)

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * (len(y_) if self.reduction == 'mean' else 1)  # need * len(y) when criterion reduction = 'mean'
            confusion += confusion_matrix(y_.argmax(dim=1), output.argmax(dim=1), labels=range(self.n_classes))

        if split == DataSplit.VALIDATE and self.scheduler is not None:
            self.scheduler.step(total_loss)

        # metric tracking. NOTE: THIS MUST AGREE WITH THE METRIC_NAMES VARIABLE
        metrics = {
            'loss': total_loss / len(loader.dataset),
            'accuracy': confusion.diagonal().sum() / confusion.sum(),
            'balanced accuracy': (confusion.diagonal() / confusion.sum(axis=1)).mean()
        }

        return metrics

    def train(self) -> dict[str, np.number]:
        return self.train_validate(DataSplit.TRAIN)

    def validate(self) -> dict[str, np.number]:
        return self.train_validate(DataSplit.VALIDATE)

    def test(self) -> dict[str, np.number]:
        return self.train_validate(DataSplit.TEST)

    def train_loop(self, epochs=100) -> dict[DataSplit, dict[str, list[np.number]]]:
        metrics = {
            DataSplit.TRAIN: {metric: [] for metric in METRIC_NAMES},
            DataSplit.VALIDATE: {metric: [] for metric in METRIC_NAMES}
        }

        for i in tqdm(range(epochs)):
            train_metrics = self.train()
            validation_metrics = self.validate()
            append_metrics(metrics[DataSplit.TRAIN], train_metrics)
            append_metrics(metrics[DataSplit.VALIDATE], validation_metrics)

        print('Training loop finished!')

        return metrics

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
