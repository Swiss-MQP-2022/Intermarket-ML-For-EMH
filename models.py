import torch
from torch import nn


class SimpleLSTMRegressor(nn.Module):
    # Model initialization
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        """
        :param input_size: input size of the model
        :param hidden_size: number of nodes to use per layer of LSTM
        :param num_layers: layers to use within LSTM
        :param **kwargs: keyword arguments to pass to nn.LSTM
        """
        super(SimpleLSTMRegressor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.out_layer = nn.Linear(hidden_size, 1)  # Output layer
        self.sigmoid = nn.Sigmoid()

    # Model operation
    def forward(self, x, *args):
        """
        :param x: tensor with input data for the model to operate over
        :param *args: arbitrary arguments to provide with x. Expected to include (h_0, c_0) if available
        """
        lstm_out, (_, c_n) = self.lstm(x, *args)
        memory = (lstm_out, c_n[-1])

        # batches = lstm_out.size(0)
        # period = lstm_out.size(1)

        lin_out = self.out_layer(lstm_out[:, -1, :])
        # lin_out = self.out_layer(lstm_out.flatten(start_dim=0, end_dim=1)).view((batches, period))
        output = self.sigmoid(lin_out)

        return output, memory

    def forecast(self, x, *args):
        """
        :param x: tensor with input data for the model to operate over
        :param *args: arbitrary arguments to provide with x. Expected to include (h_0, c_0) if available
        """
        lstm_out, (_, c_n) = self.lstm(x, *args)
        memory = (lstm_out, c_n[-1])

        lin_out = self.out_layer(lstm_out)
        output = self.sigmoid(lin_out)

        return output, memory


class SimpleLSTMClassifier(nn.Module):
    # Model initialization
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        """
        :param input_size: input size of the model
        :param hidden_size: number of nodes to use per layer of LSTM
        :param num_layers: layers to use within LSTM
        :param **kwargs: keyword arguments to pass to nn.LSTM
        """
        super(SimpleLSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs)
        self.out_layer = nn.Linear(hidden_size, 3)  # Output layer
        self.softmax = nn.Softmax(dim=1)

    # Model operation
    def forward(self, x, *args):
        """
        :param x: tensor with input data for the model to operate over
        :param *args: arbitrary arguments to provide with x. Expected to include (h_0, c_0) if available
        """
        lstm_out, (_, c_n) = self.lstm(x, *args)
        memory = (lstm_out, c_n[-1])

        # batches = lstm_out.size(0)
        # period = lstm_out.size(1)

        lin_out = self.out_layer(lstm_out[:, -1, :])
        # lin_out = self.out_layer(lstm_out.flatten(start_dim=0, end_dim=1)).view((batches, period))
        output = self.softmax(lin_out)

        return output, memory

    def forecast(self, x, *args):
        """
        :param x: tensor with input data for the model to operate over
        :param *args: arbitrary arguments to provide with x. Expected to include (h_0, c_0) if available
        """
        lstm_out, (_, c_n) = self.lstm(x, *args)
        memory = (lstm_out, c_n[-1])

        lin_out = self.out_layer(lstm_out)
        output = self.softmax(lin_out)

        return output, memory


class SimpleFFClassifier(nn.Module):
    def __init__(self, features_count, period, hidden_size, out_classes):
        super(SimpleFFClassifier, self).__init__()
        self.feature_count = features_count
        self.period = period
        self.hidden_size = hidden_size
        self.out_classes = out_classes

        self.in_layer = nn.Linear(self.period * self.feature_count, self.hidden_size)
        self.h1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.out_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x_ = x.flatten(start_dim=1, end_dim=2)
        y_0 = self.relu(self.in_layer(x_))
        y_1 = self.relu(self.h1(y_0))
        y_2 = self.relu(self.h2(y_1))
        output = self.softmax(self.out_layer(y_2))

        return output, None
