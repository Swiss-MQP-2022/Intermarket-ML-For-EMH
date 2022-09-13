import torch
from torch import nn


class SimpleLSTM(nn.Module):
    # Model initialization
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        """
        :param input_size: input size of the model
        :param hidden_size: number of nodes to use per layer of LSTM
        :param num_layers: layers to use within LSTM
        :param **kwargs: keyword arguments to pass to nn.LSTM
        """
        super(SimpleLSTM, self).__init__()
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

        batches = lstm_out.size(0)
        period = lstm_out.size(1)

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

        output = self.out_layer(lstm_out)

        return output, memory


class ComboLoss(nn.Module):
    def __init__(self,
                 c_bias=1,
                 regression_fn=nn.functional.mse_loss,
                 classification_fn=nn.functional.binary_cross_entropy):
        """
        :param c_bias: weight factor used to bias classification loss
        :param regression_fn: function for calculating regression loss
        :param classification_fn: function for calculating classification loss
        """

        super(ComboLoss, self).__init__()
        self.regression_fn = regression_fn
        self.classification_fn = classification_fn
        self.c_bias = c_bias

    def forward(self, reg_pred, reg_target, cla_pred, cla_target):
        reg_loss = self.regression_fn(reg_pred, reg_target)
        classification_loss = self.c_bias * self.classification_fn(cla_pred, cla_target)
        return reg_loss + classification_loss


class LimLundgrenLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(LimLundgrenLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, truth):
        numer = (truth - pred).square().sum()
        denom = pred.sign().eq(truth.sign()).sum() + self.epsilon
        return numer / denom
