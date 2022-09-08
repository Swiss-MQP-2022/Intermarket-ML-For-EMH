import torch
from torch import nn


class SimpleLSTM(nn.Module):
    # Model initialization
    def __init__(self, input_size, hidden_size, num_layers, *args, **kwargs):
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, *args, **kwargs)
        self.out_layer = nn.Linear(hidden_size[-1], 1)  # Output layer

    # Model operation
    def forward(self, x, memory=None):
        batch_size = x.shape[0]  # batch first
        if memory is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
            memory = (h0, c0)

        lstm_out, (_, c_n) = self.lstm(x, memory)
        memory = (lstm_out, c_n[-1])

        output = self.out_layer(lstm_out)

        return output, memory


class ComboLoss(nn.Module):
    def __init__(self,
                 c_bias=1,
                 regression_fn=nn.functional.mse_loss,
                 classification_fn=nn.functional.binary_cross_entropy):

        super(ComboLoss, self).__init__()
        self.regression_fn = regression_fn
        self.classification_fn = classification_fn
        self.c_bias = c_bias

    def forward(self, pred, target):
        pred_binary = torch.sign(pred)
        target_binary = torch.sign(target)

        loss = self.regression_fn(pred, target) + (self.c_bias * self.classification_fn(pred_binary, target_binary))

        return loss
