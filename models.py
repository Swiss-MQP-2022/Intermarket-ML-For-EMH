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
