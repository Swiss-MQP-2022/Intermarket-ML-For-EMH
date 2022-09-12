import pandas as pd
import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from models import SimpleLSTM
from trainer import Trainer
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import utils

cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))

# TODO: figure out handling NaNs (currently using forward-fill)
df = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('timestamp').select_dtypes(include=['float64']).fillna(method='ffill')  # Load data from file
X_0 = df.iloc[0]
y_0 = X_0['close']

pct_df = df.pct_change()[1:]  # Compute percent change

# X_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
# X = X_scaler.fit_transform(pct_df[:-1])
# y = y_scaler.fit_transform(pct_df['close'].to_numpy().reshape(-1, 1)[1:]).flatten()

X = torch.tensor(pct_df[:-1].to_numpy()).float()
y = torch.tensor(pct_df['close'][1:].to_numpy()).float()

validation_split = 0.20
test_split = 0.20

dataloader = TimeSeriesDataLoader(X, y, validation_split=validation_split, test_split=test_split, batch_size=10)

model = SimpleLSTM(X.shape[1], 100, 3, batch_first=True)
if cuda_available:
    model.cuda()

criterion = nn.MSELoss()
learning_rate = 0.001

optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=100, verbose=True)

trainer = Trainer(model, criterion, optim, dataloader)

train_loss, validation_loss = trainer.train_loop(epochs=10, print_freq=5)

print('Creating plots...')

if cuda_available:
    X = X.cuda()

forecast, _ = model.forecast(X.unsqueeze(0))
forecast = forecast.flatten().cpu().detach().numpy()

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

# loss
axs[0].plot(train_loss, label='Training')
axs[0].plot(validation_loss, label='Validation')
axs[0].legend()
axs[0].set_title(f'Model Loss ({criterion.__class__.__name__})')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

# S&P 500 Forecasting
# axs[1].plot(utils.pct_to_cumulative(y_scaler.inverse_transform(forecast.reshape(-1, 1)).flatten(), y_0), label='Forecast')
# axs[1].plot(utils.pct_to_cumulative(y_scaler.inverse_transform(y.reshape(-1, 1)).flatten(), y_0), label='S&P 500')
axs[1].plot(utils.pct_to_cumulative(forecast, y_0), label='Forecast')
axs[1].plot(utils.pct_to_cumulative(y.numpy(), y_0), label='S&P 500')
axs[1].legend()
axs[1].set_title('Model Prediction vs. S&P 500')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].axvline(len(y)*(1-test_split), color="tab:purple")

plt.savefig('./images/plot.png')

print('Done!')
