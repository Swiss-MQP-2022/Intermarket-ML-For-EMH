import pandas as pd
import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from models import SimpleLSTM
from trainer import Trainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import utils

cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))

# TODO: figure out handling NaNs
df = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('timestamp').select_dtypes(include=['float64']).dropna()  # Load data from file
X_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
X = X_scaler.fit_transform(df.to_numpy()[:-1])
y = y_scaler.fit_transform(df['close'][1:].values.reshape(-1, 1))

X = utils.prices_to_percent_change(X)
y = utils.prices_to_percent_change(y)

X = torch.tensor(X).float()
y = torch.tensor(y).float()

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

epochs = 30
train_loss = []
validation_loss = []

for i in range(epochs):
    print(f'Epoch {i} in progress...')
    train_loss.append(trainer.train())
    validation_loss.append(trainer.validate())

print('Done training!')

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
axs[1].plot(forecast, label='Forecast')
axs[1].plot(y, label='S&P 500')
axs[1].legend()
axs[1].set_title('Model Prediction vs. S&P 500')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].axvline(len(y)*(1-test_split), color="tab:purple")

plt.savefig('./images/plot.png')

print('Done!')
