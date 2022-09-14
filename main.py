import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as Scaler

import utils
from timeseries_dataset import TimeSeriesDataLoader
from models import SimpleLSTM, LimLundgrenLoss
from trainer import Trainer

# Get CUDA availability
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))

# Load Data # TODO: figure out handling NaNs (currently using forward-fill)
df = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('timestamp')  # Load data from file
df = utils.get_nonempty_float_columns(df).dropna()  # filter to numeric columns. Drop NaNs

X_0 = df.iloc[0]  # record initial raw X values
y_0 = X_0['close']  # record initial raw y value

# pct_df = df.pct_change()[1:]  # Compute percent change
# pct_df = utils.remove_outliers(pct_df)

X_scaler, y_scaler = Scaler(), Scaler()  # Initialize scalers for normalization
X = X_scaler.fit_transform(df[:-1])  # normalize X data
y = y_scaler.fit_transform(df['close'].pct_change().to_numpy().reshape(-1, 1)[1:]).flatten() # normalize y data

# Put data on tensors
X = torch.tensor(X).float()
y = torch.tensor(y).float()

# X = torch.tensor(pct_df[:-1].to_numpy()).float()
# y = torch.tensor(pct_df['close'][1:].to_numpy()).float()

test_split = 0.20

# Create data loader
dataloader = TimeSeriesDataLoader(X, y, validation_split=0.20, test_split=test_split, period=1000, batch_size=10)

# Initialize model
model = SimpleLSTM(X.shape[1], 100, 3, batch_first=True, dropout=0.5)
if cuda_available:
    model.cuda()  # put model on CUDA if present

#  Initialize loss, optimizer, and scheduler
criterion = LimLundgrenLoss()  # Loss criterion
optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)  # Optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=100, verbose=True)  # Learning rate scheduler

# Initialize trainer
trainer = Trainer(model, criterion, optim, dataloader)

# !!! Train model !!!
train_loss, validation_loss = trainer.train_loop(epochs=10, print_freq=1)

print('Creating plots...')

if cuda_available:
    X = X.cuda()

forecast, _ = model.forecast(X.unsqueeze(0))  # Forecast on entire X data
forecast = forecast.flatten().cpu().detach().numpy()  # Reorganize for plotting

fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

# loss during training
axs[0].plot(train_loss, label='Training')
axs[0].plot(validation_loss, label='Validation')
axs[0].legend()
axs[0].set_title(f'Model Loss ({criterion.__class__.__name__})')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

# S&P 500 Forecasting
axs[1].plot(utils.pct_to_cumulative(y_scaler.inverse_transform(forecast.reshape(-1, 1)).flatten(), y_0), label='Forecast')
axs[1].plot(utils.pct_to_cumulative(y_scaler.inverse_transform(y.reshape(-1, 1)).flatten(), y_0), label='S&P 500')
# axs[1].plot(utils.pct_to_cumulative(forecast, y_0), label='Forecast')
# axs[1].plot(utils.pct_to_cumulative(y.numpy(), y_0), label='S&P 500')
axs[1].legend()
axs[1].set_title('Model Prediction vs. S&P 500')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Value')
axs[1].axvline(len(y)*(1-test_split), color="tab:purple")

plt.savefig('./images/plot.png')

# Classification report for direction accuracy
print(utils.forecast_classification_report(forecast, y))

print('Done!')
