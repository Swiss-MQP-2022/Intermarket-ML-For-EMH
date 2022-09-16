import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.metrics import classification_report
import numpy as np

import utils
from timeseries_dataset import TimeSeriesDataLoader
import models
# from losses import LimLundgrenLoss
from trainer import Trainer
from trainer import DataSplit

# Get CUDA availability
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))

# Load Data # TODO: figure out handling NaNs
spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
spy = utils.get_nonempty_float_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

X_0 = spy.iloc[0]  # record initial raw X values

# brn = utils.generate_brownian_motion(len(spy), len(spy.columns), initial=X_0.to_numpy())
# print(len(spy))

pct_df = spy.pct_change()[1:]  # Compute percent change
pct_df = utils.remove_outliers(pct_df)

# brn = utils.generate_brownian_motion(len(pct_df), len(pct_df.columns), cumulative=False)

X_scaler = Scaler(feature_range=(-1, 1))  # Initialize scalers for normalization
X = X_scaler.fit_transform(pct_df[:-1])  # normalize X data
# # X = X_scaler.fit_transform(utils.pct_to_cumulative(pct_df, X_0)[:-1])  # normalize X data

# X_scaler = Scaler()  # Initialize scalers for normalization
# X = X_scaler.fit_transform(brn[:-1])  # normalize X data
# X = X_scaler.fit_transform(spy[:-1])  # normalize X data

y = np.sign(pct_df['close'].to_numpy())[1:] + 1

# y = np.sign(spy['close'].diff()).to_numpy()[1:] + 1  # convert y to direction classes

# Put data on tensors
X = torch.fft.fftn(torch.tensor(X), dim=0).float()
y = F.one_hot(torch.tensor(y).long()).float()

validation_split = 0.20
test_split = 0.20
period = 100
batch_size = 10000

# Create data loader
dataloader = TimeSeriesDataLoader(X, y, validation_split=validation_split, test_split=test_split, period=period, batch_size=batch_size)

# Initialize model
# model = models.SimpleLSTMClassifier(X.shape[1], 100, 3, batch_first=True, dropout=0.2)
model = models.SimpleFFClassifier(X.shape[1], period, 100, 3, dropout=0)
if cuda_available:
    model.cuda()  # put model on CUDA if present

#  Initialize loss, optimizer, and scheduler
criterion = CrossEntropyLoss(reduction='mean')  # Loss criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)  # Optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)  # Learning rate scheduler

# Initialize trainer
trainer = Trainer(model, criterion, optimizer, dataloader, scheduler=scheduler)

# !!! Train model !!!
train_loss, validation_loss = trainer.train_loop(epochs=50, print_freq=5)

print('Creating plots...')

# loss during training
plt.plot(train_loss, label='Training')
plt.plot(validation_loss, label='Validation')
plt.legend()
plt.title(f'Model Loss ({criterion.__class__.__name__})')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('./images/plot.png')

print('Plots saved.')

print(trainer.get_classification_report(DataSplit.TRAIN))
print(trainer.get_classification_report(DataSplit.VALIDATE))
print(trainer.get_classification_report(DataSplit.TEST))
print(trainer.get_classification_report(DataSplit.ALL))

print('Done!')
