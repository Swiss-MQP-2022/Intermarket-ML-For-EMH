import pandas as pd
import torch
from torch.nn import BCELoss
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.metrics import classification_report
import numpy as np

import utils
from timeseries_dataset import TimeSeriesDataLoader
from models import SimpleLSTMClassifier
# from losses import LimLundgrenLoss
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

X_scaler = Scaler()  # Initialize scalers for normalization
X = X_scaler.fit_transform(df[:-1])  # normalize X data

y = np.sign(df['close'].diff()).to_numpy()[1:] + 1  # convert y to direction classes
# y = F.one_hot(y)  # one hot encode y

# Put data on tensors
X = torch.tensor(X).float()
y = F.one_hot(torch.tensor(y).long()).float()

test_split = 0.20

# Create data loader
dataloader = TimeSeriesDataLoader(X, y, validation_split=0.20, test_split=test_split, period=100, batch_size=1000)

# Initialize model
model = SimpleLSTMClassifier(X.shape[1], 100, 3, batch_first=True, dropout=0.5)
if cuda_available:
    model.cuda()  # put model on CUDA if present

#  Initialize loss, optimizer, and scheduler
criterion = BCELoss()  # Loss criterion
optim = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)  # Optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=10, verbose=True)  # Learning rate scheduler

# Initialize trainer
trainer = Trainer(model, criterion, optim, dataloader)

# !!! Train model !!!
train_loss, validation_loss = trainer.train_loop(epochs=10, print_freq=1)

print('Creating plots...')

# loss during training
plt.plot(train_loss, label='Training')
plt.plot(validation_loss, label='Validation')
plt.legend()
plt.title(f'Model Loss ({criterion.__class__.__name__})')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.savefig('./images/plot.png')

if cuda_available:
    X = X.cuda()

forecast, _ = model.forecast(X)  # Forecast on entire X data

print(classification_report(y.argmax(dim=1), forecast.argmax(dim=1)))

print('Done!')
