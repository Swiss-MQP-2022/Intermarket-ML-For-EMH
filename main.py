import pandas as pd
import torch
from torch import nn
from timeseries_dataset import TimeSeriesDataLoader
from models import SimpleLSTM
from trainer import Trainer

cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')
if cuda_available:
    print(torch.cuda.get_device_name(0))

# TODO: figure out handling NaNs
df = pd.read_csv(r"data/stock/SPY.US.csv").select_dtypes(include=['float64']).dropna()  # Load data from file
X = torch.tensor(df.to_numpy()).float()  # get input data
y = torch.tensor(df["close"].to_numpy()).float()  # get expected data

if cuda_available:
    X.cuda()
    y.cuda()

dataloader = TimeSeriesDataLoader(X, y)

model = SimpleLSTM(X.shape[1], 100, 3, batch_first=True)
if cuda_available:
    model.cuda()

criterion = nn.MSELoss()
learning_rate = 0.01

optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)

trainer = Trainer(model, criterion, optim, dataloader)

epochs = 25
train_loss = []
test_loss = []

for i in range(epochs):
    print(f'Epoch {i} in progress...')
    train_loss.append(trainer.train())
    test_loss.append(trainer.test())

print('Done training!')
