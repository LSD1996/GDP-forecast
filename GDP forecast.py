import pandas as pd
import torch.nn
import torch
import torch.optim

# Dataset prepocessing
data=pd.read_csv("GDP/GDPdata.csv")
data=data.drop(['Series Name', 'Series Code', 'Country Code'],axis=1)
print(data)
data = data.drop([11,12,13,14,15],axis=0)
data.set_index('Country Name', inplace=True)
year = list(range(1971,2020,1))
print(year)
df = data.T
df.index= year

total = df.isnull().sum().sort_values(ascending=False)
print(total)
print(df)
print(df.columns)
print(df.index)
print(df.shape)


# LSTM model
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x[:, :, None]
        x, _ = self.lstm(x)
        x = self.classifier(x)
        x = x[:, :, 0]
        return x


year_num, sample_num = df.shape

countries = df.columns

years = df.index

net = Net(input_size=1, hidden_size=5)

print(net)



train_seq_len = sum((years >= 1971) & (years <= 2000))
test_seq_len = sum(years > 2000)

print('train_seq_len = {}，test_seq_len={}'.format(train_seq_len, test_seq_len))

# Training
inputs = torch.tensor(df.iloc[:-1].values, dtype=torch.float32)
targets = torch.tensor(df.iloc[1:].values, dtype=torch.float32)

criterion = torch.nn.MSELoss()
optmizer = torch.optim.Adam(net.parameters())
for step in range(10001):
    if step:
        optmizer.zero_grad()
        train_loss.backward()
        optmizer.step()
    preds = net(inputs)
    train_preds = preds[:train_seq_len]
    train_targets = targets[:train_seq_len]
    train_loss = criterion(train_preds, train_targets)

    test_preds = preds[-test_seq_len]
    test_targets = targets[-test_seq_len]
    test_loss = criterion(test_preds, test_targets)

    if step % 500 == 0:
        print('step{}：loss(train)={}，loss(test)={}'.format(step, train_loss, test_loss))

preds = net(inputs)
df_pred = pd.DataFrame(preds.detach().numpy(), index=years[1:], columns=df.columns)

print(df_pred.loc[2001:])