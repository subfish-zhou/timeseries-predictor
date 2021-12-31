import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from blitz.modules import BayesianLSTM
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
#from blitz.losses import kl_divergence_from_nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from collections import deque

df = pd.read_excel('equipment_loss_rate.xlsx', sheet_name='warship_generate')

# 数据预处理

consume = df['consume']

scaler = StandardScaler()
consume_arr = np.array(consume).reshape(-1, 1)
consume_scale = scaler.fit_transform(consume_arr)  # 标准化
windows_size = 12


def create_timestamps_ds(series, timestep_size=windows_size):
    time_stamps = []
    labels = []
    aux_deque = deque(maxlen=timestep_size)

    # starting the timestep deque
    for i in range(timestep_size):
        aux_deque.append(0)

    # feed the timestamps list
    for i in range(len(series)-1):
        aux_deque.append(series[i])
        time_stamps.append(list(aux_deque))

    # feed the labels list
    for i in range(len(series)-1):
        labels.append(series[i+1])

    assert len(time_stamps) == len(labels), 'something went wrong'

    # torch-tensoring it
    features = torch.tensor(time_stamps[timestep_size:]).float()
    labels = torch.tensor(labels[timestep_size:]).float()

    return features, labels
# 神经网络类


@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lstm_1 = BayesianLSTM(1, 50)

        self.linear = nn.Linear(50, 1)
        #self.linear = BayesianLinear(20, 1)
        

    def forward(self, x):
        x_, _ = self.lstm_1(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_


# 将模型放到cuda上
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Xs, ys = create_timestamps_ds(consume_scale)
X_train, X_test, y_train, y_test = train_test_split(
    Xs, ys, test_size=.15, random_state=42, shuffle=False)

X_train = X_train.clone().detach().requires_grad_(True).to(device)
X_test = X_test.clone().detach().requires_grad_(True).to(device)
y_train = y_train.clone().detach().requires_grad_(True).to(device)
y_test = y_test.clone().detach().requires_grad_(True).to(device)

ds = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds, batch_size=72, shuffle=True)

net = NN().to(device)

# 我们将使用MSE损失函数和学习率维0.001的Adam优化器
criterion = nn.MSELoss()
optimizer1 = optim.Adam(net.parameters(), lr=0.003)
optimizer2 = optim.Adam(net.parameters(), lr=0.0003)

iteration = 0
loss_less=10#保存最小loss值
def train_unit(optimizer):
    global iteration
    global loss_less
    for _, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = net.sample_elbo(
            inputs=datapoints, labels=labels, criterion=criterion, sample_nbr=3)
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 20 == 0:
            preds_test = net(X_test)[:, 0].unsqueeze(1)
            loss_test = criterion(preds_test, y_test)

            #保存最小loss的模型参数
            if loss_test<loss_less:
                loss_less = loss_test
                torch.save(net.state_dict(),'../data/models/model_best.pt')
                print("loss less is {:.4f}".format(loss_less))
            print("Iteration: {} Val-loss: {:.4f}, loss least is {:.4f}".format(str(iteration), loss_test, loss_less))


for epoch in range(2000):
    train_unit(optimizer1)
for epoch in range(2000):
    train_unit(optimizer2)

original = df['consume'][1:][windows_size:]

df_pred = pd.DataFrame(original)
df_pred["Date"] = df.time
df["Date"] = pd.to_datetime(df_pred["Date"])
df_pred = df_pred.reset_index()


def pre_consume_future(X_test, future_length, sample_nbr=10):
    global windows_size
    global X_train
    global Xs
    global scaler

    # creating auxiliar variables for future prediction
    preds_test = []
    test_begin = X_test[0:1, :, :]
    test_deque = deque(test_begin[0, :, 0].tolist(), maxlen=windows_size)

    idx_pred = np.arange(len(X_train), len(Xs))

    # predict it and append to list
    for i in range(len(X_test)):
        # print(i)
        as_net_input = torch.tensor(test_deque).unsqueeze(
            0).unsqueeze(2).to(device)
        pred = [net(as_net_input).item() for i in range(sample_nbr)]

        test_deque.append(torch.tensor(pred).mean().item())
        preds_test.append(pred)

        if i % future_length == 0:
            # our inptus become the i index of our X_test
            # That tweak just helps us with shape issues
            test_begin = X_test[i:i+1, :, :]
            test_deque = deque(
                test_begin[0, :, 0].tolist(), maxlen=windows_size)

    #preds_test = np.array(preds_test).reshape(-1, 1)
    #preds_test_unscaled = scaler.inverse_transform(preds_test)

    return idx_pred, preds_test


def get_confidence_intervals(preds_test, ci_multiplier):
    global scaler

    preds_test = torch.tensor(preds_test)

    pred_mean = preds_test.mean(1)
    pred_std = preds_test.std(1).detach().numpy()

    pred_std = torch.tensor((pred_std))

    upper_bound = pred_mean + (pred_std * ci_multiplier)
    lower_bound = pred_mean - (pred_std * ci_multiplier)
    # gather unscaled confidence intervals

    pred_mean_final = pred_mean.unsqueeze(1).detach().numpy()
    pred_mean_unscaled = scaler.inverse_transform(pred_mean_final)

    upper_bound_unscaled = upper_bound.unsqueeze(1).detach().numpy()
    upper_bound_unscaled = scaler.inverse_transform(upper_bound_unscaled)

    lower_bound_unscaled = lower_bound.unsqueeze(1).detach().numpy()
    lower_bound_unscaled = scaler.inverse_transform(lower_bound_unscaled)

    return pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled

# X_test=X_test.to('cpu')
# net.to('cpu')

#加载最优模型

net=NN().to(device)
net.load_state_dict(torch.load('../data/models/model_best.pt'))

future_length = 1
sample_nbr = 4
ci_multiplier = 1
idx_pred, preds_test = pre_consume_future(X_test, future_length, sample_nbr)
pred_mean_unscaled, upper_bound_unscaled, lower_bound_unscaled = get_confidence_intervals(preds_test,
                                                                                          ci_multiplier)
y = np.array(df.consume[-20:]).reshape(-1, 1)  # siifish:改变了-750
under_upper = upper_bound_unscaled > y
over_lower = lower_bound_unscaled < y
total = (under_upper == over_lower)

print("{} our predictions are in our confidence interval".format(np.mean(total)))

plt.figure()

params = {"ytick.color": "w",
          "xtick.color": "w",
          "axes.labelcolor": "w",
          "axes.edgecolor": "w"}
plt.rcParams.update(params)

plt.title("IBM Stock prices", color="white")

plt.plot(df_pred.index,
         df_pred.consume,
         color='black',
         label="Real")

plt.plot(idx_pred,
         pred_mean_unscaled,
         label="Prediction for {} days, than consult".format(future_length),
         color="red")

plt.fill_between(x=idx_pred,
                 y1=upper_bound_unscaled[:, 0],
                 y2=lower_bound_unscaled[:, 0],
                 facecolor='green',
                 label="Confidence interval",
                 alpha=0.5)

plt.legend()
plt.show()