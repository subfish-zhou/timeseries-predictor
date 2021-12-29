#导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'    
#导入数据
data_origin=pd.read_excel('equipment_loss_rate.xlsx',sheet_name='warship_generate')
data_origin.head()

plt.figure(figsize=(13,5))
plt.plot(data_origin['time'],data_origin['consume'])
plt.show()

print(type(data_origin['consume'].values[0])) #看看数据类型是不是float

data_float=data_origin['consume'].values

#将数据集分为训练数据集和验证数据集
test_data_size=12

data_train=data_float[:-test_data_size] #训练集数据132条
data_test=data_float[-test_data_size:]  #验证数据12条

#数据未经过标准化处理，将数据进行min/max标准化
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(-1,1))
data_train_normalized=scaler.fit_transform(data_train.reshape(-1,1))

#转化为tensor数据格式
data_train_normalized=torch.FloatTensor(data_train_normalized).view(-1)#转化成1维张量

#将数据处理成sequences和对应标签形式
#在这里我们取时间窗口为12，因为一年由12个月，这个是比较合理的
train_window=12

def creat_input_sequences(input_data,tw):
    inout_seq=[]
    L=len(input_data)
    for i in range(L-tw):
        train_seq=input_data[i:i+tw]
        train_lable=input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq,train_lable))
    return inout_seq

train_input_seq=creat_input_sequences(data_train_normalized,train_window)
#一共120个样本（132-12=120）

#创建LSTM模型
class LSTM_timeseq(nn.Module):
    def __init__(self,input_size=1,hidden_layer_size=100,output_size=1):
        super().__init__()

        self.hidden_layer_size=hidden_layer_size
        self.lstm=nn.LSTM(input_size,hidden_layer_size)
        self.linear=nn.Linear(hidden_layer_size,output_size)
        self.hidden_cell=(torch.zeros(1,1,self.hidden_layer_size),torch.zeros(1,1,self.hidden_layer_size))

    def forward(self,input_seq):
        lstm_out,self.hidden_cell=self.lstm(input_seq.view(len(input_seq),1,-1),self.hidden_cell)
        predictions=self.linear(lstm_out.view(len(input_seq),-1))
        return predictions[-1]

    #创建模型类的对象、定义损失函数和优化器
model=LSTM_timeseq()
loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

#调用GPU

#第一个地方是指定训练设备device
#第二个地方是实例化模型时候把模型加到device
#第三个地方是训练时候把特征和标签加到device, 测试时候也同样需要加到device

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device) #模型放到GPU

#查看模型是否在GPU上：
#print(next(model.parameters()).device)
#查看数据是否在GPU上：
#print(data.device)

# 模型的训练
epochs=50

for i in range(epochs):
    for seq,labels in train_input_seq:
        #特征和标签放到GPU
        #seq.to(device)
        #labels.to(device)

        optimizer.zero_grad()
        model.hidden_cell=(torch.zeros(1,1,model.hidden_layer_size),torch.zeros(1,1,model.hidden_layer_size))

        y_pred=model(seq)

        single_loss=loss_function(y_pred,labels)
        single_loss.backward()
        optimizer.step()
    
    print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')

#预测
fut_pre=12

test_inputs=data_train_normalized[-train_window:].tolist()
print(test_inputs)

model.eval()#将模型变为预测模式
for i in range(fut_pre):
    seq=torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden=(torch.zeros(1,1,model.hidden_layer_size),torch.zeros(1,1,model.hidden_layer_size))
        test_inputs.append(model(seq).item())

#预测画图
data_test_normalized=scaler.fit_transform(data_test.reshape(-1,1))

plt.figure(figsize=(13,5))
x=range(len(test_inputs))
plt.plot(x,test_inputs)
plt.plot(x[-len(data_test_normalized):],data_test_normalized)
plt.show()
