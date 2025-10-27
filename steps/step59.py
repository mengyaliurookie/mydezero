import numpy as np

import dezero.datasets
import dezero.layers as L
import dezero.functions as F
from dezero import Model
import matplotlib.pyplot as plt


# rnn=L.RNN(10)  # 只指定隐藏层的大小
# x=np.random.rand(1,1)
# h=rnn(x)
# print(h.shape)

class SimpleRNN(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.rnn=L.RNN(hidden_size)
        self.fc=L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self,x):
        h=self.rnn(x)
        y=self.fc(h)
        return y

# seq_data=[np.random.randn(1,1) for _ in range(1000)] # 虚拟的时间序列数据
# xs=seq_data[0:-1]
# ts=seq_data[1:] # xs的下一个时间步的数据
#
# model=SimpleRNN(10,1)
#
# loss,cnt=0,0
# for x,t in zip(xs,ts):
#     y=model(x)
#     loss+=F.mean_squared_error(y,t)
#     cnt+=1
#     if cnt == 2:
#         model.cleargrads()
#         loss.backward()
#         break

max_epoch=100
hidden_size=100
bptt_length=30 # BPTT的长度

train_set=dezero.datasets.SinCurve(train=True)
seqlen=len(train_set)

model=SimpleRNN(hidden_size,1)
optimizer=dezero.optimizers.Adam().setup(model)


# 训练开始
for epoch in range(max_epoch):
    model.reset_state()
    loss,count=0,0

    for x,t in train_set:
        x=x.reshape(1,1) # 1.形状转换为(1,1)
        y=model(x)
        loss+=F.mean_squared_error(y,t)
        count+=1

        # 2.调整Truncated BPTT的时机
        if count%bptt_length==0 or count==seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward() # 3.切断连接
            optimizer.update()

        avg_loss=float(loss.data)/count
        print(f'| epoch {epoch+1} | {count}/{seqlen} batches | loss {avg_loss:.2f}')

xs=np.cos(np.linspace(0,4*np.pi,1000))
model.reset_state() # 重置模型
pred_list=[]

with dezero.no_grad():
    for x in xs:
        x=np.array(x).reshape(1,1)
        y=model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)),xs,label="y=cos(x)")
plt.plot(np.arange(len(xs)),pred_list,label="predict")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


