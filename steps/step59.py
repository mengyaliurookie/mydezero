import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Model

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

seq_data=[np.random.randn(1,1) for _ in range(1000)] # 虚拟的时间序列数据
xs=seq_data[0:-1]
ts=seq_data[1:] # xs的下一个时间步的数据

model=SimpleRNN(10,1)

loss,cnt=0,0
for x,t in zip(xs,ts):
    y=model(x)
    loss+=F.mean_squared_error(y,t)
    cnt+=1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break