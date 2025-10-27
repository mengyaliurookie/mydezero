import numpy as np
import dezero
import matplotlib.pyplot as plt
from dezero import Model
from dezero import SeqDataLoader
import dezero.functions as F
import dezero.layers as L


# train_set=dezero.datasets.SinCurve(train=True)
# print(len(train_set))
# print(train_set[0])
# print(train_set[1])
# print(train_set[2])
#
# # 绘制图形
# xs=[example[0] for example in train_set]
# ts=[example[1] for example in train_set]
# plt.plot(np.arange(len(xs)),xs,label="xs")
# plt.plot(np.arange(len(ts)),ts,label="ts")
# plt.show()

max_epoch=100
batch_size=30
hidden_size=100
bptt_length=30

train_set=dezero.datasets.SinCurve(train=True)
# 1.使用时间序列数据的数据加载器
dataloader=SeqDataLoader(train_set,batch_size=batch_size)
seqlen=len(train_set)

class BetterRNN(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.rnn=L.LSTM(hidden_size) # 2.使用LSTM
        self.fc=L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self,x):
        h=self.rnn(x)
        y=self.fc(h)
        return y

model=BetterRNN(hidden_size,1)
optimizer=dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss,count=0,0

    for x,t in dataloader:
        y=model(x)
        loss+=F.mean_squared_error(y,t)
        count+=1

        if count%bptt_length==0 or count==seqlen:
            # dezero.utils.plot_dot_graph(loss) # 绘制计算图
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss=loss.data.item()/count
    print(f"epoch:{epoch+1},loss:{avg_loss}")


xs=np.cos(np.linspace(0,4*np.pi,1000))
model.reset_state() # 重置模型
pred_list=[]

with dezero.no_grad():
    for x in xs:
        x=np.array(x).reshape(1,1)
        y=model(x)
        pred_list.append(y.data.item())

plt.plot(np.arange(len(xs)),xs,label="y=cos(x)")
plt.plot(np.arange(len(xs)),pred_list,label="predict")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

