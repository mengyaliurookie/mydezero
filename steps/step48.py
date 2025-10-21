import math
import numpy as np
import dezero.datasets
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


# 设置超参数
max_epoch=300
batch_size=30
hidden_size=10
lr=1.0

# 读入数据 / 创建模型和Optimizer
x,t=dezero.datasets.get_spiral(train=True)
model=MLP((hidden_size,3))
optimizer=optimizers.SGD(lr).setup(model)

print("数据集大小：",len(x))

data_size=len(x)
max_iter=math.ceil(data_size/batch_size) #小数点向上取整
for epoch in range(max_epoch):
    # 数据集索引重排
    index=np.random.permutation(data_size)
    sum_loss=0

    for i in range(max_iter):
        # 创建小批量数据
        batch_index=index[i*batch_size:(i+1)*batch_size]
        batch_x=x[batch_index]
        batch_t=t[batch_index]

        # 算出梯度 / 更新参数
        y=model(batch_x)
        loss=F.softmax_cross_entropy_simple(y,batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss+=float(loss.data)*len(batch_t)

    # 输出每轮的训练情况
    avg_loss=sum_loss/data_size
    print('epoch:',epoch+1,'| mean loss:',avg_loss)