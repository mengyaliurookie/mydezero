import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt
import matplotlib
# 优先尝试 TkAgg（跨平台），若报错可换 QtAgg（需安装 PyQt5）
matplotlib.use('QtAgg')

x=Variable(np.linspace(-7,7,200))
y=F.sin(x)
y.backward(create_graph=True)

logs=[y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx=x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    # print(x.grad) #n阶导数

# 绘制图像
labels=["y=sin(x)","y'","y''","y'''"]
for i,v in enumerate(logs):
    plt.plot(x.data,logs[i],label=labels[i])
plt.legend(loc="lower right")
plt.show()