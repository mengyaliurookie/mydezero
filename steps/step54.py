import numpy as np
from dezero import test_mode
import dezero.functions as F

x=np.ones(5)
print(x)

# 训练时
y=F.dropout(x)
print(y)

# 测试时
with test_mode():
    y=F.dropout(x)
    print(y)

