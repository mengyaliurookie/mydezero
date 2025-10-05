import numpy as np

from steps.step11 import *


def add(x0,x1):
    return Add()(x0,x1)


if __name__=="__main__":
    x0=Variable(np.array(2))
    x1=Variable(np.array(3))
    y=add(x0,x1)
    print(y.data)