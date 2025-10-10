import numpy as np
import dezero.functions as F
from dezero import Variable

x=Variable(np.array([[1.0,2,3],[4,5,6]]))
y=F.sin(x)
print( y)