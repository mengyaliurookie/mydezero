import numpy as np

import dezero
from dezero.core import Function, as_variable, Variable, as_array
from dezero import utils, cuda


class Sin(Function):
    def forward(self,x):
        y=np.sin(x)
        return y

    def backward(self, gy):
        x,=self.inputs
        gx=gy*cos(x)
        return gx

def sin( x):
    return Sin()(x)

class Cos(Function):
    def forward(self,x):
        y=np.cos(x)
        return y

    def backward(self, gy):
        x,=self.inputs
        gx=gy*-sin(x)
        return gx

def cos( x):
    return Cos()(x)

class Tanh(Function):
    def forward(self,x):
        y=np.tanh(x)
        return y

    def backward(self, gy):
        y=self.outputs[0]()
        gx=gy*(1-y*y)
        return gx

def tanh( x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self,shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape
        y=x.reshape(self.shape)
        return y

    def backward(self,gy):
        return reshape(gy,self.x_shape)

def reshape(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)

class BoardcastTo(Function):
    def __init__(self,shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape
        y=np.broadcast_to(x,self.shape)
        return y

    def backward(self,gy):
        gx=sum_to(gy,self.x_shape)
        return gx

def broadcast_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return BoardcastTo(shape)(x)

class SumTo( Function):
    def __init__(self,shape):
        self.shape=shape

    def forward(self, x):
        self.x_shape=x.shape
        y=utils.sum_to(x,self.shape)
        return y

    def backward(self,gy):
        gx=broadcast_to(gy,self.x_shape)
        return gx

def sum_to(x,shape):
    if x.shape==shape:
        return as_variable(x)
    return SumTo(shape)(x)

class Sum(Function):
    def __init__(self,axis,keepdims):
        self.axis=axis
        self.keepdims=keepdims

    def forward(self, x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    def backward(self, gy):
        gy=utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx=broadcast_to(gy,self.x_shape)
        return gx

def sum( x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)

class MatMul(Function):
    def forward(self,x,w):
        y=x.dot(w)
        return y
    def backward(self,gy):
        x,W=self.inputs
        gx=matmul(gy,W.T)
        gW=matmul(x.T,gy)
        return gx,gW

def matmul(x,w):
    return MatMul()(x,w)

class MeanSquaredError(Function):
    def forward(self,x0,x1):
        diff=x0-x1
        y=(diff**2).sum()/len(diff)
        return y
    def backward(self,gy):
        x0,x1=self.inputs
        diff=x0-x1
        gx0=gy*diff*2/len(diff)
        gx1=-gx0
        return gx0,gx1

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*y
        return gx

def exp(x):
    return Exp()(x)

def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)

def linear_simple(x,W,b= None):
    t=matmul(x,W)
    if b is None:
        return t
    y=t+b
    t.data=None # 删除t的数据
    return y

def sigmoid_simple(x):
    x=as_variable(x)
    y=1/(1+exp(-x))
    return y

class GetItem(Function):
    def __init__(self,slices):
        self.slices=slices

    def forward(self,x):
        y=x[self.slices]
        return y

    def backward(self,gy):
        x,=self.inputs
        f=GetItemGrad(self.slices,x.shape)
        gx=f(gy)
        return gx

def get_item(x,slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices,in_shape):
        self.slices=slices
        self.in_shape=in_shape

    def forward(self,gy):
        gx=np.zeros(self.in_shape,dtype=gy.dtype)
        np.add.at(gx,self.slices,gy)
        return gx

    def backward(self,gx):
        return get_item(gx,self.slices)

def softmax_simple(x,axis=1):
    x=as_variable( x)
    y=exp(x)
    sum_y=sum(y,axis=axis,keepdims=True)
    return y/sum_y

def softmax_cross_entropy_simple(x,t):
    x,t=as_variable(x),as_variable(t)
    N=x.shape[0]

    p=softmax_simple(x)
    p=clip(p,1e-15,1.0) #为了防止log(0)，将p设为大于1e-15的值
    log_p=log(p) # 这个log是Dezero函数
    tlog_p=log_p[np.arange(N),t.data]
    y=-1*sum(tlog_p)/N
    return y

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x,self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)



def accuracy(y,t):
    y,t=as_variable(y),as_variable(t)

    pred=y.data.argmax(axis=1).reshape(t.shape)
    result=(pred==t.data)
    acc=result.mean()
    return Variable(as_array(acc))

class ReLU(Function):
    def forward(self,x):
        y=np.maximum(x,0)
        return y

    def backward(self,gy):
        x,=self.inputs
        mask=x.data>0
        gx=gy*mask
        return gx

def relu(x):
    return ReLU()(x)

def dropout(x,dropout_ratio=0.5):
    x=as_variable(x)

    if dezero.Config.train:
        xp=cuda.get_array_module(x)
        mask=xp.random.rand(*x.shape)>dropout_ratio
        scale=xp.array(1.0-dropout_ratio).astype(x.dtype)
        return x*mask/scale
    else:
        return x



from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow