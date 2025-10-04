import numpy as np

class Variable:
    def __init__(self,data):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self,func):
        self.creator=func


    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop() #获取函数
            x,y=f.input,f.output # 获取函数的输入
            x.grad=f.backward(y.grad) # backward调用backward方法

            if x.creator is not None:
                funcs.append(x.creator) # 将前一个函数添加到列表中

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(as_array(y))
        output.set_creator(self) #让输出变量保存创造者信息
        self.input=input
        self.output=output #也保存输出变量
        return output

    def forward(self,x):
        raise NotImplementedError()

    def backword(self,gy):
        raise  NotImplementedError()

class Square(Function):
    def forward(self,x):
        y=x**2
        return y

    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y

    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__=="__main__":
    A=Square()
    B=Exp()
    C=Square()

    x=Variable(np.array(0.5))
    a=A(x)
    b=B(a)
    y=C(b)

    # 反向遍历计算图的节点
    assert y.creator==C
    assert y.creator.input==b
    assert y.creator.input.creator==B
    assert y.creator.input.creator.input==a
    assert y.creator.input.creator.input.creator==A
    assert y.creator.input.creator.input.creator.input==x

    # 反向传播
    y.grad=np.array(1.0)
    y.backward()
    print(x.grad)








