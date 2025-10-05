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
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple): #对非元组情况的额外处理
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=outputs
        return outputs if len(outputs)>1 else outputs[0]
    def forward(self,x):
        raise NotImplementedError()

    def backword(self,gy):
        raise  NotImplementedError()
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    

if __name__=="__main__":
    x0=Variable(np.array(2))
    x1=Variable(np.array(3))
    f=Add()
    ys=f(x0,x1)
    # y=ys[0]
    print(ys.data)