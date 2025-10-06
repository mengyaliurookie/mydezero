import numpy as np
import weakref
import contextlib
class Variable:
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data,np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0

    def set_func(self,func):
        self.creator=func
        self.generation=func.generation+1

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n','\n'+" "*9)
        return 'variable('+p+')'

    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()

        def add_func(func):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            # 获取最靠后的函数，然后根据函数的输入输出变量
            f=funcs.pop()
            # 根据输出变量来获取输出变量的梯度
            gys=[output().grad for output in f.outputs]
            # 利用函数f的backward来计算输入变量的梯度
            gxs=f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            # 设置输入变量的梯度
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad+=gx
                # 把输入变量的生成函数添加到优先队列中
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for output in f.outputs:
                    output().grad=None


class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation=max([x.generation for x in inputs]) # 设置辈分
            for output in outputs:
                output.set_func(self) # 设置连接
        self.inputs=inputs
        self.outputs=[weakref.ref(output) for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,gy):
        raise  NotImplementedError()
class Config:
    enable_backprop=True

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config('enable_backprop',False)


class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return y

    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        return gy*x1,gy*x0

def mul(x0,x1):
    return Mul()(x0,x1)






