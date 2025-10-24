import numpy as np
import weakref
import contextlib

# import dezero.functions


class Variable:
    __array_priority__=200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_func(self, func):
        self.creator = func
        self.generation = func.generation + 1

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
        p = str(self.data).replace('\n', '\n' + " " * 9)
        return 'variable(' + p + ')'

    def __mul__(self, other):
        return mul(self,other)

    def backward(self, retain_grad=False,create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(func):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            # 获取最靠后的函数，然后根据函数的输入输出变量
            f = funcs.pop()
            # 根据输出变量来获取输出变量的梯度
            gys = [output().grad for output in f.outputs]
            # 利用函数f的backward来计算输入变量的梯度
            with using_config('enable_backprop',create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                # 设置输入变量的梯度
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    # 把输入变量的生成函数添加到优先队列中
                    if x.creator is not None:
                        add_func(x.creator)
            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

    def cleargrad(self):
        self.grad = None

    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0], (tuple,list)):
            shape = shape[0]
        import dezero.functions
        return dezero.functions.reshape(self,shape)

    def transpose(self, *axes):
        import dezero.functions
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)
    # def transpose(self):
    #     import dezero.functions
    #     return dezero.functions.transpose(self)

    @property
    def T(self):
        import dezero.functions
        return dezero.functions.transpose(self)

    def sum(self,axis=None,keepdims=False):
        import dezero.functions
        return dezero.functions.sum(self,axis,keepdims)


class Function:
    def __call__(self, *inputs):
        inputs=[as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])  # 设置辈分
            for output in outputs:
                output.set_func(self)  # 设置连接
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Parameter(Variable):
    pass

class Config:
    enable_backprop = True
    train=True

def test_mode():
    return using_config('train',False)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        if self.x0_shape != self.x1_shape:
            import dezero.functions
            gx0=dezero.functions.sum_to(gy* x1,self.x0_shape)
            gx1=dezero.functions.sum_to(gy* x0,self.x1_shape)
        else:
            gx0,gx1=gy* x1,gy* x0
        return gx0 , gx1

class Add(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0+x1
        return y

    def backward(self,gy):
        gx0,gx1=gy,gy
        if self.x0_shape != self.x1_shape:
            import dezero.functions
            gx0=dezero.functions.sum_to(gx0,self.x0_shape)
            gx1=dezero.functions.sum_to(gx1,self.x1_shape)
        return gx0,gx1

class Neg(Function):
    def forward(self,x):
        return -x

    def backward(self,gy):
        return -gy

def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0-x1
        return y

    def backward(self,gy):
        gx0, gx1=gy, -gy
        if self.x0_shape != self.x1_shape:
            import dezero.functions
            gx0=dezero.functions.sum_to(gx0,self.x0_shape)
            gx1=dezero.functions.sum_to(gx1,self.x1_shape)
        return gx0,gx1

def sub(x0,x1):
    x1=as_array(x1)
    return Sub()(x0,x1)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)

class Div(Function):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0/x1
        return y

    def backward(self,gy):
        x0,x1=self.inputs
        if self.x0_shape != self.x1_shape:
            import dezero.functions
            gx0=dezero.functions.sum_to(gy/x1,self.x0_shape)
            gx1=dezero.functions.sum_to(gy*(-x0/x1**2),self.x1_shape)
        else:
            gx0=gy/x1
            gx1=gy*(-x0/x1**2)
        gx0=gx0
        gx1=gx1
        return gx0,gx1

def div(x0,x1):
    x1=as_array(x1)
    return Div()(x0,x1)

def rdiv(x0,x1):
    x1=as_array(x1)
    return Div()(x1,x0)



class Pow(Function):
    def __init__(self,c):
        self.c=c

    def forward(self,x):
        y=x**self.c
        return y

    def backward(self,gy):
        x=self.inputs[0]
        c=self.c
        gx=c*x**(c-1)*gy
        return gx

def pow(x,c):
    return Pow(c)(x)



def mul(x0, x1):
    return Mul()(x0, x1)

def add(x0,x1):
    x1=as_array(x1)
    return Add()(x0,x1)

def as_variable(obj):
    if isinstance(obj,Variable):
        return obj
    return Variable(as_array(obj))



def setup_variable():
    import dezero.functions
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__pow__ = pow
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__rsub__ = rsub
    Variable.__sub__ = sub
    Variable.__neg__ = neg
    Variable.__getitem__ = dezero.functions.get_item
