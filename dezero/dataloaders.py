import numpy as np
import math
import random
from dezero import cuda
import dezero


class DataLoader:
    def __init__(self,dataset,batch_size,shuffle=True,gpu=False):
        self.dataset=dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.data_size=len(dataset)
        self.max_iter=math.ceil(self.data_size/batch_size)
        self.gpu=gpu

        self.reset()

    def reset(self):
        self.iteration=0
        if self.shuffle:
            self.index=np.random.permutation(self.data_size)
        else:
            self.index=np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration()

        i,batch_size=self.iteration,self.batch_size
        batch_index=self.index[i*batch_size:(i+1)*batch_size]
        batch=[self.dataset[i] for i in batch_index]
        x=np.array([i[0] for i in batch])
        t=np.array([i[1] for i in batch])

        self.iteration+=1
        return x,t

    def next(self):
        return self.__next__()


class SeqDataLoader(DataLoader):
    def __init__(self,dataset,batch_size,gpu=False):
        super().__init__(dataset=dataset,batch_size=batch_size,shuffle=False,gpu=gpu)

    def __next__(self):
        if self.iteration>=self.max_iter:
            self.reset()
            raise StopIteration()
        jump=self.data_size//self.batch_size
        batch_index=[(i*jump+self.iteration)%self.data_size for i in range(self.batch_size)]
        batch=[self.dataset[i] for i in batch_index]

        xp=cuda.cupy if self.gpu else np
        x=xp.array([example[0] for example in batch])
        t=xp.array([example[1] for example in batch])

        self.iteration+=1
        return x,t

if __name__=='__main__':
    train_set=dezero.datasets.SinCurve(train=True)
    dataloader=SeqDataLoader(train_set,batch_size=3)
    x,t=next(dataloader)
    print(x)
    print("---"*27)
    print(t)
