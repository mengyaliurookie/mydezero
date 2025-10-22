from dezero.datasets_simple import Spiral
from dezero import DataLoader,models,optimizers
import dezero.functions as F
import dezero

max_epoch=300
batch_size=30
hidden_size=10
lr=1.0

train_set=Spiral(train= True)
test_set=Spiral(train= False)
train_loader=DataLoader(train_set,batch_size)
test_loader=DataLoader(test_set,batch_size,shuffle=False)

# for epoch in range(max_epoch):
#     for x,t in train_loader:
#         print(x.shape,t.shape)
#         break
#
#     for x,t in test_loader:
#         print(x.shape,t.shape)
#         break

model= models.MLP((hidden_size,3))
optimizer=optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss,sum_acc=0,0
    for x,t in train_loader: # 用于训练的小批量数据
        y=model(x)
        loss=F.softmax_cross_entropy_simple(y,t)
        acc=F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
    print(f'epoch: {epoch+1}')
    print(f'train loss: {sum_loss/len(train_set)}, accuracy: {sum_acc/len(train_set)}')

    sum_loss,sum_acc=0,0
    with dezero.no_grad(): # 无梯度模式
        for x,t in test_loader: # 用于测试的小批量数据
            y=model(x)
            loss=F.softmax_cross_entropy_simple(y,t)
            acc=F.accuracy(y,t)
            sum_loss+=float(loss.data)*len(t)
            sum_acc+=float(acc.data)*len(t)
    print(f'test loss: {sum_loss/len(test_set)}, accuracy: {sum_acc/len(test_set)}')
