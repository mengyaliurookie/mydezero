import dezero
from PIL import Image
import numpy as np
from dezero.models import VGG16
# from steps.step42 import predict

url="https://github.com/oreilly-japan/deep-learning-from-scratch-3/"\
    "raw/images/zebra.jpg"
img_path=dezero.utils.get_file(url)
img=Image.open(img_path)
# img.show()
x=VGG16.preprocess(img)
x=x[np.newaxis] # 增加用于小批量处理的轴

model=VGG16(pretrained=True)
with dezero.test_mode():
    y=model(x)
predict_id=np.argmax(y.data)

model.plot(x,to_file='vgg16.pdf') # 计算图的可视化
labels=dezero.datasets.ImageNet.labels() # ImageNet的标签
print(labels[predict_id])