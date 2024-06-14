import cv2
from time import process_time
from image_utils.img_resize import resize
import tensorflow as tf
import torch
import torchvision.transforms as T

img=cv2.imread('/app/data/1678995822014641.gadget2d.png')

N=1000
# Test opencv using cpu
t0=process_time()
for i in range(N):
  resize(img,width=300,device='cpu')
t1=process_time()

# Test opencv using gpu
t2=process_time()
for i in range(N):
  resize(img,width=300,device='gpu')
t3=process_time()

# Test tensorflow using gpu
t4=process_time()
for i in range(N):
  tf.image.resize(img,[179,300]).numpy()
t5=process_time()

# Test torch using gpu
device='cuda:0'
t6=process_time()
for i in range(N):
  tensor = torch.from_numpy(img)
  tensor=tensor.to(device)  
  T.Resize(size=(179,300))(tensor)
t7=process_time()

print(f'CPU proc time: {(t1-t0)/float(N)*1000:.3f} ms')
print(f'GPU-cv2 proc time: {(t3-t2)/float(N)*1000:.3f} ms')
print(f'GPU-tf proc time: {(t5-t4)/float(N)*1000:.3f} ms')
print(f'GPU-torch proc time: {(t7-t6)/float(N)*1000:.3f} ms')


