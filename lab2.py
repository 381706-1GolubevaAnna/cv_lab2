import numpy as np
import math
import matplotlib.pyplot as plt
import imageio

#ReLU

def ReLU(img):
  rows, columns, channels = img.shape
  res = np.zeros((rows, columns, channels), 'uint8')
  
  for ch in range(channels):
    for r in range(rows):
      for col in range(columns):
        res[r][col][ch] = np.maximum(0, img[r][col][ch])

  return res

#Convolution

def conv(img, fil):  
    rows, columns, channels = img.shape
    n, rows_f, columns_f, channels_f = fil.shape
    res = np.zeros((rows, columns, n), 'uint8')
    img = np.pad(img, ((3,3),(3,3),(0,0)), 'constant')
    
    for n in range(n):
        for col in range(columns):
            for r in range(rows):
                for col_f in range(columns_f):
                  for r_f in range(rows_f):
                    res[r][col][n] += np.dot(fil[n][r_f][col_f][:], img[r+r_f-1][col+col_f-1][:])

    return res

#Max Pooling

def MaxPool(img, res):
  rows, columns, channels = img.shape
  half_size_r=int(rows/2)
  half_size_col=int(columns/2)

  for r in range(half_size_r):
      for col in range(half_size_col):
          max1=np.maximum(img[r*2][col*2],img[r*2][col*2+1])
          max2=np.maximum(img[r*2+1][col*2],img[r*2+1][col*2+1])
          res[r][col][:] = np.maximum(max1,max2)

  return res

image = imageio.imread("res/image.jpg")[:,:,:3]
rows, columns, channels = image.shape
filters = np.full((5, 3, 3, n), 1/7, dtype=float)
res_conv = conv(image, filters)
reLU = ReLU(res_conv)

rows_reLU, columns_reLU, channels_reLU = reLU.shape
half_rows = int(rows_reLU/2)
half_col = int(columns_reLU/2)
tmp = np.zeros((half_rows, half_col, 5), 'uint8')
res = MaxPool(reLU, tmp)


plt.imshow(image); plt.axis('off')
print('Image size: ', image.shape)
print('Max Pooling size: ', res.shape)