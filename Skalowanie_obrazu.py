import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('grey_photo.jpg')

if len(img.shape) == 3:  
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
# RGB to Grey
Grey_Image = img.astype(np.float64) 
H, W =Grey_Image.shape
factor = 2



def Reduce_Image (img, factor): 
    kernel = np.ones((factor, factor), dtype=np.float32) / (factor * factor)
    H, W = img.shape
    Hc = H - factor + 1
    Wc = W - factor + 1

    conv = np.zeros((Hc, Wc))

    for i in range(Hc):
        for j in range(Wc):
            patch = img[i:i+factor, j:j+factor]
            conv[i, j] = np.sum(patch * kernel)
    return conv[::factor, ::factor]

def Enlarge_Image(image,factor):
    #N macierz
    N=factor*factor
     


def MSE(y_org, y_new):
    N = len(y_org)
    if N == 0:
        return 0
    else:
        return np.sum((y_org - y_new)**2)/N
