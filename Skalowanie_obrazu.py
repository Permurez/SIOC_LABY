import numpy as np
import matplotlib.pyplot as plt


Grey_Image='grey_photo.jpg'
Reduction_Factor=2
Enlargment_Factor=2
Kernel_Size= 2 
M=Kernel_Size
H,W = Grey_Image.shape
def Reduce_Image (Image,Reduction_Factor, Kernel_Size):
    N = Reduction_Factor * Reduction_Factor
    kernel= np.full((Reduction_Factor, Reduction_Factor), 1 / N, dtype=np.float32) 

image_float = Grey_Image.astype(np.float32)

H_conv = H - M +1
W_conv = W - M +1

obraz_usrednoniony = np.zeros((H_conv, W_conv), dtype=np.float32)
for i in range(H_conv):
    for j in range(W_conv):
        patch = image_float[i:i+Reduction_Factor, j:j+Reduction_Factor]

def Enlarge_Image(image,factor):
    #N macierz
    N=factor*factor 
     


def MSE(y_org, y_new):
    N = len(y_org)
    if N == 0:
        return 0
    else:
        return np.sum((y_org - y_new)**2)/N
