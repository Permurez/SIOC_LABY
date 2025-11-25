import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('kamien.jpg')
if len(img.shape) == 3:  
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
# RGB to Grey
Grey_Image = img.astype(np.float64) 
H, W =Grey_Image.shape
factor = 10


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
    reduced = conv[::factor, ::factor]
    return reduced


   
#z poprzedniego ćwicznenia
def h1(t):
    return np.where((t >=0) & (t<=1), 1, 0)
def h2(t):
    return np.where((t >=-0.5) & (t <=0.5), 1,0)
def h3(t):
    return np.where((t >= -1) & (t <= 1), 1 - np.abs(t), 0)
def h4(t):
    return np.where (t !=0, np.sin(t)/t, 1)

def interpolacja(x_org, y_org, x_new, j):
    dx = x_org[1] - x_org[0]
    t = (x_new[:, None] - x_org[None, :]) / dx
    waga = j(t).astype(float)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new
jadra = {1: h1, 2: h2, 3: h3, 4: h4}

def Enlarge_Image(reduced_img, factor):
    H, W = reduced_img.shape
    new_H, new_W = H * factor, W * factor
    enlarged_img = np.zeros((new_H, new_W), dtype=reduced_img.dtype)

    for i in range(H):
        for j in range(W):
            enlarged_img[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = reduced_img[i, j]

    return enlarged_img

reduced_img = Reduce_Image(Grey_Image, factor)
final_img = Enlarge_Image(reduced_img, factor)
def MSE_(img1, img2):
    H1, W1 = img1.shape
    H2, W2 = img2.shape
    H = min(H1, H2)
    W = min(W1, W2)
    mse = np.sum((img1[:H, :W] - img2[:H, :W]) ** 2) / (H * W)
    return mse
# Oblicz MSE między oryginalnym obrazem a powiększonym obrazem
mse_value = MSE_(Grey_Image, final_img)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(Grey_Image, cmap='gray')
plt.title('Original Grey Image')
plt.subplot(1, 3, 3)
plt.imshow(final_img, cmap='gray')
plt.title(f'Enlarged Image (MSE={mse_value:.2f})')
plt.subplot(1, 3, 3)

plt.tight_layout()
plt.show()