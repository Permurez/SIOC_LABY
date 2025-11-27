import numpy as np
import matplotlib.pyplot as plt
#zmiana jader uzywanych przy skalowaniu
#wiersz
aaa=1
#kolumna
bbb=2
#ile razy zmniejszamy i powiekszamy
factor=1
img = plt.imread('kamien.jpg')
if len(img.shape) == 3:  
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
# RGB to Grey
Grey_Image = img.astype(np.float64) 
H, W =Grey_Image.shape

def MSE_(image1, image2):
    # Dopasowanie rozmiaru: bierzemy wspólny fragment
    H = min(image1.shape[0], image2.shape[0])
    W = min(image1.shape[1], image2.shape[1])
    
    diff = image1[:H, :W] - image2[:H, :W]
    mse = np.sum(diff**2) / (H * W)
    return mse

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
#Reduce_Image gotowe
def Enlarge_Image(image, factor, kernel_func1, kernel_func2):
    H, W = image.shape
    H_new, W_new = H * factor, W * factor
    enlarged = np.zeros((H_new, W_new))

    x_org = np.arange(W)
    y_org = np.arange(H)
    x_new = np.linspace(0, W - 1, W_new)
    y_new = np.linspace(0, H - 1, H_new)

#wierszowa
    temp_image = np.zeros((H, W_new))
    for i in range(H):
        temp_image[i, :] = interpolacja(x_org, image[i, :], x_new, kernel_func1)

#kolumnowa
    for j in range(W_new):
        enlarged[:, j] = interpolacja(y_org, temp_image[:, j], y_new, kernel_func2)
        
    return enlarged
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
    waga = j(t)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new
jadra = {1: h1, 2: h2, 3: h3, 4: h4}

#ZROBIĆ ZA POMOCA 2 JADER INTERPOLACJE


reduced_image = Reduce_Image(Grey_Image, factor)
reduced_enlarged_image = Enlarge_Image(reduced_image, factor, jadra[aaa], jadra[bbb])  

# Oblicza MSE między oryginalnym obrazem a powiększonym obrazem
mse_value = MSE_(Grey_Image, reduced_enlarged_image)

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(Grey_Image, cmap='gray')
plt.title('Original Grey Image')
plt.axis('off')

import numpy as np
import matplotlib.pyplot as plt
#zmiana jader uzywanych przy skalowaniu
#wiersz
aaa=1
#kolumna
bbb=2
#ile razy zmniejszamy i powiekszamy
factor=1
img = plt.imread('kamien.jpg')
if len(img.shape) == 3:  
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
# RGB to Grey
Grey_Image = img.astype(np.float64) 
H, W =Grey_Image.shape

def MSE_(image1, image2):
    # Dopasowanie rozmiaru: bierzemy wspólny fragment
    H = min(image1.shape[0], image2.shape[0])
    W = min(image1.shape[1], image2.shape[1])
    
    diff = image1[:H, :W] - image2[:H, :W]
    mse = np.sum(diff**2) / (H * W)
    return mse

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
#Reduce_Image gotowe
def Enlarge_Image(image, factor, kernel_func1, kernel_func2):
    H, W = image.shape
    H_new, W_new = H * factor, W * factor
    enlarged = np.zeros((H_new, W_new))

    x_org = np.arange(W)
    y_org = np.arange(H)
    x_new = np.linspace(0, W - 1, W_new)
    y_new = np.linspace(0, H - 1, H_new)

#wierszowa
    temp_image = np.zeros((H, W_new))
    for i in range(H):
        temp_image[i, :] = interpolacja(x_org, image[i, :], x_new, kernel_func1)

#kolumnowa
    for j in range(W_new):
        enlarged[:, j] = interpolacja(y_org, temp_image[:, j], y_new, kernel_func2)
        
    return enlarged
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
    waga = j(t)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new
jadra = {1: h1, 2: h2, 3: h3, 4: h4}

#ZROBIĆ ZA POMOCA 2 JADER INTERPOLACJE


reduced_image = Reduce_Image(Grey_Image, factor)
reduced_enlarged_image = Enlarge_Image(reduced_image, factor, jadra[aaa], jadra[bbb])  

# Oblicza MSE między oryginalnym obrazem a powiększonym obrazem
mse_value = MSE_(Grey_Image, reduced_enlarged_image)

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(Grey_Image, cmap='gray')
plt.title('Original Grey Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reduced_image, cmap='gray')
plt.title(f'Reduced Image (factor={factor})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reduced_enlarged_image, cmap='gray')
plt.title(f'Enlarged Image (h{aaa}/h{bbb}, MSE={mse_value:.2f})')
plt.axis('off')

plt.tight_layout()
plt.show()