import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color

# wczytuje obraz
image_rgb = data.astronaut()
image_rgb = image_rgb.astype(float) / 255.0
image = color.rgb2gray(image_rgb)

# Operatory Sobela i Laplace'a
Sx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Sy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
L = np.array([[0, 1, 0],
              [1, -4, 1],
              [0, 1, 0]])     

# Jadro do rozmycia 
G = 1/16 * np.array([[1, 2, 1],
                     [1, 4, 1],
                     [1, 2, 1]])

# Jądro do wyostrzania
W = np.array([[ 0, -1, 0],
              [ -1,  5, -1],
              [ 0, -1, 0]])


K_green = np.array([[0, 1, 0],
                     [1, 4, 1],
                    [0, 1, 0]]) / 4
K_rb    = np.array([[1, 2, 1],
                     [2, 4, 2],
                       [1, 2, 1]]) / 4

def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    #padding zerami
    padded_image = np.pad(image,
                          ((pad_height, pad_height),
                           (pad_width, pad_width)),
                          mode='constant', constant_values=0)
    
    convolved_image = np.zeros_like(image)

    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved_image[i, j] = np.sum(region * kernel)

    return convolved_image

def wykrywanie_krawiedzi(image):
    "Wykrywanie krawędzi (Sobel)"
    sobel_x = convolve2d(image, Sx)
    sobel_y = convolve2d(image, Sy)
    edges_image = np.sqrt(sobel_x**2 + sobel_y**2)
    return edges_image


def wykrywanie_laplace(image):
    return np.abs(convolve2d(image, L))

def rozmycie_obraz(image):
    return convolve2d(image, G)

def wyostrzanie_obraz(image):
    return np.clip(convolve2d(image, W), 0, 1)

def symulacja_mozaiki_bayera(img_rgb):
    h, w, _ = img_rgb.shape
    mosaic = np.zeros((h, w))
    mosaic[0::2, 0::2] = img_rgb[0::2, 0::2, 0] 
    mosaic[0::2, 1::2] = img_rgb[0::2, 1::2, 1] 
    mosaic[1::2, 0::2] = img_rgb[1::2, 0::2, 1] 
    mosaic[1::2, 1::2] = img_rgb[1::2, 1::2, 2] 
    return mosaic

def rekonstrukcja_mozaiki(mosaic):
    h, w = mosaic.shape
    R, G, B = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
    R[0::2, 0::2] = mosaic[0::2, 0::2]
    G[0::2, 1::2] = mosaic[0::2, 1::2]; G[1::2, 0::2] = mosaic[1::2, 0::2]
    B[1::2, 1::2] = mosaic[1::2, 1::2]

    R_rec = convolve2d(R, K_rb)
    G_rec = convolve2d(G, K_green)
    B_rec = convolve2d(B, K_rb)
    
    return np.clip(np.dstack((R_rec, G_rec, B_rec)), 0, 1)

def demozaikowanie(img_rgb):
    mosaic = symulacja_mozaiki_bayera(img_rgb)
    return rekonstrukcja_mozaiki(mosaic)

image_sobel = wykrywanie_krawiedzi(image)
image_laplace = wykrywanie_laplace(image) 
image_blur = rozmycie_obraz(image)
image_sharp = wyostrzanie_obraz(image)
image_demosaic = demozaikowanie(image_rgb) 

fig, axes = plt.subplots(2, 4, figsize=(15, 10))

axes[0, 0].imshow(image_blur, cmap='gray'); axes[0, 0].set_title('Rozmycie')
axes[0, 1].imshow(image_sharp, cmap='gray'); axes[0, 1].set_title('Wyostrzanie')
axes[0, 2].imshow(image_rgb); axes[0, 2].set_title('Oryginał (kolor)')
axes[0, 3].imshow(image_demosaic); axes[0, 3].set_title('Demozaikowanie')


axes[1, 0].imshow(image_sobel, cmap='gray'); axes[1, 0].set_title('Sobel: Suma detekcji X i Y')
axes[1, 1].imshow(image_laplace, cmap='gray'); axes[1, 1].set_title('Krawędzie: Laplace')
axes[1, 2].imshow(image_rgb); axes[1, 2].set_title('Oryginał (kolor)')

for ax in axes.flat: ax.axis('off')
plt.tight_layout()
plt.show()

