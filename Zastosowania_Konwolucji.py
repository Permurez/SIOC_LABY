import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
image= data.astronaut()
image = color.rgb2gray(image)
#Operatory Sobela i Laplace'a  do wykrwywania krawędzi
Sx= np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Sy= np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
L=np.array([[0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]])     
#Jądro do rozmycia 
G=1/16 * np.array([[1, 2, 1],
                   [1, 4, 1],
                   [1, 2, 1]])
#Jądro do wyostrzania
W=np.array([[ 0, -1, 0],
            [ -1,  5, -1],
            [ 0, -1, 0]])
def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # padding odbiciem
    padded_image = np.pad(image,
                          ((pad_height, pad_height),
                           (pad_width, pad_width)),
                          mode='reflect')
    
    convolved_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            convolved_image[i, j] = np.sum(region * kernel)

    return convolved_image


def wykrywanie_krawiedzi(image):
    "Wykrywanie krawędzi"
    sobel_x = convolve2d(image, Sx)
    sobel_y = convolve2d(image, Sy)
    edges_image = np.sqrt(sobel_x**2 + sobel_y**2)
    return edges_image

def rozmycie_obraz(image):
    return convolve2d(image, G)

def wyostrzanie_obraz(image):
    return convolve2d(image, W)
image_after_edge_detection = wykrywanie_krawiedzi(image)
image_after_blurring = rozmycie_obraz(image)
image_after_sharpening = wyostrzanie_obraz(image)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Oryginaly obraz')
axes[0].axis('off')


axes[3].imshow(image_after_edge_detection, cmap='gray')
axes[3].set_title('Wykrywanie krawędzi')
axes[3].axis('off')

axes[1].imshow(image_after_blurring, cmap='gray')
axes[1].set_title('Rozmycie')
axes[1].axis('off')

axes[2].imshow(image_after_sharpening, cmap='gray')
axes[2].set_title('Wyostrzanie')
axes[2].axis('off')

plt.tight_layout()
plt.show()
